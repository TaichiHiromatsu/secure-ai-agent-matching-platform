from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
from websockets.sync.client import connect


pytestmark = [pytest.mark.browser, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[2]
EDGE_BASE = "http://127.0.0.1:8080"
MARKETPLACE_DB = Path("/app/payment-data/marketplace.db")
PAYMENT_DB = Path("/app/payment-data/paid-agent.db")
MERCHANT_TEST_FAULT_URL = (
    "http://127.0.0.1:8005/internal/test/faults/fulfillment-rejection"
)
MERCHANT_TEST_FAULT_SECRET = "browser-e2e-merchant-fault-key-20260817"
EVIDENCE_INTERACTIONS = [
    "paid payment booking",
    "承認",
    "承認",
    "refresh",
    "hotel search",
    "承認",
    "paid refund-required booking",
    "承認",
    "承認",
    "承認",
    "paid privacy booking",
    "承認",
]
PRIVATE_MARKERS = (
    "checkoutjwt",
    "privatepaymentmaterial",
    "_secureverifiedidentityassertion",
    "secure-verified-identity+jwt",
    "x-verified-identity",
    "begin private key",
    "assertion",
    "secret",
)
_CASE_RESULTS: dict[str, bool] = {}


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for(predicate, *, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except (httpx.HTTPError, KeyError, OSError):
            pass
        time.sleep(0.1)
    raise AssertionError(f"browser condition timed out; last={last!r}")


class _ChromeDevTools:
    def __init__(self, websocket_url: str) -> None:
        self.socket = connect(websocket_url, open_timeout=10)
        self.next_id = 0
        self.events: list[dict[str, Any]] = []

    def close(self) -> None:
        self.socket.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        request_id = self.next_id
        self.socket.send(
            json.dumps({"id": request_id, "method": method, "params": params or {}})
        )
        while True:
            message = json.loads(self.socket.recv(timeout=30))
            if message.get("id") == request_id:
                if "error" in message:
                    raise AssertionError(message["error"])
                return message.get("result", {})
            self.events.append(message)

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result["result"].get("value")

    def body_text(self) -> str:
        return str(self.evaluate("document.body.innerText"))

    def fetch_json(self, path: str) -> dict[str, Any] | None:
        value = self.evaluate(
            "fetch(%s, {credentials: 'same-origin'})"
            ".then(async response => ({status: response.status, body: await response.json()}))"
            % json.dumps(path)
        )
        assert value["status"] == 200, value
        return value["body"]

    def resource_timing(self) -> str:
        return str(
            self.evaluate(
                "JSON.stringify(performance.getEntriesByType('resource').map(entry => ({"
                "name: entry.name, initiatorType: entry.initiatorType,"
                "duration: entry.duration, transferSize: entry.transferSize})))"
            )
        )

    def network_audit(self) -> tuple[str, tuple[str, ...]]:
        # A harmless evaluation flushes queued Network/Runtime events into
        # ``self.events`` before the snapshot is constructed.
        self.evaluate("true")
        requests: dict[str, dict[str, Any]] = {}
        responses: dict[str, dict[str, Any]] = {}
        finished: set[str] = set()
        console: list[dict[str, Any]] = []
        for event in self.events:
            method = event.get("method")
            params = event.get("params", {})
            request_id = str(params.get("requestId", ""))
            if method == "Network.requestWillBeSent":
                requests[request_id] = params.get("request", {})
            elif method == "Network.responseReceived":
                responses[request_id] = params.get("response", {})
            elif method == "Network.loadingFinished":
                finished.add(request_id)
            elif method in {"Runtime.consoleAPICalled", "Log.entryAdded"}:
                console.append(event)

        bodies: list[dict[str, Any]] = []
        for request_id in sorted(finished & responses.keys()):
            response = responses[request_id]
            url = str(response.get("url", ""))
            if not any(
                marker in url
                for marker in (
                    "/auth/browser-bootstrap",
                    "/auth/deployment",
                    "/apps/payment_user_agent/",
                    "/run",
                    "/mediation-api/",
                )
            ):
                continue
            try:
                response_body = self.call(
                    "Network.getResponseBody", {"requestId": request_id}
                ).get("body", "")
            except AssertionError:
                response_body = ""
            request = requests.get(request_id, {})
            bodies.append(
                {
                    "url": url,
                    "method": request.get("method"),
                    "requestBody": request.get("postData", ""),
                    "status": response.get("status"),
                    "responseBody": response_body,
                }
            )
        request_urls = tuple(
            str(request.get("url", "")) for request in requests.values()
        )
        return json.dumps(
            {"networkBodies": bodies, "console": console},
            ensure_ascii=False,
            sort_keys=True,
        ), request_urls


@dataclass(slots=True)
class _BrowserSession:
    chrome: _ChromeDevTools
    process: subprocess.Popen[bytes]

    def close(self) -> None:
        self.chrome.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


@dataclass(frozen=True, slots=True)
class _PackagedStack:
    base_url: str
    log_path: Path


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@pytest.fixture(scope="module")
def packaged_stack(tmp_path_factory: pytest.TempPathFactory) -> _PackagedStack:
    start_script = Path("/app/start.sh")
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    assert start_script.is_file(), "browser suite must run inside the release image"
    assert chromium is not None, "release image must contain Chromium"

    # These are explicit image-local test paths. Removing stale SQLite/PID files
    # makes a repeated browser-suite run deterministic without touching a mount.
    for file_path in (
        MARKETPLACE_DB,
        Path(f"{MARKETPLACE_DB}-wal"),
        Path(f"{MARKETPLACE_DB}-shm"),
        PAYMENT_DB,
        Path(f"{PAYMENT_DB}-wal"),
        Path(f"{PAYMENT_DB}-shm"),
        Path("/app/payment-evidence/evidence.db"),
        Path("/app/payment-evidence/evidence.db-wal"),
        Path("/app/payment-evidence/evidence.db-shm"),
        Path("/var/run/supervisord.pid"),
        Path("/var/run/nginx.pid"),
    ):
        file_path.unlink(missing_ok=True)

    temporary = tmp_path_factory.mktemp("packaged-browser-stack")
    log_path = temporary / "packaged-stack.log"
    environment = os.environ.copy()
    environment.update(
        {
            "DEV_MODE": "true",
            "APP_ENV": "local",
            "MEDIATION_CALLBACK_MODE": "deterministic-local",
            "MEDIATION_LOCAL_AGENT_MODE": "deterministic",
            "MEDIATION_TEST_FAULTS": "true",
            "MEDIATION_TEST_FAULT_SECRET": MERCHANT_TEST_FAULT_SECRET,
            "EPHEMERAL_CLOUD_RUN_DEMO": "true",
            "MEDIATION_STORE_MODE": "memory",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "gen-lang-client-0585901015",
            "GOOGLE_CLOUD_LOCATION": "global",
            "WANDB_DISABLED": "true",
        }
    )
    environment.pop("GOOGLE_API_KEY", None)
    environment.pop("GEMINI_API_KEY", None)
    assert {
        key: environment[key]
        for key in (
            "APP_ENV",
            "DEV_MODE",
            "EPHEMERAL_CLOUD_RUN_DEMO",
            "MEDIATION_STORE_MODE",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
        )
    } == {
        "APP_ENV": "local",
        "DEV_MODE": "true",
        "EPHEMERAL_CLOUD_RUN_DEMO": "true",
        "MEDIATION_STORE_MODE": "memory",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": "gen-lang-client-0585901015",
        "GOOGLE_CLOUD_LOCATION": "global",
    }
    with log_path.open("wb") as output:
        process = subprocess.Popen(
            [str(start_script)],
            cwd="/app",
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        _wait_for(
            lambda: process.poll() is None
            and httpx.get(f"{EDGE_BASE}/health", timeout=1).status_code == 200,
            timeout=30,
        )
        ready = _wait_for(
            lambda: (
                response
                if (response := httpx.get(
                    f"{EDGE_BASE}/mediation-api/ready", timeout=2
                )).status_code
                == 200
                else None
            ),
            timeout=30,
        )
        assert ready.json()["checks"]["routeIsolation"] is True
        assert ready.json()["checks"]["mediationComposition"] is True
        assert ready.json()["mediationStore"] == {
            "mode": "memory",
            "durabilityProfile": "ephemeral-demo",
            "schemaVersion": None,
            "writable": True,
            "decryptable": True,
        }
        yield _PackagedStack(base_url=EDGE_BASE, log_path=log_path)
    except Exception as error:
        diagnostic = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        raise AssertionError(f"packaged browser stack failed: {error}\n{diagnostic}") from error
    finally:
        _terminate_process_group(process)


@pytest.fixture
def browser_session(packaged_stack: _PackagedStack, tmp_path: Path) -> _BrowserSession:
    del packaged_stack
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    assert chromium is not None
    debug_port = _free_port()
    process = subprocess.Popen(
        [
            chromium,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-component-update",
            "--no-first-run",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={debug_port}",
            f"--user-data-dir={tmp_path / 'chrome-profile'}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    session: _BrowserSession | None = None
    try:
        target = _wait_for(
            lambda: next(
                (
                    item
                    for item in httpx.get(
                        f"http://127.0.0.1:{debug_port}/json", timeout=1
                    ).json()
                    if item.get("type") == "page"
                ),
                None,
            )
        )
        chrome = _ChromeDevTools(str(target["webSocketDebuggerUrl"]))
        session = _BrowserSession(chrome=chrome, process=process)
        chrome.call("Runtime.enable")
        chrome.call("Log.enable")
        chrome.call("Network.enable", {"maxTotalBufferSize": 20_000_000})
        chrome.call("Page.enable")
        chrome.call("Page.navigate", {"url": EDGE_BASE})
        assert _wait_for(
            lambda: chrome.evaluate(
                'Boolean(document.querySelector(\'textarea[placeholder="Type a Message..."]\'))'
            )
        )
        bootstrap = _wait_for(
            lambda: chrome.evaluate(
                "window.__secureMediationBrowserReady || null"
            )
        )
        assert bootstrap == {"subject": "demo-local-user"}
        assert "payment_user_agent" in chrome.body_text()
        yield session
    finally:
        if session is not None:
            session.close()
        elif process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def _submit_message(chrome: _ChromeDevTools, message: str) -> None:
    encoded = json.dumps(message)
    assert chrome.evaluate(
        """
        (() => {
          const element = document.querySelector('textarea[placeholder="Type a Message..."]');
          if (!element) return false;
          const setter = Object.getOwnPropertyDescriptor(
            HTMLTextAreaElement.prototype, 'value').set;
          setter.call(element, %s);
          element.dispatchEvent(new Event('input', {bubbles: true}));
          element.focus();
          return true;
        })()
        """ % encoded
    )
    chrome.call(
        "Input.dispatchKeyEvent",
        {
            "type": "rawKeyDown",
            "key": "Enter",
            "code": "Enter",
            "windowsVirtualKeyCode": 13,
            "nativeVirtualKeyCode": 13,
        },
    )
    chrome.call(
        "Input.dispatchKeyEvent",
        {
            "type": "keyUp",
            "key": "Enter",
            "code": "Enter",
            "windowsVirtualKeyCode": 13,
            "nativeVirtualKeyCode": 13,
        },
    )


def _submit_and_wait(
    chrome: _ChromeDevTools, message: str, state: str, *, timeout: float = 30
) -> str:
    _submit_message(chrome, message)
    try:
        return str(
            _wait_for(
                lambda: (
                    body
                    if f"状態: {state}" in (body := chrome.body_text())
                    else None
                ),
                timeout=timeout,
            )
        )
    except AssertionError as error:
        audit, _ = chrome.network_audit()
        view = chrome.fetch_json("/mediation-api/v1/view")
        raise AssertionError(
            f"browser state {state!r} did not appear; "
            f"view={view!r}; body={chrome.body_text()!r}; audit={audit}"
        ) from error


def _assert_plan_target(text: str) -> None:
    assert "状態: WaitingForPlanApproval" in text
    assert '"approvalKind":"plan"' in text
    assert '"approvalToken":"承認"' in text
    assert "承認対象digest: sha256:" in text


def _assert_payment_target(text: str) -> None:
    assert "状態: WaitingForPaymentApproval" in text
    assert '"approvalKind":"payment"' in text
    assert '"distinctFromPlanApproval":true' in text
    assert '"amountMinor":1250' in text
    assert '"currency":"USD"' in text
    assert '"approvalToken":"承認"' in text
    assert "承認対象digest: sha256:" in text


def _assert_callback_order(view: dict[str, Any], *, minimum_operations: int) -> None:
    stages_by_operation: dict[str, list[str]] = {}
    for event in view["trace"]:
        stages_by_operation.setdefault(event["operationId"], []).append(event["stage"])
    callback_operations = 0
    for stages in stages_by_operation.values():
        if "legacy-callback-before" not in stages:
            continue
        callback_operations += 1
        assert stages.count("legacy-callback-before") == 1
        assert stages.count("legacy-callback-after") == 1
        before = stages.index("legacy-callback-before")
        after = stages.index("legacy-callback-after")
        assert before < stages.index("transport") < stages.index("response-persisted") < after
    assert callback_operations >= minimum_operations


def _mediation_session_id(view: dict[str, Any]) -> str:
    values = {
        str(event["operationId"])
        for event in view["trace"]
        if event.get("stage") == "plan-created"
    }
    assert len(values) == 1, values
    return values.pop()


def _refund_fault_target(mediation_session_id: str) -> dict[str, str]:
    with sqlite3.connect(
        f"file:{MARKETPLACE_DB}?mode=ro", uri=True, timeout=10
    ) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT continuation_id,task_id,order_id,state "
            "FROM payment_continuations_v3 WHERE mediation_session_id=?",
            (mediation_session_id,),
        ).fetchone()
    assert row is not None
    assert row["state"] == "waiting_for_payment_approval"
    continuation_id = row["continuation_id"]
    task_id = row["task_id"]
    order_id = row["order_id"]
    assert all(
        isinstance(value, str) and value
        for value in (continuation_id, task_id, order_id)
    )
    target = {
        "orderId": order_id,
        "taskId": task_id,
        "operationId": f"fulfillment-commit:{continuation_id}:1",
    }
    return target


def _arm_refund_fault(target: dict[str, str]) -> None:
    with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
        response = client.post(
            MERCHANT_TEST_FAULT_URL,
            headers={"X-Mediation-Test-Fault-Secret": MERCHANT_TEST_FAULT_SECRET},
            json=target,
        )
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "armed", "target": target}


def _assert_refund_fault_consumed(target: dict[str, str]) -> None:
    with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
        response = client.get(
            MERCHANT_TEST_FAULT_URL,
            headers={"X-Mediation-Test-Fault-Secret": MERCHANT_TEST_FAULT_SECRET},
        )
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "consumed"
    assert value["target"] == target
    assert [event["event"] for event in value["audit"]].count("consumed") == 1
    assert value["audit"][-2]["event"] == "armed"
    assert value["audit"][-1]["event"] == "consumed"
    assert all(event["target"] == target for event in value["audit"][-2:])


def _assert_merchant_db_integrity() -> None:
    with sqlite3.connect(
        f"file:{PAYMENT_DB}?mode=ro", uri=True, timeout=10
    ) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)


def _write_browser_evidence() -> None:
    evidence_path = os.environ.get("BROWSER_EVIDENCE_OUTPUT")
    if not evidence_path:
        return
    assert _CASE_RESULTS == {
        "paid": True,
        "free": True,
        "refund": True,
        "privacy": True,
    }
    destination = Path(evidence_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schemaVersion": "ap2-x402-browser-evidence/2",
                "status": "PASS",
                "imageDigest": os.environ.get("RELEASE_IMAGE_DIGEST", "UNSET"),
                "releaseManifestDigest": "sha256:"
                + hashlib.sha256(
                    (ROOT / "tests/release/release_manifest.json").read_bytes()
                ).hexdigest(),
                "browser": "chromium-cdp-real-ui",
                "listApps": ["payment_user_agent"],
                "appSelected": "payment_user_agent",
                "interactions": EVIDENCE_INTERACTIONS,
                "cases": sorted(_CASE_RESULTS),
                "productionEdge": "nginx-auth-csrf-adk-workflow",
                "completedAfterRefresh": True,
                "privateMaterialExposed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_01_paid_two_exact_approvals_and_refresh(
    browser_session: _BrowserSession,
) -> None:
    chrome = browser_session.chrome
    planned = _submit_and_wait(chrome, "paid payment booking", "WaitingForPlanApproval")
    _assert_plan_target(planned)

    payment = _submit_and_wait(chrome, "承認", "WaitingForPaymentApproval")
    _assert_payment_target(payment)

    completed = _submit_and_wait(chrome, "承認", "Completed")
    assert "シミュレーション: x402 wire-shape fixture (NOT CONFORMANT)" in completed
    before_refresh = chrome.fetch_json("/mediation-api/v1/view")
    assert before_refresh is not None and before_refresh["state"] == "Completed"
    _assert_callback_order(before_refresh, minimum_operations=3)

    chrome.call("Page.reload", {"ignoreCache": True})
    recovered = _wait_for(
        lambda: (
            body
            if "状態: Completed" in (body := chrome.body_text())
            and "payment_user_agent" in body
            else None
        ),
        timeout=30,
    )
    assert chrome.fetch_json("/mediation-api/v1/view") == before_refresh
    _CASE_RESULTS["paid"] = True


def test_02_free_plan_approval_completes_without_payment_target(
    browser_session: _BrowserSession,
) -> None:
    chrome = browser_session.chrome
    planned = _submit_and_wait(chrome, "hotel search", "WaitingForPlanApproval")
    _assert_plan_target(planned)
    assert '"canonicalAgentId":"agent-002"' in planned

    completed = _submit_and_wait(chrome, "承認", "Completed")
    assert "WaitingForPaymentApproval" not in completed
    view = chrome.fetch_json("/mediation-api/v1/view")
    assert view is not None and view["state"] == "Completed"
    assert view["pendingAction"]["kind"] == "none"
    _assert_callback_order(view, minimum_operations=1)
    _CASE_RESULTS["free"] = True


def test_03_settled_fulfillment_rejection_requires_explicit_refund(
    browser_session: _BrowserSession,
) -> None:
    chrome = browser_session.chrome
    planned = _submit_and_wait(
        chrome, "paid refund-required booking", "WaitingForPlanApproval"
    )
    _assert_plan_target(planned)
    payment = _submit_and_wait(chrome, "承認", "WaitingForPaymentApproval")
    _assert_payment_target(payment)

    payment_view = chrome.fetch_json("/mediation-api/v1/view")
    assert payment_view is not None
    fault_target = _refund_fault_target(_mediation_session_id(payment_view))
    _arm_refund_fault(fault_target)

    refund_pending = _submit_and_wait(chrome, "承認", "RefundPending")
    view = chrome.fetch_json("/mediation-api/v1/view")
    assert view is not None and view["state"] == "RefundPending"
    assert view["pendingAction"]["kind"] == "request-refund"
    _assert_refund_fault_consumed(fault_target)

    refunded = _submit_and_wait(chrome, "承認", "Refunded")
    final_view = chrome.fetch_json("/mediation-api/v1/view")
    assert final_view is not None and final_view["state"] == "Refunded"
    assert final_view["pendingAction"]["kind"] == "none"
    _assert_merchant_db_integrity()
    _CASE_RESULTS["refund"] = True


def test_04_browser_surfaces_hide_private_material_and_show_demo_scope(
    browser_session: _BrowserSession,
) -> None:
    chrome = browser_session.chrome
    planned = _submit_and_wait(chrome, "paid privacy booking", "WaitingForPlanApproval")
    _assert_plan_target(planned)
    payment = _submit_and_wait(chrome, "承認", "WaitingForPaymentApproval")
    _assert_payment_target(payment)

    deployment = chrome.fetch_json("/auth/deployment")
    assert deployment == {
        "ephemeral": True,
        "notice": "EPHEMERAL DEMO: state and keys may reset on restart",
        "officialX402": "NOT RUN",
        "onChainSettlement": "NOT RUN",
        "target": "ephemeral-cloud-run-demo",
        "durability": "NOT PROVIDED",
    }
    network, request_urls = chrome.network_audit()
    resource_timing = chrome.resource_timing()
    browser_surface = "\n".join((payment, resource_timing, network)).lower()
    for marker in PRIVATE_MARKERS:
        assert marker not in browser_surface

    assert "シミュレーション: x402 wire-shape fixture (NOT CONFORMANT)" in payment
    assert deployment["ephemeral"] is True
    assert deployment["durability"] == "NOT PROVIDED"
    assert request_urls
    http_urls = tuple(
        url for url in request_urls if url.startswith(("http://", "https://"))
    )
    assert http_urls
    dynamic_http_urls = tuple(
        url for url in http_urls if not urlsplit(url).path.endswith(".woff2")
    )
    unexpected_http_urls = tuple(
        url
        for url in dynamic_http_urls
        if urlsplit(url).hostname not in {"127.0.0.1", "localhost", "::1"}
        or urlsplit(url).port != 8080
    )
    assert not unexpected_http_urls, unexpected_http_urls
    assert all("firebase" not in url.lower() for url in request_urls)
    assert all("vertex" not in url.lower() for url in request_urls)
    _CASE_RESULTS["privacy"] = True
    _write_browser_evidence()
