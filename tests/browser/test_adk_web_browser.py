from __future__ import annotations

import json
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from websockets.sync.client import connect


pytestmark = [pytest.mark.browser, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[2]
PAYMENT_APP_SOURCE = (
    ROOT / "payment_user_agent"
    if (ROOT / "payment_user_agent").is_dir()
    else ROOT / "payment-apps/payment_user_agent"
)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class _ChromeDevTools:
    def __init__(self, websocket_url: str) -> None:
        self.socket = connect(websocket_url, open_timeout=10)
        self.next_id = 0

    def close(self) -> None:
        self.socket.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        request_id = self.next_id
        self.socket.send(
            json.dumps({"id": request_id, "method": method, "params": params or {}})
        )
        while True:
            message = json.loads(self.socket.recv(timeout=20))
            if message.get("id") == request_id:
                if "error" in message:
                    raise AssertionError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result["result"].get("value")

    def body_text(self) -> str:
        return str(self.evaluate("document.body.innerText"))


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


def _body_when(chrome: _ChromeDevTools, expected: str) -> str | None:
    text = chrome.body_text()
    return text if expected in text else None


def test_real_chromium_drives_two_approval_adk_web_session(
    tmp_path: Path, workflow_fixture
) -> None:
    adk = shutil.which("adk") or str(Path(sys.executable).with_name("adk"))
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    assert Path(adk).is_file(), "release image must contain the ADK CLI"
    assert chromium is not None, "release image must contain Chromium"
    port, workflow_port, debug_port = _free_port(), _free_port(), _free_port()
    log = tmp_path / "adk-web.log"
    apps_root = tmp_path / "payment-apps"
    shutil.copytree(PAYMENT_APP_SOURCE, apps_root / "payment_user_agent")

    workflow_server = uvicorn.Server(
        uvicorn.Config(
            workflow_fixture["app"],
            host="127.0.0.1",
            port=workflow_port,
            log_level="error",
        )
    )
    workflow_thread = threading.Thread(target=workflow_server.run, daemon=True)
    workflow_thread.start()
    _wait_for(
        lambda: httpx.get(
            f"http://127.0.0.1:{workflow_port}/health", timeout=1
        ).status_code
        == 200,
        timeout=15,
    )

    environment = os.environ.copy()
    environment["WORKFLOW_API_URL"] = f"http://127.0.0.1:{workflow_port}"
    environment["WORKFLOW_IDENTITY_ASSERTION"] = workflow_fixture["assertion"]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    with log.open("wb") as output:
        adk_process = subprocess.Popen(
            [
                adk,
                "web",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-reload",
                str(apps_root),
            ],
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    chrome_process: subprocess.Popen[bytes] | None = None
    chrome: _ChromeDevTools | None = None
    try:
        base = f"http://127.0.0.1:{port}"
        apps = _wait_for(lambda: httpx.get(f"{base}/list-apps", timeout=1).json())
        assert apps == ["payment_user_agent"]
        chrome_process = subprocess.Popen(
            [
                chromium,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={debug_port}",
                f"--user-data-dir={tmp_path / 'chrome-profile'}",
                base,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
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
        chrome.call("Runtime.enable")
        assert _wait_for(
            lambda: chrome.evaluate(
                'Boolean(document.querySelector(\'textarea[placeholder="Type a Message..."]\'))'
            )
        )
        initial = chrome.body_text()
        assert "payment_user_agent" in initial

        _submit_message(chrome, "browser booking")
        plan_notice = _wait_for(lambda: _body_when(chrome, "計画の承認"))
        _submit_message(chrome, "承認")
        payment_notice = _wait_for(lambda: _body_when(chrome, "決済の承認"))
        for exact_price in (
            "merchandiseAmount=1250",
            "customerSurcharge=0",
            "collectionRailCost=0",
            "customerTotal=1250",
            "providerCommission=0",
            "merchantPayableAmount=1250",
            "payoutRailCost=0",
        ):
            assert exact_price in payment_notice
        assert "approval expiry (UTC)" in payment_notice
        assert "課金警告" in payment_notice
        _submit_message(chrome, "承認")
        completed = _wait_for(lambda: _body_when(chrome, "完了"))
        assert "x402 v0.1 wire-shape test fixture (NOT CONFORMANT)" in completed
        assert "simulated; no real asset or on-chain transaction" in completed

        chrome.call("Page.reload", {"ignoreCache": True})
        recovered = _wait_for(
            lambda: (
                text
                if "完了" in (text := chrome.body_text())
                and "payment_user_agent" in text
                else None
            ),
            timeout=30,
        )
        assert "Demo booking confirmed" in recovered

        repository = workflow_fixture["repository"]
        # UI-generated ADK session identifiers are intentionally opaque.
        with repository._connect(repository.paths.marketplace) as connection:
            row = connection.execute(
                "SELECT workflow_id,state FROM workflows ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        assert row is not None and row["state"] == "completed"
        workflow_id = str(row["workflow_id"])

        evidence_path = os.environ.get("BROWSER_EVIDENCE_OUTPUT")
        if evidence_path:
            destination = Path(evidence_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    {
                        "schemaVersion": "ap2-x402-browser-evidence/1",
                        "status": "PASS",
                        "imageDigest": os.environ.get("RELEASE_IMAGE_DIGEST", "UNSET"),
                        "releaseManifestDigest": "sha256:"
                        + hashlib.sha256(
                            (ROOT / "tests/release/release_manifest.json").read_bytes()
                        ).hexdigest(),
                        "browser": "chromium-cdp-real-ui",
                        "listApps": apps,
                        "appSelected": "payment_user_agent",
                        "interactions": ["browser booking", "承認", "承認", "refresh"],
                        "workflowId": workflow_id,
                        "completedAfterRefresh": True,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        assert "計画の承認" in plan_notice
    finally:
        if chrome is not None:
            chrome.close()
        if chrome_process is not None:
            chrome_process.terminate()
            try:
                chrome_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                chrome_process.kill()
        adk_process.terminate()
        try:
            adk_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            adk_process.kill()
        workflow_server.should_exit = True
        workflow_thread.join(timeout=10)
