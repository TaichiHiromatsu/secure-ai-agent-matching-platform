# Mediator Payment Integration — Implementation Review

> [!WARNING]
> この文書は作成時点の引継ぎ／レビューsnapshotであり、現在仕様の正本ではない。現行責務は[アーキテクチャ](ARCHITECTURE.md#actorと責務の正本)と[Payment Bridge設計](mediator-payment-integration-design/04_PAYMENT_BRIDGE_AP2_X402.md)を参照する。本文は履歴証跡として変更しない。

## 1. Review status

- Review date: 2026-08-16 (Asia/Tokyo)
- Scope: the complete uncommitted implementation diff, the requirements, design set, implementation plan, new and changed tests, container packaging, and the fixed Cloud Run update procedure
- Review method: source-to-requirement inspection, current-source container build, local single-container runtime reproduction, and focused automated tests
- Cloud Run: **NOT RUN**. This review did not update a revision, tag, traffic, IAM, origin policy, Cloud SQL, or any other cloud resource.
- Code changes: none. This file is the only review artifact created by this review.
- Overall verdict after closure review: **NOT READY / DO NOT UPDATE CLOUD RUN**
- Closure addendum: Section 10 supersedes the original four-BLOCKER status and
  the pre-fix runtime observations in Sections 2–4. The original findings are
  retained as the audit record of the first review.
- Current candidate addendum: Section 13 re-adjudicates every BLOCKER and HIGH
  against final6. It supersedes candidate-state conclusions in Sections 1–12
  without deleting the historical findings or authorizing Cloud Run.

The implementation has useful isolated pieces: exact approval comparison, a deterministic payment tool surface, signed simulated mandates and guarantee, same-Task checks inside the payment bridge, Merchant-side signature verification, settlement/refund idempotency, loopback service listeners, and a default-deny edge. Those pieces do not yet form a usable or secure production path. In the current image, neither the free nor paid public happy path can reach the first plan approval.

## 2. Release decision

| Priority | Count | Decision |
| --- | ---: | --- |
| BLOCKER | 4 | All must be fixed before another candidate or Cloud Run update is attempted. |
| HIGH | 8 | All are Release-1 requirements and must be closed before promotion. |
| MEDIUM | 3 | May be scheduled after the blockers, but must not be represented as completed behavior. |

The critical-path acceptance oracles are currently:

| Oracle | Result | Evidence |
| --- | --- | --- |
| `FREE-HAPPY-01` | **FAIL / not reachable** | Public ADK invocation lacks trusted identity; with a supplied identity the production matcher raises a strict DTO validation error before plan creation. |
| `PAID-HAPPY-01` | **FAIL / not reachable** | Same two failures occur before plan approval, payment requirement, payment approval, AP2, guarantee, settlement, or same-Task completion. |
| `REFUND-01` | **FAIL / not reachable end-to-end** | Isolated bridge refund tests pass, but no public paid flow can create the eligible original payment; the refund authorization/evidence contract is also incomplete. |

Passing unit tests do not override these runtime failures. The controller happy-path tests inject `FakeMatcher`, `FakePlanner`, `PassingGates`, `RecordingHook`, `SequenceTransport`, `FakeBridge`, and `AcceptFinal`; they establish useful local behavior but deliberately bypass the production seams that failed here.

## 3. Reproduction evidence

### 3.1 Current-source image

The current working tree was built successfully as a new local image and started as an explicitly ephemeral local demo with all backends on loopback. The root health endpoint returned `OK`.

### 3.2 Normal public ADK request has no verified identity

1. Created an ordinary ADK session for the authenticated local subject with `state: {}`.
2. Submitted a normal free-agent goal through `POST /run`.
3. The endpoint returned HTTP 200 containing `VERIFIED_IDENTITY_REQUIRED`.

This is the expected result of [adk_adapter.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/adk_adapter.py:43), because [nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:233) authenticates the ADK session and run routes but does not inject the auth subrequest's signed identity into the ADK session/controller.

### 3.3 The same boundary accepts browser-selected identity state

Using the same authenticated local request context, an ADK session was created under `userId=victim-user` with browser-supplied state:

```json
{
  "verifiedIdentity": {
    "subject": "victim-user",
    "tenantId": "tenant-victim",
    "adkSessionId": "review-forged"
  }
}
```

The session creation returned HTTP 200. The following `/run` passed the adapter's identity check and made live calls to the Trusted Agent Store and paid Merchant Agent Card. Thus the edge proved only that *some* user was authenticated; it did not bind the Firebase subject to the ADK `userId` or the session state accepted by the mediator.

### 3.4 Production matcher stops all goals

After calling the production controller with an otherwise valid scope, the first live Agent Card produced:

```text
pydantic_core.ValidationError: SelectedAgentSnapshot.paymentExtensionUris
Input should be a valid tuple; input_value=[...], input_type=list
```

[adapters.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/adapters.py:201) converts the tuple of extension URIs back to a list before constructing the strict immutable DTO defined at [models.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/models.py:100). The ADK wrapper hides this as `MEDIATION_INTERNAL_ERROR`. The same conversion occurs for an empty free extension set, so both paths fail before plan display.

### 3.5 Public mediation façade and readiness do not match the backend

Against the same current-source container:

- `GET /mediation-api/ready` returned 503 with every check true except `routeIsolation=false`.
- `POST /mediation-api/v1/turns` returned backend 404.
- `GET /mediation-api/v1/view` returned backend 404.

[nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:287) publishes exact `ready`, `v1/turns`, and `v1/view` routes. [workflow/api.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/workflow/api.py:321) still implements the old workflow-ID API, and its route-readiness source check still requires a removed prefix location at [workflow/api.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/workflow/api.py:131).

### 3.6 Automated checks run

The focused suite covering the new mediation, payment bridge, Merchant A2A, migration, public-boundary, and Cloud Run update changes passed:

```text
52 passed, 1 dependency-version warning, 12.01s
```

This is valuable evidence for the isolated components. It is not candidate evidence because it did not exercise the current production composition from authenticated browser entry through the real matcher, planner, callbacks, gates, payment bridge, Merchant, final validator, and refund.

## 4. BLOCKER findings

### B-01 — The real matcher cannot construct a selected Agent snapshot

Requirements affected: FR-001, FR-003, FR-012; implementation-plan Phase 3 and `FREE-HAPPY-01`.

Evidence:

- The strict DTO requires `tuple[StrictStr, ...]` at [models.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/models.py:100).
- The production adapter supplies `list(extension_uris)` at [adapters.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/adapters.py:201).
- Controller tests supply a tuple from a fake matcher at [test_controller.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/tests/mediation/test_controller.py:43), so they cannot catch this defect.
- The current-source container reproduced the Pydantic exception for the live paid Card; the same expression supplies an empty list for free Cards.

Impact: every ordinary public goal stops before a plan is persisted or displayed. No first approval, free completion, paid continuation, payment, or refund can be reached.

Minimum safe correction:

1. Preserve the canonical tuple when constructing `SelectedAgentSnapshot` and calculate `snapshotDigest` from the same canonical representation used by the DTO.
2. Add a production-adapter integration test that calls `LegacyMatcherAdapter.match` against the real local Registry plus the real free and paid live Cards.
3. Add a current-image smoke test that asserts both free and paid goals reach `WaitingForPlanApproval`, not merely that a mocked controller does.

### B-02 — Authenticated identity is neither delivered to nor securely bound at the only public ADK root

Requirements affected: SEC-001, SEC-002, SEC-003, FR-002, FR-015; implementation-plan Phase 2 and Phase 6.

Evidence:

- The adapter trusts only `context.session.state["verifiedIdentity"]` at [adk_adapter.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/adk_adapter.py:43).
- The session, `/run`, and `/run_sse` locations authenticate but never copy the auth subrequest assertion into the mediator at [nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:233), [nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:250), and [nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:266).
- Normal empty session state reproduces `VERIFIED_IDENTITY_REQUIRED`.
- The ADK create-session body accepts initial state; a browser-selected subject, tenant, and matching `userId` passed the adapter and triggered internal live reads under that scope.
- [test_adk_adapter.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/tests/mediation/test_adk_adapter.py:12) manually constructs trusted-looking session state, so it proves only local comparison logic, not provenance.

Impact: the normal UI is unusable. If a caller supplies state, an authenticated subject can select another subject/tenant scope. This defeats the owner tuple before any plan/payment checks and is a cross-subject authorization flaw.

Minimum safe correction:

1. Choose one authoritative session-level ingress contract. The preferred implementation-plan route is authenticated `POST /v1/turns` / `GET /v1/view`, with the proxy-generated signed identity as the only owner source and with no subject, tenant, ADK session, mediation session, or workflow selector accepted from the body.
2. If ADK `/run` remains the transport, add a trusted server-side bridge that verifies the signed assertion and binds it to the actual ADK session. Do not store a browser-supplied `verifiedIdentity` value as trusted state and do not treat path/body `userId` as authentication.
3. Deny or strip `state.verifiedIdentity` and `stateDelta.verifiedIdentity` from public requests.
4. Add live edge tests: authenticated A with path/body/state B must fail before mediator/store access; an ordinary A session without client identity state must reach plan display as A.

### B-03 — The public Trusted Surface cannot present the exact plan or full payment target before accepting approval

Requirements affected: FR-004, FR-007, UI-002, UI-003, NFR-002, SEC-009; implementation-plan Phases 3–5.

Evidence:

- `MediationPublicView` contains only message, agent label, opaque plan/step/task references, pending action, and trace at [models.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/models.py:301).
- Plan approval display is a generic sentence plus an agent label and opaque references; it does not show the actual approved plan, step goal/conditions, currency, or payment limit at [controller.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/controller.py:905).
- Payment approval display exposes amount, currency, and payee only at [controller.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/controller.py:928). It omits the Merchant product, expiry, payment method/profile/scheme/network/asset, safe step/Task target in the consent text, and an explicit statement that this is distinct from plan approval.
- The bridge builds a richer deterministic `display` digest internally at [payment_bridge.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/payment_bridge.py:410), but that exact object is not returned to the UI before the raw `承認` is accepted.

Impact: string equality is deterministic, but it is not informed, target-bound consent. A user cannot verify what exact plan or complete payment terms the approval will authorize. The signed display digest can therefore differ in scope from what was visible on the Trusted Surface.

Minimum safe correction:

1. Return a deterministic, schema-versioned plan approval view containing the exact plan ID/version/digest, every selected step/Agent, goal/conditions, currency and limit, expiry, and the exact-token instruction.
2. Return the exact deterministic payment display object used for `displayDigest`, extended with product, expiry, payment method/profile/scheme/network/asset, and safe step/Task references.
3. The Trusted Surface must hash/sign the exact canonical bytes it rendered; the controller and bridge must compare that digest immediately before transition/mandate issuance.
4. Add browser tests that capture the rendered terms and prove a changed plan/Checkout invalidates the prior approval with zero remote/payment side effects.

### B-04 — The packaged public API contract is internally inconsistent and the candidate is never ready

Requirements affected: FR-002, FR-015, OPS-002/003, implementation-plan Phases 2, 6, and 7.

Evidence:

- Edge routes point to `/v1/turns` and `/v1/view` at [nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:300).
- The backend exposes `/v1/workflows`, `/v1/workflows/active`, and workflow-ID message routes instead at [workflow/api.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/workflow/api.py:321).
- The readiness checker requires the no-longer-present text `location /mediation-api/ {` at [workflow/api.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/workflow/api.py:138).
- Live results were 404 for both intended API routes and 503 for readiness.

Impact: the planned secure alternative to the raw ADK session API is unavailable, and any current-source candidate is not ready by its own readiness contract. Promotion must not use the shallow `/health` endpoint to bypass this result.

Minimum safe correction:

1. Implement the session-level turn/view API against the same `MediationController` and store used by the public app, or remove the unused public façade and replace it with one complete identity-bound entry route.
2. Remove the old externally shaped workflow-ID API from the public runtime contract.
3. Make readiness inspect the actual exact route allowlist semantically or via a maintained manifest, then require `/mediation-api/ready` 200 before candidate verification.
4. Add a packaged-container test for exact routes and real response schemas, rather than source substring assertions.

## 5. HIGH findings

### H-01 — The preserved legacy security callback does not receive the actual A2A response

Requirements affected: SEC-008, SEC-016, FR-010, FR-011.

The wrapper calls the real symbol, but replaces the actual response, history, tool calls, metadata, and Artifact with `{phase, taskDigest}` at [adapters.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/adapters.py:306). The legacy callback explicitly extracts `output`, `conversation_history`, `tool_calls`, and `tool_responses` for its Judge Agent at [orchestration_agent.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/subagents/orchestration_agent.py:466). Consequently the callback-hook-centered anomaly layer is invoked by name but cannot inspect the untrusted content it is supposed to judge.

Required correction and test: pass a bounded, redacted but complete structured A2A result/history into the unchanged callback contract; inject a prompt-injection Artifact and prove callback-after blocks before settlement/finalization. Retain callback and deterministic gates as separate events.

### H-02 — The five named deterministic gates are labels, not the required policy decisions

Requirements affected: FR-010, SEC-009, SEC-011, AUD-001.

[DeterministicStableGate.decide](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/adapters.py:349) passes every recognized pre-gate without inspecting an approved plan, step, Agent snapshot, capability, approvals, AP2 evidence, profile, or request digest relationships. Post gates check only that a response exists, and the payment-requirement gate only that a requirement object exists. Gate evidence records no policy version, schema version, reason, call ordinal, input/output digests, actor, or start/completion times; [TraceEvent](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/models.py:249) cannot represent the required audit record.

Required correction and test: define a strict typed input per stable gate, validate all minimum fields listed by FR-010, persist the exact decision record, and use negative tests at each gate to assert downstream side effects remain zero.

### H-03 — Payment A2A capabilities are materially under-scoped and are not bound to the selected Card

Requirements affected: FR-003, FR-009, SEC-007, SEC-015.

- [capability.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/capability.py:24) signs only operation, workflow ID, Task ID, order ID, request digest, and times. It omits plan/version/digest, step, canonical Agent, Card digest, skill, RPC endpoint, context, quote, profile, and approval/evidence bindings.
- The bridge payment adapter synthesizes a different Card digest from only canonical Agent ID and endpoint at [payment_bridge_adapter.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/payment_bridge_adapter.py:63), rather than reusing the approved live Card snapshot.
- The payment request body omits context and quote at [payment_bridge_adapter.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/payment_bridge_adapter.py:96).
- Merchant capability binding checks only workflow/task/order and capability ID at [merchant/api.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/merchant/api.py:174).

Required correction and test: sign and verify the complete approved scope before every Merchant side effect. Mutating any one of plan, step, canonical Agent, Card digest, skill, RPC endpoint, task, context, order, quote, operation, profile, or expiry must yield zero guarantee/settlement/fulfillment rows.

### H-04 — The AP2/offline evidence chain is incomplete and has no completion manifest

Requirements affected: FR-008, DATA-001 through DATA-005, SEC-012; implementation-plan Phase 4.

The continuation persists only the plan approval ID, not its nonce or issued-at, when attaching at [payment_bridge_adapter.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/payment_bridge_adapter.py:167). The pre-payment authorization envelope at [payment_bridge.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/payment_bridge.py:573) omits the owner tuple, selected Card/skill/RPC, plan approval nonce/issued-at, payment approval nonce/issued-at, and full AP2 object descriptors required for offline verification. There is no implementation of the planned post-result `mediation-completion-manifest/v1`; a code search found no completion-manifest producer or verifier.

Required correction and test: persist both approval artifacts in full, issue the pre-payment authorization envelope from their exact immutable bytes, create a one-way post-result completion manifest, and verify an exported bundle without consulting the workflow database. Every required-field deletion or digest mutation must fail verification.

### H-05 — Refund is a local balance reversal without the required refund authorization/evidence contract

Requirements affected: FR-016, DATA-006, SEC-009, implementation-plan Phase 5.

The controller routes exact `承認`, but creates no distinct owner-bound refund approval ID/nonce/issued-at/expiry/display digest. [PaymentBridge.refund](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/payment_bridge.py:781) builds a request without the original authorization-envelope digest, completion-manifest digest, settlement Receipt digest, fulfillment-failure digest, or approval artifact, then directly adjusts two local rail balances. The stored refund result has no signed/refund Receipt verification. It also does not traverse the shared A2A callback/gate executor.

Required correction and test: create and persist a deterministic refund consent artifact bound to the original owner/payment/receipt/failure and exact amount/reason; submit through the designated settlement owner with idempotency/CAS; verify a correlated result/receipt; test one refund, replay remains one, cross-owner/over-limit/unsettled remain zero.

### H-06 — The production mediator state is ephemeral and disconnected from the existing durable workflow API

Requirements affected: FR-006, FR-007 routing, NFR-003, DATA-001, implementation-plan Phases 1–2.

[composition.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/composition.py:47) constructs a new `InMemoryMediationStore`; payment bridge records go to SQLite, while plan/session/pending-action/idempotency/trace remain in the ADK process. [store.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/store.py:11) confirms that these records are memory-only. The loopback workflow API uses a separate legacy `WorkflowController`, not this mediator.

Impact: an ADK child-process restart loses the pending plan/payment/refund router while durable payment rows may remain. This is not the permitted Cloud Run instance-replacement limitation; it is a split authority inside a still-running instance.

Required correction and test: implement the mediation repository/CAS/outbox API on the single authoritative SQLite repository and make UI/API use that controller. Restart the ADK/API child between approvals and prove the same pending owner-bound record resumes without a new Task/payment.

### H-07 — A loopback endpoint still mints a signed identity for an arbitrary subject

Requirements affected: SEC-001 through SEC-003; implementation-plan Phase 2.

[verify.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/auth/verify.py:178) accepts an arbitrary `subject` from any loopback caller and signs a service identity assertion. The design and implementation plan explicitly require removing this endpoint. Loopback is a network boundary, not proof that the calling process is authorized to select a user; the container runs several components that process untrusted Agent/model content.

Required correction and test: delete arbitrary-subject minting. Derive assertions only from a successfully verified Firebase cookie (or from a mutually authenticated internal principal with a fixed non-user subject). Assert the route is absent from inside and outside the container.

### H-08 — Cloud Run promotion verification trusts a hand-authored PASS file and probes shallow health

Requirements affected: OPS-003, TEST-015, AC-015; implementation-plan Phase 7.

[update-payment-demo-cloudrun.sh](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/update-payment-demo-cloudrun.sh:242) curls only `/health`, then accepts a local JSON file whose check fields equal the string `PASS`. The script does not run the tagged revision's readiness/model/browser/free/paid/refund/boundary checks or verify cryptographic evidence produced by such a runner. [test_cloud_run_update.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/tests/container/test_cloud_run_update.py:165) demonstrates promotion by writing that PASS JSON directly. In the reviewed current image, `/health` is OK while readiness is 503 and both public happy paths are broken.

Required correction and test: candidate verification must execute or cryptographically verify outputs from the tag-bound E2E runner, require authoritative readiness 200, and bind source/image/revision/tag URL plus captured scenario operation IDs and side-effect counts. A hand-written PASS file and `/health` alone must not unlock promotion.

## 6. MEDIUM findings

### M-01 — The mediator, rather than the Merchant, pre-generates paid Task/order/context IDs

[controller.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/controller.py:303) generates Task/order IDs and uses the mediation session as context before the initial Merchant response. The design assigns task/context/order/quote ownership to the Merchant. This can be made safe as a protocol contract, but the design, capability, and evidence documents must then be updated consistently; otherwise return Merchant-owned identifiers and bind them to a client request-correlation ID.

### M-02 — Final validation receives a fabricated one-entry summary, not the complete execution history

[LegacyFinalValidationAdapter.validate](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/adapters.py:603) constructs one history entry from the final result and active step. It does not receive all A2A requests/responses, callback/gate outcomes, AP2/payment summaries, retries, or every plan step required by FR-011. Feed a bounded immutable final-validation dossier and add paid/free tests where an earlier anomalous response cannot be hidden by a benign final Artifact.

### M-03 — Trace rows reconstruct labels after execution instead of persisting real component events

[controller.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/controller.py:879) converts an in-memory `event_order` string list into trace labels later. It lacks implementation revision, exact input/output digests, actor, call ordinal, and start/end timestamps, and does not prove which production symbol actually ran. Persist events at the enforcement point and assert exact symbol/revision/count in real HTTP scenario tests.

## 7. Requirement coverage conclusion

### Satisfied in isolated scope

- Exact single-part `承認` comparison exists for plan, payment, and refund routing.
- The payment tool surface can execute from server lookup keys only.
- The simulation profile is labeled `NOT CONFORMANT` and official x402 fails closed.
- Merchant signed-guarantee verification and same-Task completion logic have useful isolated coverage.
- Pre-settlement guarantee rejection and post-settlement refund-required behavior are separated.
- Basic bridge idempotency/CAS and free-payment-record-zero tests pass.
- Internal services listen on loopback and the edge has a narrow default-deny route set.
- The Cloud Run update script fixes project, region, service, immutable image, zero traffic, rollback revision, and no Cloud SQL.

### Not satisfied end-to-end

- Public authenticated subject → authoritative mediator ownership
- Real matcher/planner → visible exact plan → first deterministic consent
- Real callbacks and all stable gates over the actual A2A content
- Free completion and final validation through the production composition
- Paid second informed consent → complete AP2 binding → fully scoped capability → signed guarantee → same remote Task → settlement → fulfillment → completion manifest → final validation
- Correlated authorized refund through the public path
- Durable same-instance pending-state recovery
- Candidate-bound browser/E2E evidence and authoritative readiness

The implementation therefore must not claim Release-1 PASS for the 126 blocking requirements, and existing historical Cloud Run/browser evidence must not be reused for this implementation.

## 8. Mandatory regression gate after correction

No new candidate should be built until all four blockers have a local current-source container PASS. The minimum non-mocked suite is:

1. **Identity/entry:** authenticate A; ordinary session/turn reaches A's plan without client identity state. Attempts to use B in path, body, state, `stateDelta`, query, or header are rejected before mediator/store access.
2. **Real selection:** free and paid goals call the real Store and live Cards, persist the same exact snapshot the planner and transport consume, and reach `WaitingForPlanApproval`.
3. **Plan consent:** the browser renders the canonical plan target; only exact `承認` advances it; any plan mutation invalidates it; Task starts remain zero before approval.
4. **Free:** one real Task start, callback before/after exact once, required gates exact once, payment rows zero, complete history reaches final validation, final `ACCEPT`, and a safe result is displayed.
5. **Paid:** one real Task start returns payment-required; no mandate/guarantee/settlement before the second consent; exact visible Checkout terms are signed; one guarantee submission and one fulfillment commit use the same task/context/order/quote and approved Card/capability scope; final `ACCEPT` precedes success display.
6. **Refund:** force post-settlement fulfillment failure; render and sign a distinct refund consent; one correlated full refund and Receipt; exact replay remains one; different owner/order/amount/unsettled state remain zero.
7. **Callback/gates:** actual untrusted response content reaches the legacy callback; callback exception/prompt injection and each deterministic gate mutation fail closed with the specified downstream effect count zero.
8. **Restart:** restart the public ADK/API child between each approval stage and resume the same durable owner-bound record without a new Task/payment/refund.
9. **Packaged boundary/readiness:** `list-apps` returns only `payment_user_agent`; internal surfaces remain unreachable; authoritative readiness is 200; exact public route schemas work; unknown methods/paths and forged identity are rejected.
10. **Candidate:** the same immutable image passes free, paid, refund, browser, boundary, model probe, and readiness at the exact zero-traffic tag URL. The verification runner—not a manually authored JSON file—produces candidate-bound evidence before promotion is enabled.

## 9. Recommended order of work

1. Fix B-01 so production selection can be exercised.
2. Fix B-02 and B-04 together around one identity-bound authoritative turn route and store.
3. Fix B-03 so both approvals authorize exactly what the user saw.
4. Close H-01 through H-07 while building the real free, paid, and refund integration tests.
5. Close H-08 and only then build an immutable `linux/amd64` candidate.
6. Run the tagged zero-traffic Cloud Run verification and promote only after every Release-1 gate has candidate-bound evidence.

Until those conditions are met, the correct operational state is: **local implementation under review; Cloud Run unchanged; promotion prohibited**.

## 10. BLOCKER closure addendum

### 10.1 Closure scope and decision

- Closure review date: 2026-08-16 (Asia/Tokyo)
- Scope: B-01 through B-04 only, after the corrective implementation diff
- Method: source-to-acceptance inspection, independent cross-review, a fresh
  current-tree image, packaged Python 3.12 regression tests, and local edge
  requests against the running single-container image
- Cloud Run: **NOT RUN**. No revision, tag, traffic, IAM, origin, database, or
  other cloud resource was read or changed for acceptance.
- Code changes by this closure review: none; only this review document was
  updated.
- Closure verdict: **NOT READY**

Two original blockers are closed and two remain open:

| BLOCKER | Closure result | Reason |
| --- | --- | --- |
| B-01 production matcher DTO/runtime failure | **PASS / CLOSED** | The strict tuple boundary is preserved, the digest is derived from the same canonical snapshot, and both real free and paid Cards reach plan approval in the packaged image. |
| B-02 authoritative authenticated identity | **FAIL / OPEN** | Signed identity delivery and direct-backend fail-closed behavior were added, but the public state-changing routes still lack exact Origin/CSRF enforcement, subject selectors are rewritten or ignored instead of being rejected before store access, and the Firebase session Cookie is forwarded beyond the auth verifier. |
| B-03 exact visible approval target | **PASS / CLOSED** | Plan and payment targets are typed, canonical, displayed with their digest, and re-derived immediately before the corresponding side effect. Mutation tests stop with zero new downstream effects. |
| B-04 one consistent public API/readiness contract | **FAIL / OPEN** | Exact routes and schemas now work, but ADK `/run` and `/mediation-api/v1/turns` instantiate separate controllers with separate in-memory stores. Readiness can report 200 while the two public entry paths disagree about the active mediation. |

The gate is therefore not `READY FOR FULL TEST`. Passing focused tests and
reaching the first approval do not close the remaining security and authority
split.

### 10.2 B-01 acceptance review — PASS

The correction preserves `extension_uris` as a tuple when constructing the
strict `SelectedAgentSnapshot` and calculates `snapshotDigest` from the same
wire object at [adapters.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/adapters.py:171). The production regression calls the real Registry reader and real local free and paid Agent Cards at [test_production_matcher.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/tests/mediation/test_production_matcher.py:40), and checks the validated DTO digest rather than using a fake matcher.

Independent cross-review reproduced the packaged edge behavior with the free
`agent-002` and paid `agent-005` Cards. Both returned HTTP 200,
`WaitingForPlanApproval`, a `plan-approval-target/1` object, and its target
digest. The closure reviewer independently reproduced the same result in the
integrated fresh image. Neither response contained `MEDIATION_INTERNAL_ERROR`.

B-01's three minimum corrections—tuple/canonical consistency, a real
Registry/live-Card adapter test, and packaged free/paid smoke to the first
approval—are satisfied.

### 10.3 B-02 acceptance review — FAIL

This component was not self-approved by its implementer. The decision below is
from independent cross-review plus local black-box evidence.

Positive changes are present: the edge obtains a signed identity, the bridge
injects a private assertion into ADK state, identity-looking client state is
rejected, and the ADK adapter requires a valid signed assertion. Those changes
make the normal ADK route usable and make unsigned direct ADK use fail closed.
They do not complete the public ingress contract:

1. The session bridge explicitly rewrites an arbitrary path user to the
   authenticated subject at [verify.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/auth/verify.py:338), and `/run` overwrites an arbitrary body `userId` at [verify.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/auth/verify.py:388). The public `view` route also accepts unknown subject/tenant query selectors and ignores them. Cross-review and the current container both observed a victim selector request succeed rather than fail before mediator/store access.
2. Session creation, `/run`, `/run_sse`, and `/mediation-api/v1/turns` forward
   `Origin` and `X-CSRF-Token`, but neither bridge handler nor turn handler calls
   the existing exact same-origin and CSRF checks. A state-changing turn with
   neither header returned HTTP 200 in the current container.
3. Nginx disables inherited request headers, then explicitly restores the
   Firebase session Cookie to the ADK and workflow upstreams at
   [nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:185),
   [nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:293),
   and [nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:306).
   The signed identity assertion should be the sole upstream identity material;
   the Firebase Cookie must terminate at auth.

Minimum remaining correction: choose one public state-changing entry contract;
reject every path/body/query/header subject, tenant, session, mediation, or
workflow selector before store access; enforce exact Origin plus CSRF at that
entry; and never forward the Firebase Cookie to ADK, workflow, or other
non-auth components. Add edge negative tests that prove zero controller/store
calls for each selector and CSRF failure.

### 10.4 B-03 acceptance review — PASS

The public model now carries schema-versioned plan and payment targets at
[models.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/models.py:154). The plan target includes plan identity/version/digest,
all steps, selected Agent/Card/skill/RPC snapshot, goal, conditions, currency,
limit, expiry, and the exact approval token. The distinct payment target
includes the bridge display and its digest, product, expiry, payment method,
profile/scheme/network/asset, safe step/Task references, requirement and
Checkout digests, and the exact approval token.

The controller persists `canonical_digest(target)` when displaying each target
and re-derives it immediately before plan or payment approval at
[controller.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/controller.py:210) and
[controller.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/controller.py:579). The durable payment adapter independently reconstructs the target and checks both the target digest and the bridge's persisted `displayDigest` before payment execution at
[payment_bridge_adapter.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/payment_bridge_adapter.py:221). The ADK response renders the exact compact canonical JSON, target digest, and complete-message `承認` instruction.

Independent cross-review confirmed that plan mutation, Checkout mutation, and a
stale displayed digest stop before new remote execution, bridge approval,
guarantee, or settlement. The integrated container returned canonical plan
targets for both free and paid goals. B-03 is closed at the component boundary;
this does not waive the B-02/B-04 single-entry and single-store failures.

### 10.5 B-04 acceptance review — FAIL

The route-shape portion is corrected. The edge now publishes exact authenticated
`/mediation-api/ready`, `/mediation-api/v1/turns`, and
`/mediation-api/v1/view` locations at
[nginx.conf](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/deploy/nginx.conf:293).
The backend implements typed turn/view responses over a mediation controller at
[api.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/workflow/api.py:418), and the old workflow-ID paths and unknown paths are not exposed by the edge. The fresh container returned:

```text
/mediation-api/ready        200  routeIsolation=true, mediationComposition=true
/mediation-api/v1/turns     200  mediation-turn-response/1, WaitingForPlanApproval
/mediation-api/v1/view      200  exact view from the preceding turn
/mediation-api/v1/workflows 404
unknown mediation path      404
```

The authoritative-store portion is not corrected. Each call to
`create_production_controller` defaults to a new `InMemoryMediationStore` at
[composition.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/mediation/composition.py:20). The ADK process lazily creates one controller through
[agent.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/payment_user_agent/agent.py:6), while the separate workflow API process creates another at startup. They therefore expose two public mediation authorities.

The integrated black-box reproduction created a plan through
`/mediation-api/v1/turns`, then created a different paid plan through ADK
`/run`. A later `/mediation-api/v1/view` remained byte-for-byte equal to the
first API plan rather than reflecting the ADK plan. Readiness nevertheless
remained 200. In addition, `routeIsolation` is still derived from source
substring checks at [api.py](/Users/taichihiromatsu/Documents/enterprise-a2a-pf/secure_mediation_agent/workflow/api.py:185), not from a maintained route manifest or semantic runtime inspection.

Minimum remaining correction: expose exactly one authenticated public turn/view
authority backed by one controller and one owner-bound store; remove or make the
other path a thin client of that authority; and make readiness fail when either
the route manifest or authoritative composition is split. Add a packaged test
that alternates the supported UI calls and proves every read observes the same
versioned mediation record.

### 10.6 Fresh-image evidence

The final reviewed working tree built as
`enterprise-a2a-pf:blocker-closure` with manifest-list digest:

```text
sha256:af222aaee88e1d476c965a276e55ee7cdf88705766e015e93e2658c0b7b1f616
```

The packaged Python 3.12 focused suite covering the production matcher,
controller approval targets and mutation guards, durable bridge target binding,
ADK rendering/identity bridge, public turn/view API, workflow/payment/Merchant
integration, migration, and edge boundaries returned:

```text
65 passed, 1 dependency-version warning, 5.14s
```

Additional independent packaged runs returned 51/51 for B-01/B-03 coverage and
39/39 for B-04 workflow/payment integration. The current integrated live checks
returned both free and paid ADK goals at `WaitingForPlanApproval` with canonical
targets, and the route results shown above. The same live run also proved the
unclosed negative cases: a turn without Origin/CSRF succeeded, a victim query
selector returned 200 and was ignored, and the API view did not observe the ADK
store's later plan.

These results are good reasons to retain the B-01 and B-03 fixes. They are not
evidence that the candidate is safe to promote. The operational conclusion
remains: **NOT READY; Cloud Run unchanged; promotion prohibited**.

## 11. B-02/B-04 independent re-review addendum (2026-08-16)

### 11.1 Scope, provenance, and decision

This addendum independently re-reviewed only the corrections claimed for the
previous B-02 and B-04 findings. No application, deployment, or test code was
changed during the review. Exactly one image was freshly built from the shared
working tree:

```text
enterprise-a2a-pf:b2-b4-rereview
sha256:87ab93dd45bc51e189b5dd3c29a0253b8c647d7becec8b3769a049fa1ed3cf7d
```

Checksums of the packaged and working-tree copies of `verify.py`, `nginx.conf`,
the ADK adapter and HTTP authority, and the workflow API matched. The packaged
B-02/B-04 security, mediation, workflow, and integration selection returned:

```text
48 passed, 1 dependency-version warning
```

The re-review decision is **NOT READY FOR FULL TEST**. B-04's single-authority
and live-readiness corrections are reproducibly closed, and nearly all of
B-02's boundary correction is closed. The exact B-02 acceptance contract is
still missed for an unknown selector in a JSON body: it is rejected with HTTP
422 rather than HTTP 403.

No Cloud Run deployment or mutation was performed.

### 11.2 Acceptance matrix

| Required result | Fresh-container result | Decision |
|---|---|---|
| Other-subject or unknown selectors in path/body/query/header/state/stateDelta return 403 before store access; no rewrite | Other-subject path and body, unknown path and query, selector header, nested `state`, and nested `stateDelta` all returned 403. The same public record state/version/digest remained unchanged. However, body `unknownSelector` returned 422 on both `/mediation-api/v1/turns` and `/run`; non-null `selectionToken` also returned 422. | **FAIL** |
| Every state-changing public route requires exact Origin plus CSRF, while the normal browser contract remains usable | Missing-CSRF session create, delete, `/run`, `/run_sse`, mediation turn, login, and logout returned 403; wrong-Origin turn returned 403. Correct-origin bootstrap, session create, `/run`, `/run_sse`, mediation turn, logout, and delete reached their normal responses. | **PASS** |
| Firebase session Cookie terminates at auth verification and is never sent to ADK or workflow | The packaged nginx has one raw `Cookie` forwarding directive, inside internal `/auth/verify`. Public upstream locations disable request headers and inject only the signed assertion. Capturing bridge tests observed no Cookie on either ADK or workflow upstream requests. | **PASS** |
| ADK and workflow API share one controller/store and alternating operations observe one session/state/digest, with no ADK-local fallback | The only production-controller construction site is workflow startup; ADK uses the loopback HTTP authority. Workflow -> ADK -> workflow -> ADK alternation repeatedly observed mediation session `med-6b7f492c-87f6-4f25-acb9-a5eacbb3355d`, `WaitingForPlanApproval`, version 0, and digest `sha256:09d9283cc041da84168b7c8537dc4a9c5c4295bf16dc00c537c40b034346bc3a`. | **PASS** |
| Readiness performs a live method-by-path isolation probe | `/mediation-api/ready` returned 200 with `checks.routeIsolation=true` and `checks.mediationComposition=true`. The live matrix returned 404 for wrong-method `/run`, wrong-method mediation routes, raw `/v1/view`, and internal/store-looking paths; unknown mediation selector paths returned 403. The implementation performs HTTP requests against the edge rather than nginx-source substring inspection. | **PASS** |

The normal free goal and the paid goal each returned HTTP 200 at
`WaitingForPlanApproval` with a canonical approval-target digest. In separate
fresh-container runs, a subsequent ADK status operation rendered the same
digest and state as `/mediation-api/v1/view`; the paid run also rendered
`paid_booking_agent`. This is sufficient for the B-02/B-04 ingress and
single-authority review. It does not waive the other open end-to-end findings
in this document.

The in-app visual browser could not be attached because the runtime exposed no
browser target. The same-origin browser bootstrap sequence was therefore
reproduced over the real nginx edge: `/dev-ui/` contained the bootstrap script,
`/auth/browser-bootstrap` returned the fixed subject and CSRF token/cookie, and
the resulting session/create/run/turn requests succeeded. Visual UI behavior
remains an evidence limitation, not the blocker that determines this gate.

### 11.3 Remaining B-02 blocker

The public bridge recursively denies known owner keys, but it does not apply an
exact body-selector allowlist before forwarding. Consequently, an otherwise
valid mediation request containing `"unknownSelector":"attacker"` reaches
workflow schema validation and returns 422. `/run` likewise returns 422 from its
unsupported-field validation. Neither case wrote the mediation store, but the
required security-boundary response is 403, so the stated acceptance condition
is not met.

Minimum correction: classify and reject caller-supplied unknown selector fields
and non-null selection tokens at the authenticated public bridge with HTTP 403,
before any ADK/workflow upstream call. Add capturing tests for both `/run` and
`/mediation-api/v1/turns` that assert status 403 and zero upstream/controller/
store calls. After that change, repeat this exact one-image negative matrix and
the visual browser smoke before changing the gate to `READY FOR FULL TEST`.

## 12. B-02 final selector closure (2026-08-17)

This closure supersedes the gate in section 11. Exactly one fresh image was
built from the current shared tree:

```text
enterprise-a2a-pf:b2-final-closure-20260817
sha256:f01b9c3d6a37b1ce02efda3e25970cc1b3316db01b6a4caf8fded3a6aacedc43
```

The focused packaged suite covering the ADK identity bridge, public workflow
bridge, shared mediation authority, ADK adapter, and live route probe returned:

```text
32 passed, 1 dependency-version warning
```

Capturing tests proved that `unknownSelector` and a non-null `selectionToken`
return HTTP 403 before an ADK or workflow upstream call. The fresh live
container then returned 403 for both inputs on `/run`, `/run_sse`, and
`/mediation-api/v1/turns`. `/mediation-api/v1/view` was `null` before and after
the complete rejection matrix, proving that no mediation record was created.

The corresponding normal matrix returned HTTP 200 for `/run`, `/run_sse`, and
`/mediation-api/v1/turns`; the mediation request included
`"selectionToken":null` and returned `WaitingForPlanApproval`, version 0, and a
canonical approval-target digest. The selector rejection therefore does not
break the supported null-token/browser path.

The last B-02 acceptance gap is closed. Together with the B-04 evidence in
section 11, the independent closure decision is **READY FOR FULL TEST**. This
is a test-progression gate only: it does not waive the other findings in this
review and does not authorize deployment or promotion. Cloud Run was not
changed.

## 13. final6 implementation closure appendix (2026-08-17)

### 13.1 Scope and evidence

This appendix is the current candidate assessment. Sections 1–12 remain an
append-only audit record of earlier images and defects. A finding marked
`CLOSED` below means closed for the exact final6 local simulation boundary; it
does not imply production durability, live-cloud verification, or official
x402/on-chain conformance.

The reviewed candidate is:

```text
enterprise-a2a-pf:full-test-final6-20260817-amd64
sha256:3e4e089643564e00bd6563d08a575fcf2aa2eff94ae60fd1ca518900022a89f0
```

The candidate evidence is bound to the same image and release-manifest digest:

| Artifact | Status | File SHA-256 |
| --- | --- | --- |
| `artifacts/regression-result-final6.json` | `PASS` | `f64da6ec882b3a6a14f27a8df5448ad971c01c208c3f8bcf6070335edfa84ded` |
| `artifacts/browser-evidence-final6.json` | `PASS` | `1059985e2fac45b8c7c70ed316e2359d1c6da64acc004ebf0207560a3796fa50` |
| `artifacts/ap2-x402-release-validation-final6.json` | `PASS`, `failures={}` | `4f4aa723d9a5bc02eec4c09d6f097c749f2d6f6652c66f0dc2b0a72573cf96ce` |

The HIGH table uses `R`, `B`, and `V` as exact references to the regression,
browser, and release-validation artifacts in that order. Each has
`imageDigest=sha256:3e4e089643564e00bd6563d08a575fcf2aa2eff94ae60fd1ca518900022a89f0`
and
`releaseManifestDigest=sha256:852aeaba0e024469eb35adfa45a1dd6fabd054484d68aa1b58739ddaf8457f37`.
The JSON artifacts contain aggregate suite/case evidence, not per-test node
IDs. The named packaged tests below are therefore the reproducible test
oracles inside that exact candidate source/image; a row is not called closed
when the exercised runtime profile or required producer is absent.

Canonical regression passed payment 285/285, evaluation 17/17, and jury 5
pass plus exactly 8 allowed skips. The 11-marker validator passed with zero
failure, error, or skip in every marker. Real Chromium passed paid, free,
refund, and privacy. The literal unfiltered pytest result was 304 pass, three
known W&B authentication failures, eight skips, and zero collection errors.

### 13.2 BLOCKER re-adjudication

| Original finding | final6 decision | Closure evidence |
| --- | --- | --- |
| B-01 real matcher cannot construct an Agent snapshot | **CLOSED** | The production typed matcher/planner path reaches plan approval for paid and free flows; canonical integration and browser scenarios pass from the packaged image. |
| B-02 authenticated identity is not securely bound | **CLOSED** | The public route uses server-owned session identity, exact Origin/CSRF, and signed internal identity; selector injection is rejected, privacy passes, and a valid wrong-owner assertion returns JSON `null`. |
| B-03 Trusted Surface cannot present exact targets | **CLOSED** | Chromium observes separate plan and payment cards/digests and consumes separate exact `承認` requests before side effects. Refresh restores the authoritative view. |
| B-04 packaged public API is inconsistent / not ready | **CLOSED for local final6** | Public mutation and view use the loopback workflow authority, readiness is HTTP 200 with schema v4/writable/decryptable checks, and exact-image browser/container suites pass. Cloud Run remains NOT RUN. |

No original BLOCKER remains open for the final6 local simulation demo. This is
not a Cloud Run promotion decision.

### 13.3 HIGH re-adjudication

| Original finding | final6 decision | Closure evidence / remaining boundary |
| --- | --- | --- |
| H-01 legacy callback does not receive the real A2A response | **PARTIAL — deterministic-local tested; production Legacy NOT RUN** | `tests/mediation/test_local_callback.py::test_local_callback_validates_the_actual_remote_task_binding` proves the local hook receives and checks the actual `RemoteTaskSnapshot`; `tests/mediation/test_a2a_executor.py::test_executor_has_one_enforced_callback_gate_transport_order` fixes ordering; browser tests `test_01_paid_two_exact_approvals_and_refresh` and `test_02_free_plan_approval_completes_without_payment_target` assert one before/after pair while the packaged browser explicitly sets `MEDIATION_CALLBACK_MODE=deterministic-local`. `B.cases` contains paid/free and `R.suites[payment-release]=285 PASS`. Neither artifact executes production `LegacyCallbackHook`; its real-response/history contract remains unclosed. |
| H-02 deterministic gates are labels rather than decisions | **CLOSED for final6 deterministic policy** | `tests/mediation/test_a2a_executor.py::test_callback_failure_is_fail_closed`, `tests/mediation/test_controller.py::test_paid_path_uses_same_executor_and_second_exact_approval`, and `::test_changed_checkout_invalidates_payment_approval_before_bridge_side_effects` assert enforcement order, BLOCK behavior, and zero forbidden effects. Exact-image `V.suites.security=84 PASS`, `V.suites.integration=56 PASS`, and `R.suites[payment-release]=285 PASS` cover the packaged policy; this does not claim an LLM detector or production Legacy callback run. |
| H-03 payment capability is under-scoped / not Card-bound | **CLOSED for simulation** | `tests/integration/test_external_merchant_a2a.py::test_external_capability_cannot_replay_across_workflow_task_or_order`, `::test_integrated_capability_attack_matrix_has_zero_merchant_effects`, and `::test_integrated_revoked_capability_is_rejected_before_replay` bind Card/profile/task/context/order/request and reject replay. Exact-image `V.suites.security=84 PASS`, `integration=56 PASS`, `contract_ap2=17 PASS`, and `contract_x402_simulation=2 PASS`; `V.officialX402=NOT RUN` and `V.onChainSettlement=NOT RUN`. |
| H-04 AP2/offline evidence chain is incomplete | **PARTIAL — canonical pair NOT IMPLEMENTED** | `tests/ap2/test_offline_evidence_chain.py::test_completed_graph_verifies_offline_with_role_separated_keys` and exact-image `V.suites.contract_ap2=17 PASS` verify the existing payment-marketplace evidence graph. They do not prove the accepted `mediation-authorization-envelope/v1` plus `mediation-completion-manifest/v1` pair: final6 still emits implementation-local `pre-payment-authorization-envelope/1` with `typ=JWT`, and no completion-manifest producer/verifier exists. The canonical schema/digest/signature contract is unique in the design, but implementation closure remains open. |
| H-05 refund lacks authorization/evidence | **PARTIAL — functional local refund tested** | `tests/mediation/test_controller.py::test_settled_fulfillment_rejection_requires_exact_refund_approval`, `tests/payment_bridge/test_payment_bridge.py::test_post_settlement_failure_requires_one_full_refund`, and browser `test_03_settled_fulfillment_rejection_requires_explicit_refund` prove `RefundPending`, distinct exact text action, one refund, and replay safety. `B.cases` contains refund, `B.status=PASS`, and `V.suites.browser=4 PASS`. There is still no distinct signed refund approval ID/nonce/issued-at/expiry/display-digest artifact or canonical completion-manifest binding, so the original evidence-contract portion is not closed. |
| H-06 mediator state is ephemeral and split from durable workflow | **CLOSED for local durable profile** | `tests/integration/test_mediation_public_api.py::test_sqlite_restores_waiting_and_terminal_turns_across_controllers`, `tests/mediation/test_sqlite_store.py::test_five_stable_states_restart_with_exact_view_and_binding`, and `::test_request_reservation_exact_replay_conflict_and_processing` cover encrypted schema-v4 restart, owner binding, and exact replay. Exact-image `V.suites.restart=41 PASS`, `concurrency=4 PASS`, and `container=16 PASS`; the recorded runtime observation is `WaitingForPaymentApproval` v2 and `Completed` v5 across two restarts with all three DB `quick_check` results OK. Cloud Run remains memory-backed and NOT durable. |
| H-07 loopback endpoint mints arbitrary user identities | **CLOSED** | `tests/integration/test_mediation_public_api.py::test_turn_body_cannot_select_identity_or_workflow`, `tests/security/test_public_workflow_auth_bridge.py::test_workflow_selectors_are_rejected_before_proxy`, and `tests/security/test_adk_identity_bridge.py::test_bridge_binds_session_and_run_to_signed_subject_without_leaking_assertion` prove server-derived scope and pre-authority selector rejection. Exact-image `V.suites.security=84 PASS`, `integration=56 PASS`; the runtime wrong-owner view was JSON `null`. |
| H-08 promotion trusts shallow or hand-authored PASS evidence | **PARTIAL / CLOUD RUN NOT RUN** | `tests/container/test_release_validator_binding.py::test_validator_binds_exact_image_manifest_regression_and_browser` and `tests/container/test_cloud_run_update.py::test_candidate_requires_exact_evidence_before_promotion_and_rolls_back` cover local binding/fail-closed promotion logic. `V.status=PASS`, `V.failures={}`, all 11 `V.suites` groups pass, and `R`/`B` carry the same image/manifest digests. No live tagged revision, readiness, Firebase/Vertex, traffic shift, observation, or rollback was run, so production promotion remains open. |

### 13.4 MEDIUM disposition

M-01 through M-03 no longer block the local demo: the current wire contract
defines identifier ownership, final validation consumes the actual execution
result, and safe trace/state are persisted and recovered by the v4 authority.
These decisions remain limited to the current single-active-session simulation
scope; complex remote reconciliation stays in the future-work register.

### 13.5 Durability and deployment boundary

The local durable profile and Cloud Run demo profile are intentionally
different:

- Local: `MEDIATION_STORE_MODE=sqlite`, named data/evidence/key volumes, SQLite
  schema v4. Container restart/recreate on the same host is verified.
- Cloud Run: `EPHEMERAL_CLOUD_RUN_DEMO=true` and
  `MEDIATION_STORE_MODE=memory`. Mediation can be lost on a child-process
  restart; all instance-local SQLite/key state can be lost on replacement,
  scale-down, or revision update. The UI and claim surface must say
  `durability=NOT PROVIDED`.

No Cloud Run build, push, revision, tag, traffic, IAM, Origin, or Cloud SQL
change was made. Local v4 evidence must not be cited as Cloud Run durability.

### 13.6 Current readiness decision

| Release surface | Decision |
| --- | --- |
| Local paid/free/refund/privacy simulation demo | **READY / VERIFIED** |
| Local single-host SQLite v4 restart and exact replay | **READY / VERIFIED** |
| Raw repository test green | **NOT READY** — three known W&B failures |
| Requirement-specific 139-record release closure | **PARTIAL** — 126 required records lack a candidate-specific PASS ledger; 13 future records are DESIGNED |
| Cloud Run tagged revision and traffic promotion | **NOT RUN / NOT AUTHORIZED** |
| Production durability | **NOT PROVIDED** |
| Official x402 wallet/facilitator/on-chain conformance | **NOT RUN / NOT CONFORMANT** |

The final6 implementation is therefore **ready as a local simulation demo
candidate** and **not ready for Cloud Run or production promotion**. The open
promotion gates are live Firebase/Vertex/Cloud Run verification,
candidate-specific 139-record closure, official conformance boundaries, and
the raw W&B failures. This appendix supersedes prior candidate-state findings
only within that explicit boundary.
