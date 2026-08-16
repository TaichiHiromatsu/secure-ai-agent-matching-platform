# Mediator Payment Integration — Documentation Review

- review date: 2026-08-17 (Asia/Tokyo)
- review type: independent, documentation-only final review
- scope: the 13-file design set, integration requirements, implementation plan, test report, implementation review, and the three `final6` evidence artifacts
- code/deployment changes: none
- Cloud Run: **NOT RUN / NOT TOUCHED**
- initial verdict: **CHANGES REQUIRED**
- current verdict: **APPROVED** — closure re-review in section 9 supersedes the initial verdict

## 1. Executive decision

The documentation package has a strong overall structure and correctly avoids claiming Cloud Run, production durability, official x402, on-chain settlement, or complete Release-1 closure. The normative requirement set and design coverage manifest are exactly aligned at 139 unique IDs, partitioned into 126 `release-1-required` and 13 `future-work` records. The three `final6` artifact hashes, image binding, suite counts, and browser case labels match the test report and implementation plan.

Approval is nevertheless withheld because the serialized public contract and signed evidence contract each contain an authoritative contradiction. A consumer following the design literally could reintroduce a client-selected session identifier or implement a different signed schema identifier. Three medium consistency/evidence issues should be corrected in the same documentation pass.

No finding in this review authorizes a code, deployment, traffic, IAM, Origin, Cloud SQL, or Cloud Run change.

## 2. Scope and method

Reviewed design files:

1. `mediator-payment-integration-design/README.md`
2. `01_OVERVIEW_ARCHITECTURE.md`
3. `02_DOMAIN_DATA_STATE.md`
4. `03_MEDIATION_FLOW.md`
5. `04_PAYMENT_BRIDGE_AP2_X402.md`
6. `05_SECURITY_TRUST_BOUNDARIES.md`
7. `06_API_A2A_CONTRACTS.md`
8. `07_UI_TRACE.md`
9. `08_PERSISTENCE_RECOVERY.md`
10. `09_DEPLOYMENT_PUBLIC_BOUNDARY.md`
11. `10_TEST_STRATEGY.md`
12. `11_TRACEABILITY_RELEASE.md`
13. `12_DECISIONS_OPEN_QUESTIONS.md`

The review also compared:

- `MEDIATOR_PAYMENT_INTEGRATION_REQUIREMENTS.md`
- `MEDIATOR_PAYMENT_INTEGRATION_IMPLEMENTATION_PLAN.md`
- `MEDIATOR_PAYMENT_INTEGRATION_TEST_REPORT.md`
- `MEDIATOR_PAYMENT_INTEGRATION_IMPLEMENTATION_REVIEW.md`
- `artifacts/regression-result-final6.json`
- `artifacts/browser-evidence-final6.json`
- `artifacts/ap2-x402-release-validation-final6.json`

Checks covered document ownership and reading order; exact requirement cardinality; paid, free, and refund flows; AP2, Trusted Surface, payment authority, and demo guarantee roles; production versus local callback profiles; single mutation authority; SQLite v4 versus Cloud Run ephemeral durability; authentication, Origin, CSRF, and identity binding; final6 counts and hashes; historical finding closure; NOT RUN/future boundaries; and local links and terminology.

## 3. Findings

### DOC-H-01 — The canonical public turn request both forbids and includes a client session selector

Severity: **HIGH**

`06_API_A2A_CONTRACTS.md` states that packaged final6 accepts only `requestId`, `expectedVersion`, and text parts beyond schema/null selection fields, and that the server derives the authoritative ADK/mediation session from verified identity. Its canonical `mediation-turn-request/1` example nevertheless includes `adkSessionId`. The following rules omit `adkSessionId` from the explicit forbidden-field list even though they say the ADK session is server-derived.

This conflicts with the security and deployment documents, which make the server-owned identity/session resolution part of the authorization boundary. Because `06` is the serialized-contract owner and declares unknown-field rejection, the example is normative enough to mislead a client or implementer.

Minimum documentation correction:

- Remove `adkSessionId` from the public request example and explicitly list it with all prohibited identity/workflow selectors.
- State that the internal authority may carry the server-derived ADK session outside the public body, but the browser cannot choose it.
- Keep the negative contract oracle: sending `adkSessionId`, `mediationSessionId`, `workflowId`, `subject`, or `tenantId` must be rejected before controller/store access.

Acceptance: the prose, JSON example, security document, deployment document, and final6 request schema all describe one selector-free public body.

### DOC-H-02 — OQ-008 has two incompatible signed envelope identifiers

Severity: **HIGH**

The accepted OQ-008 decision, the payment evidence design, the persistence mapping, and the schema definition in `06` use `mediation-authorization-envelope/v1`. However, the decision-reference sections in `04` and `06`, plus `FIG-API-01`, call the same artifact `mediation-correlation-envelope/v1`.

This is not cosmetic: `schemaVersion`, JWS `typ`, canonical bytes, digest references, offline verification, guarantee binding, and completion-manifest binding depend on the exact identifier. The serialized owner must not expose two names for one signed object.

Minimum documentation correction:

- Replace all stale `mediation-correlation-envelope/v1` references with the accepted `mediation-authorization-envelope/v1`, unless a new ADR intentionally introduces a second artifact.
- Re-run the terminology/link/diagram check and confirm that the authorization envelope and completion manifest remain a one-way pre/post evidence pair.

Acceptance: one exact identifier appears in OQ-008, `04`, `06`, `08`, diagrams, examples, and final6 claims.

### DOC-M-01 — UI field names do not match the authoritative turn response

Severity: **MEDIUM**

`06_API_A2A_CONTRACTS.md` defines the public response with `state` and `version`. `07_UI_TRACE.md` says the UI consumes `workflowState` and `viewVersion`. The documents therefore disagree at the UI/API ownership seam. The `pendingAction` example and enum should also be reconciled with the exact current public-view schema used by plan, payment, refund, wait, and terminal flows.

Minimum documentation correction: publish one exact `mediation-turn-response/1`/`mediation-public-view/1` field table and make `06` and `07` use the same names and pending-action variants.

### DOC-M-02 — The SQLite v4 CAS example contradicts its HMAC owner mapping

Severity: **MEDIUM**

`08_PERSISTENCE_RECOVERY.md` describes `mediation_sessions_v4` as keyed by an HMAC owner scope and states that AAD binds owner/request/session/version/schema. Its sample CAS predicate instead names raw `subject`, `tenant_id`, and `adk_session_id` columns. The same table also documents `version>=1`, while the final v4 mediation contract uses a zero-based initial version.

Minimum documentation correction: express the physical CAS with `scope_key`, mediation session ID, expected state, and expected version, and document the actual `version >= 0` invariant. Keep the semantic four-tuple owner check separately from the physical HMAC index.

### DOC-M-03 — Several historical HIGH closures are not independently traceable from the cited final6 artifacts

Severity: **MEDIUM**

The final6 appendix in `MEDIATOR_PAYMENT_INTEGRATION_IMPLEMENTATION_REVIEW.md` closes H-01 through H-07, but several rows cite only aggregate marker counts or broad suite names. In particular, H-01 claims the legacy production callback received actual paid/free A2A content, while the test report explicitly identifies the exercised browser runtime as the local deterministic callback/agent profile. The three final6 JSON artifacts expose suite totals and high-level browser cases, not a named callback trace or per-finding evidence key.

This does not prove the closures false, and the package correctly leaves all 126 requirement records `PARTIAL`. It does mean the historical closure table is not independently reproducible from the artifacts it cites.

Minimum documentation correction: for each closed B/H finding, cite a named packaged test, captured trace/artifact field, and exact image-bound evidence reference. For production-only seams such as the legacy callback, distinguish contract-test closure from real-browser local-profile evidence. If that evidence is unavailable, label the row `PARTIAL` rather than `CLOSED`.

## 4. Verified strengths

The following checks passed:

- The design set contains exactly 13 files and gives each domain a clear primary owner, reviewer, upstream source, and downstream consumer.
- The requirements contain 139 unique normative H3 IDs. The coverage YAML contains the same 139 IDs exactly once, with no missing or unknown IDs.
- Release scope is exactly 126 required and 13 future-work records; the sets are exclusive and exhaustive.
- Required records remain conservatively `PARTIAL`, future records remain `DESIGNED`, and the package does not call the 139-record release ledger complete.
- Paid, free, and refund normal flows are separated. Free flow explicitly creates no payment/guarantee/settlement/refund record; refund requires a settled, correlated original payment and a distinct approval.
- AP2 roles are sensibly separated: the non-agentic Trusted Surface owns informed consent/user signature, deterministic code validates and processes artifacts, the Shopping Agent may orchestrate, and the signed guarantee is explicitly project-local rather than an AP2 standard artifact.
- The production `legacy` callback and the local `deterministic-local` profile are distinguished in architecture prose. The evidence limitation for the local runtime is captured by DOC-M-03.
- Public mutation is documented as one workflow authority behind `payment_user_agent`; caller-selected workflow/subject authority is rejected in principle.
- Local SQLite v4 durability and Cloud Run memory-backed `EPHEMERAL DEMO` are clearly separated. Cloud Run durability is explicitly `NOT PROVIDED`.
- Authentication, secure session cookie, exact Origin, CSRF, internal signed identity, header stripping, and loopback/default-deny boundaries are documented coherently apart from DOC-H-01.
- All local Markdown link targets and checked fragments resolve within the reviewed set.

## 5. final6 evidence verification

The file digests stated in the plan, test report, and implementation-review appendix match the files on disk:

| Artifact | Observed SHA-256 | Result |
| --- | --- | --- |
| `regression-result-final6.json` | `f64da6ec882b3a6a14f27a8df5448ad971c01c208c3f8bcf6070335edfa84ded` | match |
| `browser-evidence-final6.json` | `1059985e2fac45b8c7c70ed316e2359d1c6da64acc004ebf0207560a3796fa50` | match |
| `ap2-x402-release-validation-final6.json` | `4f4aa723d9a5bc02eec4c09d6f097c749f2d6f6652c66f0dc2b0a72573cf96ce` | match |

All three bind to image `sha256:3e4e089643564e00bd6563d08a575fcf2aa2eff94ae60fd1ca518900022a89f0` and release manifest `sha256:852aeaba0e024469eb35adfa45a1dd6fabd054484d68aa1b58739ddaf8457f37` where applicable.

Observed counts also match:

- canonical regression: payment 285, evaluation 17, jury 13 with 8 allowed skips and no failures/errors;
- release validation: 11 marker groups, each with zero failures/errors/skips or xfails;
- browser evidence: paid, free, refund, and privacy; `completedAfterRefresh=true`; `privateMaterialExposed=false`.

The artifacts themselves preserve the declared limits: `officialX402=NOT RUN`, `onChainSettlement=NOT RUN`, and `conformanceReportDigest=null`.

## 6. NOT RUN and future-work boundary

The package consistently leaves these outside verified final6 scope:

- real Firebase credential/ID-token exchange;
- Vertex ADC, IAM, quota, model availability, and live model probes;
- official x402 wallet/facilitator and on-chain settlement/refund;
- Cloud Run build, push, revision, tag, traffic, observation, and rollback;
- candidate-bound conformance report;
- the 13 future-work items covering advanced crash recovery, response loss, complex retry/concurrency, DNS rebinding, and expanded malicious/price/expiry matrices;
- raw repository green status, because three W&B-authentication tests failed outside the canonical release profile.

These boundaries are suitable and must remain after the documentation corrections.

## 7. Protected-file check

This review did not edit the handoff or payments README. Their observed pre-review hashes were:

- `MEDIATOR_PAYMENT_INTEGRATION_HANDOFF.md`: `9761cb3e3a6a683d45e9be9618f7aef1968f787b747eacd41e5082c3140ab24d`
- `docs/payments/README.md`: `783e9a22f1d5e17486473381f6ae30e266394a7c7d5c75aeb3b0006fdc64edbd`

The only file created by this review is this review record.

## 8. Final verdict

**CHANGES REQUIRED**

The package is close to approval and its release/NOT RUN boundaries are substantially correct. Approval requires closure of DOC-H-01 and DOC-H-02, followed by the three medium consistency/traceability corrections and a repeat of the deterministic documentation checks. This verdict is about documentation quality only; it neither invalidates the recorded local final6 test results nor authorizes Cloud Run or production promotion.

## 9. Closure re-review appendix (2026-08-17)

### 9.1 Scope and decision

This appendix is a static documentation re-review of the corrections made after the initial `CHANGES REQUIRED` decision. It supersedes the current decision in section 8 without deleting the original findings. No implementation, deployment, Cloud Run, HANDOFF, README, or source-design document was changed by this re-review; only this review record was updated.

Current decision: **APPROVED**

This approval means that the reviewed documentation package is internally consistent enough to serve as the current target/current-candidate record. It does not change the documented candidate status: Release-1 requirement closure remains `PARTIAL`, production Legacy callback and canonical authorization/completion evidence remain unclosed where stated, and Cloud Run/production/official x402 gates remain `NOT RUN` or not provided.

### 9.2 Finding closure

| Finding | Re-review result | Closure evidence |
| --- | --- | --- |
| DOC-H-01 public request includes a client session selector | **CLOSED** | `06_API_A2A_CONTRACTS.md` removes `adkSessionId` from `mediation-turn-request/1`, enumerates the exact public fields, explicitly rejects ADK/mediation/workflow/subject/tenant selectors before authority access, and defines the internal ADK session as a server-derived hash of verified tenant and subject. `selectionToken` is fixed to `null` for `/1`. `07` also prohibits returning these selectors from the UI. |
| DOC-H-02 conflicting envelope identifiers | **CLOSED** | The design set has zero occurrences of `mediation-correlation-envelope/v1`. OQ-008 references, `04`, `06`, `08`, the canonical schema/JWS type, and `FIG-API-01` consistently use `mediation-authorization-envelope/v1`. |
| DOC-M-01 UI/API field drift | **CLOSED** | `06` now provides an exact field/variant table for turn response, public view, pending action, and trace. It explicitly excludes `workflowState`, `viewVersion`, and `targetId`; `07` consumes `state`, `version`, `pendingAction.targetRef`, `approvalTarget`, and `approvalTargetDigest`. |
| DOC-M-02 SQLite CAS/version drift | **CLOSED** | `08` documents the actual v4 column/check/index contract, HMAC-derived `scope_key`, absence of raw subject columns, initial `version == 0`, `CHECK(version >= 0)`, and physical CAS over scope key/session/state/version with exact-one row count. Semantic four-tuple ownership remains separate from the stored index. |
| DOC-M-03 historical closure traceability | **CLOSED** | The implementation-review appendix defines exact `R`/`B`/`V` artifact references, names reproducible packaged tests for the historical findings, and no longer claims closure when the required runtime/profile/producer is absent. H-01 is explicitly `PARTIAL — deterministic-local tested; production Legacy NOT RUN`; H-04 is `PARTIAL — canonical pair NOT IMPLEMENTED`; H-05 is limited to functional local refund evidence. All cited test node names checked by this re-review exist in the repository. |

### 9.3 Deterministic checks repeated

| Check | Result |
| --- | --- |
| YAML front matter parse | **PASS** |
| Requirements normative IDs | **139 unique** |
| Design coverage records | **139 unique; exact set match** |
| Release partition | **126 release-1-required / 13 future-work** |
| Verification status partition | **126 PARTIAL / 13 DESIGNED** |
| Missing/unknown/duplicate IDs | **0 / 0 / 0** |
| Missing local link targets | **0** |
| Broken checked local fragments | **0** |
| Stale `mediation-correlation-envelope/v1` in the design set | **0** |
| Named historical evidence tests missing | **0** |

The final6 artifact files remain byte-for-byte bound to the hashes recorded in section 5:

- regression: `f64da6ec882b3a6a14f27a8df5448ad971c01c208c3f8bcf6070335edfa84ded`
- browser: `1059985e2fac45b8c7c70ed316e2359d1c6da64acc004ebf0207560a3796fa50`
- release validation: `4f4aa723d9a5bc02eec4c09d6f097c749f2d6f6652c66f0dc2b0a72573cf96ce`

The HANDOFF and payments README hashes also remain unchanged from section 7.

### 9.4 Final approval boundary

**APPROVED** for documentation quality and internal consistency.

This is not approval of full Release-1 closure or deployment. The package correctly continues to state all of the following:

- the 126 required ledger records are `PARTIAL`, not `PASS`;
- the accepted `mediation-authorization-envelope/v1` plus `mediation-completion-manifest/v1` pair is target design, while final6's current `pre-payment-authorization-envelope/1` implementation is explicitly `PARTIAL` and lacks the canonical completion producer/verifier;
- the production Legacy callback path is `NOT RUN` by final6 browser evidence;
- Cloud Run, real Firebase/Vertex gates, official x402 wallet/facilitator, on-chain behavior, and candidate-bound conformance remain unexecuted or unsupported;
- local SQLite v4 durability must not be promoted into a Cloud Run durability claim.
