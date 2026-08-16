#!/bin/bash
# Update only the fixed payment demo service through a tagged zero-traffic
# revision. Build and push remain separate, mandatory release-gate steps.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: update-payment-demo-cloudrun.sh candidate|verify|promote|rollback|cleanup|status" >&2
    exit 2
fi

readonly ACTION="$1"
readonly PROJECT_ID="gen-lang-client-0585901015"
readonly REGION="asia-northeast1"
readonly SERVICE_NAME="payment-user-agent-demo"
readonly CANDIDATE_ARTIFACT="artifacts/cloud-run-candidate.json"
readonly UPDATE_STATE="artifacts/cloud-run-update-state.json"
readonly E2E_EVIDENCE="artifacts/cloud-run-tag-e2e.json"
readonly DEPLOY_ENV_VARS="EPHEMERAL_CLOUD_RUN_DEMO=true,MEDIATION_STORE_MODE=memory,APP_ENV=ephemeral-demo,DEV_MODE=false"
readonly MAX_SERVICE_TAG_LENGTH=46

case "$ACTION" in
    candidate|verify|promote|rollback|cleanup|status) ;;
    *)
        echo "Refusing Cloud Run update: unknown action ${ACTION}." >&2
        exit 2
        ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail() {
    echo "Refusing Cloud Run update: $*" >&2
    exit 2
}

assert_candidate_tag() {
    local service="$1" tag="$2" service_tag
    [[ "$tag" =~ ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ ]] \
        || fail "candidate traffic tag has an invalid Cloud Run format."
    service_tag="${service}-${tag}"
    [ "${#service_tag}" -le "$MAX_SERVICE_TAG_LENGTH" ] \
        || fail "service and candidate traffic tag exceed the ${MAX_SERVICE_TAG_LENGTH}-character limit."
}

describe_service() {
    gcloud run services describe "$SERVICE_NAME" \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --platform managed \
        --format=json
}

describe_revision_image() {
    local revision="$1"
    gcloud run revisions describe "$revision" \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --platform managed \
        --format='value(status.imageDigest)'
}

assert_revision_ephemeral_profile() {
    local revision="$1" revision_json
    revision_json="$(
        gcloud run revisions describe "$revision" \
            --project "$PROJECT_ID" \
            --region "$REGION" \
            --platform managed \
            --format=json
    )"
    printf '%s' "$revision_json" | jq -e '
        def env($name):
          [.spec.containers[]?.env[]? | select(.name == $name) | .value] | unique;
        (.spec.containers | length) == 1 and
        env("EPHEMERAL_CLOUD_RUN_DEMO") == ["true"] and
        env("MEDIATION_STORE_MODE") == ["memory"] and
        env("APP_ENV") == ["ephemeral-demo"] and
        env("DEV_MODE") == ["false"]
    ' >/dev/null || fail "candidate revision is not the exact ephemeral memory-store profile."
}

assert_fixed_target() {
    local active_project service_names exact_count service_json
    active_project="$(gcloud config get-value project 2>/dev/null)"
    [ "$active_project" = "$PROJECT_ID" ] || fail "active project is not ${PROJECT_ID}."

    service_names="$(
        gcloud run services list \
            --project "$PROJECT_ID" \
            --region "$REGION" \
            --platform managed \
            --format='value(metadata.name)'
    )"
    exact_count="$(printf '%s\n' "$service_names" | awk -v target="$SERVICE_NAME" '$0 == target { count++ } END { print count + 0 }')"
    [ "$exact_count" = "1" ] || fail "the fixed existing service was not found exactly once."

    service_json="$(describe_service)"
    [ "$(printf '%s' "$service_json" | jq -r '.metadata.name')" = "$SERVICE_NAME" ] \
        || fail "service describe returned a different target."
    if printf '%s' "$service_json" | jq -e '
        ([paths | map(tostring) | join(".")] | any(test("cloudsql"; "i"))) or
        ([.. | strings] | any(test("cloudsql"; "i")))
    ' >/dev/null; then
        fail "Cloud SQL configuration is forbidden for the ephemeral demo."
    fi
}

require_state() {
    [ -f "$UPDATE_STATE" ] || fail "deployment state is missing."
    jq -e \
        --arg project "$PROJECT_ID" \
        --arg region "$REGION" \
        --arg service "$SERVICE_NAME" \
        '.schemaVersion == "cloud-run-payment-demo-update/1" and
         .project == $project and .region == $region and .service == $service' \
        "$UPDATE_STATE" >/dev/null || fail "deployment state target is invalid."
}

state_value() {
    jq -r "$1" "$UPDATE_STATE"
}

set_state_status() {
    local status="$1" temporary
    temporary="$(mktemp "${UPDATE_STATE}.XXXXXX")"
    jq --arg status "$status" '.status = $status' "$UPDATE_STATE" >"$temporary"
    chmod 600 "$temporary"
    mv "$temporary" "$UPDATE_STATE"
}

check_default_traffic() {
    local expected_revision="$1" service_json observed count
    service_json="$(describe_service)"
    observed="$(printf '%s' "$service_json" | jq -r '
        [.status.traffic[]? | select((.percent // 0) == 100 and ((.tag // "") == "")) | .revisionName] | unique | .[]?
    ')"
    count="$(printf '%s\n' "$observed" | awk 'NF { count++ } END { print count + 0 }')"
    [ "$count" = "1" ] && [ "$observed" = "$expected_revision" ]
}

assert_default_traffic() {
    local expected_revision="$1"
    check_default_traffic "$expected_revision" \
        || fail "default traffic is not 100% on ${expected_revision}."
}

candidate_tag_url() {
    local tag="$1" revision="$2" service_json
    service_json="$(describe_service)"
    printf '%s' "$service_json" | jq -r \
        --arg tag "$tag" \
        --arg revision "$revision" \
        '[.status.traffic[]? |
          select(.tag == $tag and .revisionName == $revision and ((.percent // 0) == 0)) |
          .url] | unique | if length == 1 then .[0] else empty end'
}

assert_candidate_binding() {
    local old_revision candidate_revision candidate_image tag expected_url observed_image observed_url
    old_revision="$(state_value '.oldRevision')"
    candidate_revision="$(state_value '.candidateRevision')"
    candidate_image="$(state_value '.candidateImage')"
    tag="$(state_value '.candidateTag')"
    expected_url="$(state_value '.candidateUrl')"

    observed_image="$(describe_revision_image "$candidate_revision")"
    [ "$observed_image" = "$candidate_image" ] || fail "candidate revision digest changed."
    assert_revision_ephemeral_profile "$candidate_revision"
    observed_url="$(candidate_tag_url "$tag" "$candidate_revision")"
    [ -n "$observed_url" ] && [ "$observed_url" = "$expected_url" ] \
        || fail "traffic tag no longer targets the exact zero-traffic candidate."
    assert_default_traffic "$old_revision"
}

create_candidate() {
    local image_reference expected_digest registry_digest service_json old_revision old_image old_traffic
    local digest_hex candidate_tag candidate_revision candidate_image candidate_url temporary

    assert_fixed_target
    [ ! -e "$UPDATE_STATE" ] || fail "deployment state already exists; rollback or cleanup it first."

    image_reference="$(
        python3 scripts/cloud_run_candidate.py verify-deploy --artifact "$CANDIDATE_ARTIFACT"
    )"
    [[ "$image_reference" =~ ^asia-northeast1-docker\.pkg\.dev/gen-lang-client-0585901015/secure-mediation-agent/payment-user-agent-demo@sha256:[0-9a-f]{64}$ ]] \
        || fail "candidate is not an immutable image in the fixed repository."
    expected_digest="${image_reference##*@}"
    registry_digest="$(
        gcloud artifacts docker images describe "$image_reference" \
            --project "$PROJECT_ID" \
            --format='value(image_summary.digest)'
    )"
    [ "$registry_digest" = "$expected_digest" ] || fail "registry digest verification failed."

    service_json="$(describe_service)"
    old_revision="$(printf '%s' "$service_json" | jq -r '
        [.status.traffic[]? | select((.percent // 0) == 100 and ((.tag // "") == "")) | .revisionName] |
        unique | if length == 1 then .[0] else empty end
    ')"
    [ -n "$old_revision" ] || fail "the current 100% rollback revision is ambiguous."
    old_image="$(describe_revision_image "$old_revision")"
    [[ "$old_image" =~ @sha256:[0-9a-f]{64}$ ]] || fail "the rollback revision image is not immutable."
    old_traffic="$(printf '%s' "$service_json" | jq -c '.status.traffic')"

    digest_hex="${expected_digest#sha256:}"
    candidate_tag="pc-${digest_hex:0:12}"
    assert_candidate_tag "$SERVICE_NAME" "$candidate_tag"
    if printf '%s' "$service_json" | jq -e --arg tag "$candidate_tag" \
        '.status.traffic[]? | select(.tag == $tag)' >/dev/null; then
        fail "candidate traffic tag already exists."
    fi

    mkdir -p "$(dirname "$UPDATE_STATE")"
    temporary="$(mktemp "${UPDATE_STATE}.XXXXXX")"
    jq -n \
        --arg project "$PROJECT_ID" \
        --arg region "$REGION" \
        --arg service "$SERVICE_NAME" \
        --arg old_revision "$old_revision" \
        --arg old_image "$old_image" \
        --argjson old_traffic "$old_traffic" \
        --arg candidate_image "$image_reference" \
        --arg candidate_tag "$candidate_tag" \
        '{schemaVersion:"cloud-run-payment-demo-update/1", status:"PREFLIGHT",
          project:$project, region:$region, service:$service,
          oldRevision:$old_revision, oldImage:$old_image, oldTraffic:$old_traffic,
          candidateImage:$candidate_image, candidateTag:$candidate_tag,
          candidateRevision:"NOT_CREATED", candidateUrl:"NOT_CREATED"}' >"$temporary"
    chmod 600 "$temporary"
    mv "$temporary" "$UPDATE_STATE"

    gcloud run services update "$SERVICE_NAME" \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --platform managed \
        --image "$image_reference" \
        --port 8080 \
        --memory 2Gi \
        --cpu 1 \
        --min-instances 1 \
        --max-instances 1 \
        --concurrency 1 \
        --timeout 3600s \
        --update-env-vars "$DEPLOY_ENV_VARS" \
        --no-traffic \
        --tag "$candidate_tag"

    service_json="$(describe_service)"
    candidate_revision="$(printf '%s' "$service_json" | jq -r '.status.latestReadyRevisionName')"
    [ -n "$candidate_revision" ] && [ "$candidate_revision" != "$old_revision" ] \
        || fail "Cloud Run did not create a distinct ready candidate revision."
    candidate_image="$(describe_revision_image "$candidate_revision")"
    [ "$candidate_image" = "$image_reference" ] || fail "ready revision image differs from candidate."
    assert_revision_ephemeral_profile "$candidate_revision"
    candidate_url="$(candidate_tag_url "$candidate_tag" "$candidate_revision")"
    [ -n "$candidate_url" ] || fail "candidate tag URL was not created at zero traffic."
    [[ "$candidate_url" =~ ^https://[a-z0-9-]+\.run\.app$ ]] \
        || fail "candidate tag URL is not an exact Cloud Run HTTPS origin."
    assert_default_traffic "$old_revision"

    temporary="$(mktemp "${UPDATE_STATE}.XXXXXX")"
    jq \
        --arg revision "$candidate_revision" \
        --arg url "$candidate_url" \
        '.status = "CANDIDATE" | .candidateRevision = $revision | .candidateUrl = $url' \
        "$UPDATE_STATE" >"$temporary"
    chmod 600 "$temporary"
    mv "$temporary" "$UPDATE_STATE"

    echo "CANDIDATE_CREATED revision=${candidate_revision} image=${candidate_image} url=${candidate_url}"
    echo "Default service traffic remains on ${old_revision}."
}

verify_candidate() {
    local status candidate_revision candidate_image candidate_url candidate_tag health
    assert_fixed_target
    require_state
    status="$(state_value '.status')"
    [ "$status" = "CANDIDATE" ] || fail "candidate verification requires CANDIDATE state."
    assert_candidate_binding

    candidate_revision="$(state_value '.candidateRevision')"
    candidate_image="$(state_value '.candidateImage')"
    candidate_url="$(state_value '.candidateUrl')"
    candidate_tag="$(state_value '.candidateTag')"
    health="$(curl --fail --silent --show-error --proto '=https' --tlsv1.2 "${candidate_url}/health")"
    [ "$health" = "OK" ] || fail "candidate health probe returned an unexpected body."

    [ -f "$E2E_EVIDENCE" ] || fail "tag-bound E2E evidence is missing."
    jq -e \
        --arg project "$PROJECT_ID" \
        --arg region "$REGION" \
        --arg service "$SERVICE_NAME" \
        --arg revision "$candidate_revision" \
        --arg image "$candidate_image" \
        --arg url "$candidate_url" \
        --arg tag "$candidate_tag" \
        '.schemaVersion == "cloud-run-tag-e2e/1" and .status == "PASS" and
         .project == $project and .region == $region and .service == $service and
         .revision == $revision and .image == $image and .url == $url and .tag == $tag and
         .publicDurabilityProfile == "ephemeral-demo" and
         .readiness.status == "ready" and
         .readiness.target == "ephemeral-cloud-run-demo" and
         .readiness.durability == "NOT PROVIDED" and
         .readiness.mediationStore == {
           mode:"memory", durabilityProfile:"ephemeral-demo", schemaVersion:null,
           writable:true, decryptable:true
         } and
         ([.readiness.checks.mediationStoreMode,
           .readiness.checks.mediationStoreProfile,
           .readiness.checks.mediationStoreSchema,
           .readiness.checks.mediationStoreProbe] | all(. == true)) and
         ([.checks.readiness, .checks.modelProbe, .checks.paid, .checks.free,
           .checks.refund, .checks.browser, .checks.publicBoundary] | all(. == "PASS"))' \
        "$E2E_EVIDENCE" >/dev/null || fail "E2E evidence is incomplete or bound to another candidate."

    set_state_status "VERIFIED"
    echo "CANDIDATE_VERIFIED revision=${candidate_revision} image=${candidate_image} url=${candidate_url}"
}

promote_candidate() {
    local status candidate_revision old_revision
    assert_fixed_target
    require_state
    status="$(state_value '.status')"
    [ "$status" = "VERIFIED" ] || fail "promotion requires VERIFIED state."
    assert_candidate_binding
    candidate_revision="$(state_value '.candidateRevision')"
    old_revision="$(state_value '.oldRevision')"

    gcloud run services update-traffic "$SERVICE_NAME" \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --platform managed \
        --to-revisions "${candidate_revision}=100"

    if ! check_default_traffic "$candidate_revision"; then
        echo "Promotion verification failed; restoring ${old_revision}." >&2
        gcloud run services update-traffic "$SERVICE_NAME" \
            --project "$PROJECT_ID" \
            --region "$REGION" \
            --platform managed \
            --to-revisions "${old_revision}=100"
        exit 4
    fi
    set_state_status "PROMOTED"
    echo "CANDIDATE_PROMOTED revision=${candidate_revision}"
}

rollback_candidate() {
    local old_revision old_image observed_image tag service_json
    assert_fixed_target
    require_state
    old_revision="$(state_value '.oldRevision')"
    old_image="$(state_value '.oldImage')"
    tag="$(state_value '.candidateTag')"
    observed_image="$(describe_revision_image "$old_revision")"
    [ "$observed_image" = "$old_image" ] || fail "saved rollback revision digest changed."

    gcloud run services update-traffic "$SERVICE_NAME" \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --platform managed \
        --to-revisions "${old_revision}=100"
    assert_default_traffic "$old_revision"

    service_json="$(describe_service)"
    if printf '%s' "$service_json" | jq -e --arg tag "$tag" \
        '.status.traffic[]? | select(.tag == $tag)' >/dev/null; then
        gcloud run services update-traffic "$SERVICE_NAME" \
            --project "$PROJECT_ID" \
            --region "$REGION" \
            --platform managed \
            --remove-tags "$tag"
    fi
    set_state_status "ROLLED_BACK"
    echo "ROLLBACK_PASS revision=${old_revision} image=${old_image}"
}

cleanup_candidate() {
    local status tag service_json next_status
    assert_fixed_target
    require_state
    status="$(state_value '.status')"
    case "$status" in
        PROMOTED) next_status="CLEANED_AFTER_PROMOTION" ;;
        ROLLED_BACK) next_status="CLEANED_AFTER_ROLLBACK" ;;
        *) fail "cleanup requires PROMOTED or ROLLED_BACK state." ;;
    esac
    tag="$(state_value '.candidateTag')"
    service_json="$(describe_service)"
    if printf '%s' "$service_json" | jq -e --arg tag "$tag" \
        '.status.traffic[]? | select(.tag == $tag)' >/dev/null; then
        gcloud run services update-traffic "$SERVICE_NAME" \
            --project "$PROJECT_ID" \
            --region "$REGION" \
            --platform managed \
            --remove-tags "$tag"
    fi
    set_state_status "$next_status"
    echo "CANDIDATE_TAG_CLEANED tag=${tag}"
}

case "$ACTION" in
    candidate) create_candidate ;;
    verify) verify_candidate ;;
    promote) promote_candidate ;;
    rollback) rollback_candidate ;;
    cleanup) cleanup_candidate ;;
    status)
        require_state
        jq '{status,project,region,service,oldRevision,oldImage,candidateRevision,candidateImage,candidateTag,candidateUrl}' "$UPDATE_STATE"
        ;;
esac
