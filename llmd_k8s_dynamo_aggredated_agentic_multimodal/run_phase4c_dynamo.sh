#!/usr/bin/env bash
# ==============================================================================
# Phase 4C: MMShopBench Multi-Turn Concurrency Sweep (Arm C: NVIDIA Dynamo)
#
# - Arm C (dynamo): dynamo-frontend:8000 (NVIDIA Dynamo KV Router, --router-mode kv)
# - Concurrencies:  8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
# - Iterations:     3 per Concurrency -> 30 total runs
# - Cache Reset:    POST /engine/reset_all_caches (resets KV prefix, MM processor,
#                   and vision encoder caches on all 8 dynamo-vllm-worker pods)
#                   + dynamo-frontend rollout restart before EVERY run
# - Storage Clean:  Clears pod /workspace/results, /tmp/aiperf*, and workload-pvc
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="llm-d-multimodal-aggregation"
MODEL="Qwen/Qwen3-VL-32B-Instruct"
HOST_RESULTS_DIR="${SCRIPT_DIR}/results"
CONCURRENCIES=(${CONCURRENCIES:-8 16 32 64 128 256 512 1024 2048 4096 8192})
ITERATIONS=(${ITERATIONS:-1 2 3})

mkdir -p "${HOST_RESULTS_DIR}"
LOG_FILE="${HOST_RESULTS_DIR}/phase4c_dynamo_sweep.log"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${LOG_FILE}"
}

clean_caches_and_storage() {
  log "==> [Cleanup] Killing any lingering aiperf processes & clearing pod local storage (/workspace/results, /tmp/aiperf*) & PVC storage..."
  kubectl exec -n "${NAMESPACE}" mmshopbench-client -- \
    sh -c "pkill -9 -x aiperf 2>/dev/null || true; rm -rf /workspace/results/* /tmp/aiperf*"
  kubectl exec -n llm-d-p2p-kv-cache-sharing access-to-harness-data-workload-pvc -- \
    sh -c "rm -rf /requests/aiperf-* /requests/inference-perf-*" 2>/dev/null || true

  log "==> [Cache Reset] Resetting KV prefix, MM processor, and vision encoder caches on all 8 dynamo-vllm-worker pods..."
  local pod_ips
  pod_ips=$(kubectl get pods -n "${NAMESPACE}" -l app=dynamo-vllm-worker --field-selector=status.phase=Running -o jsonpath='{.items[*].status.podIP}')
  kubectl exec -n "${NAMESPACE}" mmshopbench-client -- sh -c "
    set -e
    for ip in ${pod_ips}; do
      curl -sf -X POST http://\${ip}:8000/engine/reset_all_caches -H 'Content-Type: application/json' -d '{}' >/dev/null
    done
  "

  log "==> [Router Reset] Restarting dynamo.frontend process to clear KV router state..."
  kubectl exec -n "${NAMESPACE}" deploy/dynamo-frontend -- pkill -f "dynamo.frontend" || true
  sleep 2

  local frontend_ip
  frontend_ip=$(kubectl get service dynamo-frontend -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
  kubectl exec -n "${NAMESPACE}" mmshopbench-client -- sh -c "
    for i in \$(seq 1 60); do
      if curl -sf http://${frontend_ip}:8000/v1/models | grep -q 'Qwen3-VL-32B-Instruct'; then
        exit 0
      fi
      sleep 1
    done
    exit 1
  "
  sleep 2
}

kubectl rollout status deployment/dynamo-frontend -n "${NAMESPACE}" --timeout=180s
FRONTEND_IP=$(kubectl get service dynamo-frontend -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
ARM_NAME="dynamo"
ARM_URL="http://${FRONTEND_IP}:8000"

log "========================================================================"
log " Starting MMShopBench Multi-Turn Benchmark Sweep (Arm C: NVIDIA Dynamo)"
log " Arm C (dynamo): ${ARM_URL}"
log " Concurrencies:  ${CONCURRENCIES[*]}"
log " Iterations:     ${ITERATIONS[*]}"
log " Results Dir:    ${HOST_RESULTS_DIR}"
log "========================================================================"

for CONC in "${CONCURRENCIES[@]}"; do
  if (( CONC <= 256 )); then
    CONV_NUM=576
  else
    CONV_NUM=$(( (CONC / 256) * 576 ))
  fi

  for ITER in "${ITERATIONS[@]}"; do
    SEED=$(( 42 + CONC * 100 + ITER ))
    POD_OUT_DIR="/workspace/results/${ARM_NAME}_c${CONC}_iter${ITER}"
    HOST_OUT_DIR="${HOST_RESULTS_DIR}/${ARM_NAME}/c${CONC}/iter_${ITER}"

    if [[ -f "${HOST_OUT_DIR}/profile_export_aiperf.json" ]]; then
      log ">>> [SKIP] Already completed: Arm=${ARM_NAME} | Concurrency=${CONC} | Iter=${ITER}"
      continue
    fi

    mkdir -p "${HOST_OUT_DIR}"
    log "------------------------------------------------------------------------"
    log ">>> [START] Arm=${ARM_NAME} | Concurrency=${CONC} | Iter=${ITER}/3 | Convs=${CONV_NUM} | Seed=${SEED}"
    log "------------------------------------------------------------------------"

    until [[ "$(kubectl get deployment dynamo-vllm-worker -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')" == "8" ]]; do
      log "Waiting for all 8 dynamo-vllm-worker replicas to be Ready (currently $(kubectl get deployment dynamo-vllm-worker -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')/8)..."
      sleep 15
    done

    clean_caches_and_storage

    START_TS=$(date +%s)
    VLLM_IPS=$(kubectl get pods -n "${NAMESPACE}" -l app=dynamo-vllm-worker --field-selector=status.phase=Running -o jsonpath='{.items[*].status.podIP}')
    METRICS_URLS=()
    for ip in ${VLLM_IPS}; do
      METRICS_URLS+=("http://${ip}:8000/metrics")
    done

    kubectl exec -n "${NAMESPACE}" mmshopbench-client -- sh -c "
      ulimit -n 65535 2>/dev/null || true
      exec aiperf profile \
        --model '${MODEL}' \
        --url '${ARM_URL}' \
        --endpoint-type chat \
        --streaming \
        --input-file /workspace/mmshopbench/mmshopbench_aiperf_multiturn.jsonl \
        --custom-dataset-type multi_turn \
        --conversation-num '${CONV_NUM}' \
        --conversation-turn-delay-mean 2000 \
        --concurrency '${CONC}' \
        --extra-inputs '{\"ignore_eos\": true}' \
        --random-seed '${SEED}' \
        --ui-type none \
        --no-gpu-telemetry \
        --server-metrics ${METRICS_URLS[*]} \
        --output-artifact-dir '${POD_OUT_DIR}'
    "
    END_TS=$(date +%s)
    ELAPSED=$(( END_TS - START_TS ))

    # Remove raw Base64 request/response blobs inside pod before copying to save disk space
    kubectl exec -n "${NAMESPACE}" mmshopbench-client -- \
      sh -c "rm -f ${POD_OUT_DIR}/inputs.json ${POD_OUT_DIR}/*_raw.jsonl" 2>/dev/null || true

    log "==> Copying results to ${HOST_OUT_DIR} (elapsed: ${ELAPSED}s)..."
    kubectl cp "${NAMESPACE}/mmshopbench-client:${POD_OUT_DIR}/." "${HOST_OUT_DIR}/"

    log "==> Clearing pod local storage after copy..."
    kubectl exec -n "${NAMESPACE}" mmshopbench-client -- \
      sh -c "rm -rf ${POD_OUT_DIR} /tmp/aiperf*"

    log ">>> [DONE] Arm=${ARM_NAME} | Concurrency=${CONC} | Iter=${ITER}/3 (${ELAPSED}s)"
  done
done

log "========================================================================"
log " All 30 MMShopBench Phase 4C runs (Arm C: Dynamo, c=8..4096) completed!"
log "========================================================================"
