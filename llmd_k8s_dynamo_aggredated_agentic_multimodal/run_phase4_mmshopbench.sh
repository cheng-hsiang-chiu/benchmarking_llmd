#!/usr/bin/env bash
# ==============================================================================
# Phase 4: MMShopBench Multi-Turn Concurrency Sweep (Arm A vs. Arm B)
#
# - Arm A (llmd):   aggregation-epp:80 (llm-d Multimodal Router)
# - Arm B (k8s_rr): multimodal-k8s-rr-baseline:8000 (K8s Round-Robin Service)
# - Concurrencies:  8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
# - Iterations:     3 per (Arm, Concurrency) -> 60 total runs
# - Cache Reset:    vLLM (/reset_prefix_cache, /reset_mm_cache, /reset_encoder_cache)
#                   + EPP router rollout restart before EVERY run
# - Storage Clean:  Clears pod /workspace/results, /tmp/aiperf*, and workload-pvc
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="llm-d-multimodal-aggregation"
GUIDE_NAME="aggregation"
MODEL="Qwen/Qwen3-VL-32B-Instruct"
HOST_RESULTS_DIR="${SCRIPT_DIR}/results"
CONCURRENCIES=(${CONCURRENCIES:-8 16 32 64 128 256 512 1024 2048 4096})
ITERATIONS=(${ITERATIONS:-1 2 3})

mkdir -p "${HOST_RESULTS_DIR}"
LOG_FILE="${HOST_RESULTS_DIR}/phase4_sweep.log"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${LOG_FILE}"
}

clean_caches_and_storage() {
  log "==> [Cleanup] Killing any lingering aiperf processes & clearing pod local storage (/workspace/results, /tmp/aiperf*) & PVC storage..."
  kubectl exec -n "${NAMESPACE}" mmshopbench-client -- \
    sh -c "pkill -9 -x aiperf 2>/dev/null || true; rm -rf /workspace/results/* /tmp/aiperf*"
  kubectl exec -n llm-d-p2p-kv-cache-sharing access-to-harness-data-workload-pvc -- \
    sh -c "rm -rf /requests/aiperf-* /requests/inference-perf-*" 2>/dev/null || true

  log "==> [Cache Reset] Resetting KV prefix, MM processor, and vision encoder caches on all 8 vLLM pods..."
  local pod_ips
  pod_ips=$(kubectl get pods -n "${NAMESPACE}" -l llm-d.ai/guide=aggregation --field-selector=status.phase=Running -o jsonpath='{.items[*].status.podIP}')
  kubectl exec -n "${NAMESPACE}" mmshopbench-client -- sh -c "
    set -e
    for ip in ${pod_ips}; do
      curl -sf -X POST http://\${ip}:8000/reset_prefix_cache >/dev/null
      curl -sf -X POST http://\${ip}:8000/reset_mm_cache >/dev/null
      curl -sf -X POST http://\${ip}:8000/reset_encoder_cache >/dev/null
    done
  "

  log "==> [Router Reset] Restarting ${GUIDE_NAME}-epp to clear in-memory prefix hash tree..."
  kubectl rollout restart deployment/"${GUIDE_NAME}-epp" -n "${NAMESPACE}"
  until kubectl rollout status deployment/"${GUIDE_NAME}-epp" -n "${NAMESPACE}" --timeout=180s; do
    log "Warning: rollout status encountered transient API error, retrying in 3s..."
    sleep 3
  done
  sleep 2
}

EPP_IP=$(kubectl get service "${GUIDE_NAME}-epp" -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
K8S_RR_IP=$(kubectl get service multimodal-k8s-rr-baseline -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')

log "========================================================================"
log " Starting MMShopBench Multi-Turn Benchmark Sweep"
log " Arm A (llmd):   http://${EPP_IP}:80"
log " Arm B (k8s_rr): http://${K8S_RR_IP}:8000"
log " Concurrencies:  ${CONCURRENCIES[*]}"
log " Iterations:     ${ITERATIONS[*]}"
log " Results Dir:    ${HOST_RESULTS_DIR}"
log "========================================================================"

for CONC in "${CONCURRENCIES[@]}"; do
  # Ensure conversation_num >= max(576, 2 * CONC) in multiples of 288:
  # c<=256 -> 576 | c=512 -> 1152 | c=1024 -> 2304 | c=2048 -> 4608 | c=4096 -> 9216
  if (( CONC <= 256 )); then
    CONV_NUM=576
  else
    CONV_NUM=$(( (CONC / 256) * 576 ))
  fi

  for ARM in "llmd:${EPP_IP}:80" "k8s_rr:${K8S_RR_IP}:8000"; do
    ARM_NAME="${ARM%%:*}"
    ARM_URL="http://${ARM#*:}"

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

      until [[ "$(kubectl get deployment mm-baseline-nvidia-gpu-vllm-decode -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')" == "8" ]]; do
        log "Waiting for all 8 vLLM decode replicas to be Ready (currently $(kubectl get deployment mm-baseline-nvidia-gpu-vllm-decode -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')/8)..."
        sleep 30
      done

      clean_caches_and_storage

      START_TS=$(date +%s)
      VLLM_IPS=$(kubectl get pods -n "${NAMESPACE}" -l llm-d.ai/guide=aggregation --field-selector=status.phase=Running -o jsonpath='{.items[*].status.podIP}')
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
done

log "========================================================================"
log " All 60 MMShopBench Phase 4A runs (c=8..4096) completed successfully!"
log "========================================================================"

