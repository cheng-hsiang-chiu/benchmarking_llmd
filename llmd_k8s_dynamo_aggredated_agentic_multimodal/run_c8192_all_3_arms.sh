#!/usr/bin/env bash
# ==============================================================================
# Resume Concurrency = 8192 for Arm A (llmd) & Arm B (k8s_rr)
# (Arm C dynamo c=8192 iter 1..3 is already completed)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="llm-d-multimodal-aggregation"
WORKLOAD_DIR="${SCRIPT_DIR}"
LOG_FILE="${WORKLOAD_DIR}/results/phase4_c8192_all_arms.log"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${LOG_FILE}"
}

log "=== 1. Cleaning partial results/llmd/c8192 from OOMKilled attempt ==="
rm -rf "${WORKLOAD_DIR}/results/llmd/c8192" "${WORKLOAD_DIR}/results/k8s_rr/c8192"

log "=== 2. Scaling up aggregation-epp memory (256Gi) and CPU (32) on gpu-pool for c=8192 ==="
kubectl patch deployment aggregation-epp -n "${NAMESPACE}" --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/nodeSelector", "value": {"cloud.google.com/gke-accelerator": "nvidia-h200-141gb"}},
  {"op": "add", "path": "/spec/template/spec/tolerations", "value": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]},
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources", "value": {"requests": {"cpu": "8", "memory": "64Gi"}, "limits": {"cpu": "32", "memory": "256Gi"}}},
  {"op": "replace", "path": "/spec/template/spec/containers/1/resources", "value": {"requests": {"cpu": "8", "memory": "64Gi"}, "limits": {"cpu": "32", "memory": "256Gi"}}},
  {"op": "replace", "path": "/spec/template/spec/containers/1/livenessProbe/timeoutSeconds", "value": 10},
  {"op": "replace", "path": "/spec/template/spec/containers/1/readinessProbe/timeoutSeconds", "value": 10}
]'
kubectl rollout status deployment/aggregation-epp -n "${NAMESPACE}" --timeout=180s

log "=== 5. Running Concurrency=8192 on Arm A (llmd) & Arm B (k8s_rr) (3 iterations each) ==="
CONCURRENCIES="8192" ITERATIONS="1 2 3" "${WORKLOAD_DIR}/run_phase4_mmshopbench.sh"

log "========================================================================"
log " ALL 9 RUNS FOR CONCURRENCY=8192 COMPLETED SUCCESSFULLY!"
log "========================================================================"
