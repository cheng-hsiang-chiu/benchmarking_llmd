# `llm-d` vs. Kubernetes Round-Robin vs. NVIDIA Dynamo: Aggregated Agentic Multimodal Serving Benchmark (`MMShopBench`)

This directory contains the complete benchmark report, 99-run empirical results (`c = 8` to `8192`), comparison plots, and reproducibility recipes evaluating **`llm-d`'s Multimodal Optimized Baseline** against **Kubernetes Service (Round-Robin)** and **NVIDIA Dynamo (`v1.5.1` KV Router)** on **`16 × NVIDIA H200 (141 GB)` GPUs** (`8` replicas $\times$ `TP=2`, `Qwen/Qwen3-VL-32B-Instruct`, `vLLM v0.30.0`) using real-world multi-turn e-commerce shopping logs from [`kellyhao/mmshopbench`](https://huggingface.co/datasets/kellyhao/mmshopbench).

👉 **[Read the Full Benchmark Report (`results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md`)](./results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md)**

---

## 1. Key Findings (`99` Runs Across `c = 8 .. 8192`, `0` Failed Requests)

* **Highest Multimodal Prefix-Cache Hit Rate (`85.3%–89.7%` across `c = 8..4096`; `68.4%` at `c = 8192`)**:
  * **Arm A (`llm-d` Multimodal EPP Router)** hashes both text and image blocks (using dynamic $32 \times 32\text{ px}$ patch estimation in `token-producer` and `affinityThreshold: 0.6` in `prefix-cache-affinity-filter`) to keep multi-turn shopping sessions—including the `28.3%` of image turns uploaded on Round 2+—pinned to the replica holding prior KV and ViT caches.
  * By comparison, **Arm C (`NVIDIA Dynamo KV Router`)** achieves `69.3%–83.0%` across `c = 8..4096` (`60.8%` at `c = 8192`), and **Arm B (`Kubernetes Round-Robin`)** drops to `49.0%–63.8%` at `c = 128..4096` (`53.4%` at `c = 8192`).
* **Up to `3.94×` Lower TTFT p90 at Peak Throughput (`c = 4096`) & `2.50×` Lower TTFT p50 Under KV Saturation (`c = 8192`)**:
  * At `c = 256`, `llm-d` achieves **`61.02 ms` TTFT p50** and **`259.16 ms` TTFT p90**, outperforming **Dynamo** (`109.93 ms` p50 / `395.72 ms` p90, **`1.80×` / `1.53×` faster**) and **K8s Round-Robin** (`158.41 ms` p50 / `645.55 ms` p90, **`2.60×` / `2.49×` faster**).
  * At `c = 4096` (peak throughput), `llm-d` holds TTFT p50 to **`311.85 ms`**, TTFT p90 to **`1,014.91 ms`**, and TTFT p99 to **`2,796.55 ms`**—cutting TTFT p90 by **`22.8%` vs. Dynamo** (`1,315.28 ms`) and **`74.6%` (`3.94×`) vs. K8s Round-Robin** (`3,997.23 ms`).
  * At `c = 8192` (`1,024` concurrent sessions/replica, beyond aggregate H200 KV capacity), `llm-d` holds TTFT p50 to **`9,987.56 ms`**, cutting TTFT p50 by **`47.6%` (`1.91×` faster) vs. Dynamo** (`19,047.79 ms`) and **`60.1%` (`2.50×` faster) vs. K8s Round-Robin** (`25,009.71 ms`).
* **`+19.8%` Higher Peak Throughput vs. NVIDIA Dynamo & `+53.9%` vs. K8s Round-Robin (`c = 4096`)**:
  * At `c = 4096`, `llm-d` reaches **`121,280.2 total tokens/sec`** (**`34,233.3 output tokens/sec`**, `157.73 req/s`), surpassing **NVIDIA Dynamo** (`101,060.3 total` / `28,573.8 output tok/s`) by **`+20.0%` / `+19.8%`** and **K8s Round-Robin** (`78,759.3 total` / `22,246.1 output tok/s`) by **`+54.0%` / `+53.9%`**.

---

## 2. Summary Visualizations

| TTFT p50 vs. Concurrency | TTFT p90 vs. Concurrency |
| :---: | :---: |
| ![TTFT p50](./results/plots/mmshopbench_ttft_p50_vs_concurrency.png) | ![TTFT p90](./results/plots/mmshopbench_ttft_p90_vs_concurrency.png) |
| **ITL p50 vs. Concurrency** | **ITL p90 vs. Concurrency** |
| ![ITL p50](./results/plots/mmshopbench_itl_p50_vs_concurrency.png) | ![ITL p90](./results/plots/mmshopbench_itl_p90_vs_concurrency.png) |
| **Total Token Throughput vs. Concurrency** | **Output Token Throughput vs. Concurrency** |
| ![Total Token Throughput](./results/plots/mmshopbench_total_token_throughput_vs_concurrency.png) | ![Output Token Throughput](./results/plots/mmshopbench_output_token_throughput_vs_concurrency.png) |
| **Hardware & Effective Roofline (`TP=2` H200 Replica)** | **Serving Roofline: Throughput vs. Interactivity Pareto** |
| ![Hardware Roofline](./results/plots/mmshopbench_hardware_roofline.png) | ![Serving Roofline Pareto](./results/plots/mmshopbench_serving_roofline_pareto.png) |

---

## 3. Repository Structure

```text
llmd_k8s_dynamo_aggredated_agentic_multimodal/
├── README.md                                           # Overview, headline findings, and directory index
├── dynamo-arm-c.yaml                                   # Complete Kubernetes manifest for Arm C (NVIDIA Dynamo v1.5.1 + 8x TP=2 vLLM)
├── generate_roofline_plots.py                          # Roofline & Throughput-Interactivity Pareto plot generator
├── run_phase4_mmshopbench.sh                           # Automated concurrency sweep script for Arm A (llm-d) & Arm B (K8s Round-Robin)
├── run_phase4c_dynamo.sh                               # Automated concurrency sweep script for Arm C (NVIDIA Dynamo KV Router)
├── run_c8192_all_3_arms.sh                             # High-concurrency (c=8192) EPP memory scaling & sweep helper
└── results/
    ├── MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md # Full technical report + Section 4.3 Roofline Analysis + Section 6 reproducibility guide
    ├── mmshopbench_phase4_summary.csv                  # 33-row mean ± std summary table across all 99 runs (c=8..8192 x 3 arms)
    ├── plots/                                          # 8 high-resolution (300 DPI) comparison & roofline PNG charts
    ├── llmd/c{8..8192}/iter_{1..3}/                    # Per-iteration AIPerf & vLLM Prometheus metrics for Arm A (llm-d)
    ├── k8s_rr/c{8..8192}/iter_{1..3}/                  # Per-iteration AIPerf & vLLM Prometheus metrics for Arm B (K8s Round-Robin)
    └── dynamo/c{8..8192}/iter_{1..3}/                  # Per-iteration AIPerf & vLLM Prometheus metrics for Arm C (NVIDIA Dynamo)
```

> **Note on Excluded Raw Files**: Per-turn raw SSE trace dumps (`results/**/profile_export.jsonl`, up to `399 MB` per file at `c = 8192`) and downloaded HuggingFace images (`mmshopbench/`) are excluded via `.gitignore`. The dataset and images are downloaded automatically using [`convert_mmshopbench_to_aiperf.py`](https://github.com/llm-d/llm-d-benchmark/blob/main/workload/profiles/aiperf/agentic_multimodal/convert_mmshopbench_to_aiperf.py).

---

## 4. Reproducing the Benchmark

Complete step-by-step instructions and Kubernetes recipes for all three arms are documented in **[Section 6 of `results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md`](./results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md#6-step-by-step-reproducibility-guide--complete-recipes)**:

1. **[Section 6.1](./results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md#61-common-prerequisites--workload-preparation-shared-across-all-3-arms)**: Common Prerequisites & `kellyhao/mmshopbench` Workload Preparation ([`llm-d-benchmark/workload/profiles/aiperf/agentic_multimodal`](https://github.com/llm-d/llm-d-benchmark/tree/main/workload/profiles/aiperf)).
2. **[Section 6.2](./results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md#62-shared-model-server-recipe-for-arms-a--b-8--tp2-qwen3-vl-32b-instruct-on-16--h200)**: Shared `8 × TP=2 Qwen3-VL-32B-Instruct` Model Server Deployment on `16 × NVIDIA H200` ([`llm-d` Multimodal Aggregation Guide](https://github.com/llm-d/llm-d/blob/main/guides/multimodal-serving/aggregation/README.md)).
3. **[Section 6.3](./results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md#63-arm-a-step-by-step-instructions--recipe-llm-d-multimodal-router-aggregation-epp)**: Arm A (`llm-d` Multimodal EPP Router) Step-by-Step Instructions & `aggregation.values.yaml` Recipe.
4. **[Section 6.4](./results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md#64-arm-b-step-by-step-instructions--recipe-kubernetes-round-robin-baseline-multimodal-k8s-rr-baseline)**: Arm B (`Kubernetes Round-Robin`) Step-by-Step Instructions & `ClusterIP` Recipe.
5. **[Section 6.5](./results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md#65-arm-c-step-by-step-instructions--recipe-nvidia-dynamo-v151-kv-router)**: Arm C (`NVIDIA Dynamo v1.5.1 KV Router`) Step-by-Step Instructions & [`dynamo-arm-c.yaml`](./dynamo-arm-c.yaml) Recipe.
