# `benchmarking_llmd`

Empirical benchmarks, workload recipes, and reproducibility guides for [`llm-d`](https://github.com/llm-d/llm-d).

## Benchmarks

* **[`llmd_k8s_dynamo_aggredated_agentic_multimodal/`](./llmd_k8s_dynamo_aggredated_agentic_multimodal/README.md)**:
  * **3-Arm Aggregated Agentic Multimodal Serving Benchmark (`MMShopBench` on `16 × NVIDIA H200`)** comparing **`llm-d` Multimodal Router (`aggregation-epp`)**, **Kubernetes Service (`Round-Robin`)**, and **NVIDIA Dynamo (`v1.5.1` KV Router)** across `10` concurrency levels (`c = 8` to `4096`, `90` runs total) on `8 × TP=2 Qwen/Qwen3-VL-32B-Instruct` (`vLLM v0.30.0`).
  * 📄 **[Full Benchmark Report & Step-by-Step Reproducibility Guide](./llmd_k8s_dynamo_aggredated_agentic_multimodal/results/MMSHOPBENCH_LLMD_MULTIMODAL_BENCHMARK_REPORT.md)**
  * 📊 **[Aggregated 90-Run CSV Summary (`mmshopbench_phase4_summary.csv`)](./llmd_k8s_dynamo_aggredated_agentic_multimodal/results/mmshopbench_phase4_summary.csv)**
