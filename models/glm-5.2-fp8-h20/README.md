# GLM-5.2-FP8 on 8 x NVIDIA H20 Benchmark

This directory contains the benchmark material for serving GLM-5.2-FP8 on a single 8-GPU NVIDIA H20 machine using vLLM.

## Start here

Read the unified report first:

[report.md](report.md)

The `reports/` directory contains detailed stage reports that support the unified report.

## Environment summary

| Item | Value |
|---|---|
| Model | GLM-5.2-FP8 |
| Model path | `/data/models/GLM-5.2-FP8-ms` |
| Hardware | 8 x NVIDIA H20-3e |
| Runtime | vLLM OpenAI-compatible API server |
| Tensor parallelism | TP=8 |
| KV cache dtype | FP8 |
| Primary host | h20-02 |

## Document map

| Path | Purpose |
|---|---|
| [report.md](report.md) | Unified benchmark report and recommendations |
| [reports/01-performance-baseline.md](reports/01-performance-baseline.md) | Baseline performance, prefill, mixed traffic |
| [reports/02-decode-observability-ab.md](reports/02-decode-observability-ab.md) | Decode matrix, observability, MTP A/B |
| [reports/03-context-window-comparison.md](reports/03-context-window-comparison.md) | 32K / 256K / 1M startup-context comparison |
| [plans/glm52-performance-test-plan.md](plans/glm52-performance-test-plan.md) | General performance test plan and execution log |
| [plans/glm52-context-window-comparison-test-plan.md](plans/glm52-context-window-comparison-test-plan.md) | Context-window comparison test plan |
| [notes/glm52-h20-installation-notes.md](notes/glm52-h20-installation-notes.md) | Installation and deployment notes |

## Headline findings

1. Token throughput is the primary capacity metric; QPS must be tied to request shape and SLA.
2. 32K startup context is a strong low-latency pool candidate.
3. 256K startup context is usable for long-context / RAG / Agent workloads.
4. 1M startup context did not become ready within the initial 30-minute test window.
5. MTP improves throughput but can worsen ITL, so it should be used selectively.
6. Gateway should route by estimated input tokens, requested output tokens, SLA, and user tier.
