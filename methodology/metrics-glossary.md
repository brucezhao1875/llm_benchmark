# LLM Serving Metrics Glossary

## Core throughput metrics

| Metric | Meaning | Notes |
|---|---|---|
| Output tokens/s | Generated tokens per second | Primary decode capacity metric |
| Total tokens/s | Input + output tokens per second | Useful for whole-system throughput |
| Prefill tokens/s | Input tokens processed per second | Important for long-context TTFT |
| Request throughput / QPS | Completed requests per second | Only meaningful with input/output shape and SLA |

## Latency metrics

| Metric | Meaning |
|---|---|
| TTFT | Time to first token |
| TPOT | Time per output token |
| ITL | Inter-token latency between streamed tokens |
| E2E | End-to-end request latency |
| p50 | Median percentile |
| p95 | 95th percentile |
| p99 | 99th percentile |

## Serving-system metrics

| Metric | Meaning |
|---|---|
| GPU SM util | GPU compute utilization |
| GPU memory used | Allocated/reserved GPU memory |
| KV cache usage | KV cache pressure and capacity usage |
| Running requests | Requests currently executing inside the engine |
| Waiting requests | Requests queued due to capacity or scheduling |
| Power draw | GPU power consumption |

## QPS interpretation

QPS is a derived business metric:

```text
QPS ~= available output tokens/s / average output tokens
```

This approximation ignores prefill, queueing, KV cache pressure, latency SLA, and error rate. A production-grade QPS claim must include request shape and SLA.
