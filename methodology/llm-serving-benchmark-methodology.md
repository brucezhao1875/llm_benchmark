# LLM Serving Benchmark Methodology

## 1. Define request shapes

Always define workloads by token shape:

| Field | Example |
|---|---|
| Input tokens | 1K, 8K, 32K, 256K, 1M |
| Output tokens | 128, 512, 1024, 2048 |
| Request rate | 1, 2, 4, 8 req/s |
| Concurrency | Observed max in-flight requests |
| Streaming | true / false |

## 2. Separate prefill and decode

Use different tests for different bottlenecks:

| Test type | Purpose |
|---|---|
| Decode matrix | Measures output throughput, TPOT, ITL, E2E |
| Prefill probe | Measures TTFT and long-context input processing |
| Mixed traffic | Measures interference between short and long requests |
| A/B recipe test | Compares serving configurations such as MTP vs baseline |
| Context-window startup test | Compares service behavior under different `max-model-len` values |

## 3. Collect client and server metrics together

Client-side metrics alone cannot explain bottlenecks. Collect both sides:

| Client | Server |
|---|---|
| output tokens/s | GPU SM util |
| TTFT / TPOT / ITL / E2E | GPU memory and power |
| request throughput | KV cache usage |
| error rate | running/waiting requests |

## 4. Use bounded long-context tests

For 1M-context validation, avoid large decode outputs. Use staged probes:

```text
384K input / 1 output
512K input / 1 output
768K input / 1 output
1M input / 1 output
1M input / 16 output
```

Each probe must have a hard timeout and fail-stop rule.

## 5. Report limitations

Every report should state:

1. Sample size.
2. Whether workload is synthetic or real.
3. Runtime and serving configuration.
4. Whether results are same-run A/B or historical comparison.
5. Whether percentiles are statistically meaningful.
6. Current service state after the test.
