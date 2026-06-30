# GLM-5.2-FP8 on 8 x NVIDIA H20 Unified Benchmark Report

Date: 2026-06-30  
Model: GLM-5.2-FP8  
Hardware: 8 x NVIDIA H20-3e  
Runtime: vLLM OpenAI-compatible API server  
Primary host: h20-02

## 1. Executive summary

GLM-5.2-FP8 can run on a single 8-card NVIDIA H20 machine with vLLM and tensor parallelism `TP=8`.

The most important conclusions are:

1. The meaningful serving capacity metric is `tokens/s`, especially `output tokens/s`, not isolated QPS.
2. A 32K startup context is suitable as a low-latency online pool candidate.
3. A 256K startup context is usable as a standard long-context / Agent / RAG pool.
4. A 1M startup context did not become ready within the 30-minute test window using the initial vLLM configuration, so 1M should be treated as an experimental ultra-long pool, not a default production setting.
5. MTP/speculative decoding improves output throughput in several decode scenarios, but can worsen ITL, so it should be used selectively for high-throughput or non-strict-streaming pools.
6. A production gateway must be token-aware and route requests by input length, requested output length, SLA, user tier, and backend pool load.

## 2. Environment

| Item | Value |
|---|---|
| Model | GLM-5.2-FP8 |
| Model directory | `/data/models/GLM-5.2-FP8-ms` |
| Runtime | vLLM OpenAI-compatible API server |
| Tensor parallelism | `--tensor-parallel-size 8` |
| KV cache dtype | `--kv-cache-dtype fp8` |
| GPU | 8 x NVIDIA H20-3e |
| GPU topology | all GPU pairs reported as `NV18` |
| Service manager | systemd |
| API endpoint | `/v1/chat/completions` |

Representative baseline command:

```bash
/data/venvs/glm52/bin/vllm serve /data/models/GLM-5.2-FP8-ms \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --served-model-name glm-5.2-fp8 \
  --trust-remote-code
```

## 3. How to read the results

Do not interpret LLM capacity by standalone QPS.

A request is not a fixed unit. These are very different workloads:

| Shape | Meaning |
|---|---|
| 1K input / 128 output | short chat |
| 1K input / 512 output | normal chat |
| 8K input / 1024 output | RAG or Agent response |
| 64K+ input / short output | long-context prefill-heavy task |
| 1K input / 2048 output | long decode-heavy task |

Use this wording instead:

```text
Under <input tokens>/<output tokens> and SLA <TTFT/E2E/ITL/error rate>, the service sustains <X output tokens/s> and <Y req/s>.
```

## 4. Key measured results

### 4.1 32K startup context, baseline

The 32K startup configuration completed the full short-request matrix and boundary probes.

Startup result:

| max-model-len | Ready time | Result |
|---:|---:|---|
| 32768 | 220s | success |

Representative results:

| Scenario | Rate | Output tok/s | Req/s | TTFT p95 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1K / 128 | 8 | 412.01 | 3.22 | 3476ms | 51.08ms | 10398ms |
| 1K / 512 | 4 | 590.28 | 1.15 | 162ms | 44.58ms | 20940ms |
| 1K / 512 | 8 | 785.86 | 1.53 | 171ms | 54.64ms | 26100ms |
| 1K / 1024 | 4 | 645.89 | 0.63 | 157ms | 45.35ms | 43827ms |
| 8K / 512 | 4 | 326.36 | 0.64 | 13365ms | 37.73ms | 34004ms |
| 8K / 1024 | 2 | 426.85 | 0.42 | 196ms | 31.69ms | 31185ms |

32K boundary probes:

| Input | Output | Result | TTFT |
|---:|---:|---|---:|
| 16K | 64 | success | 2473ms |
| 24K | 64 | success | 2586ms |
| 32K | 1 | success | 2602ms |

Interpretation:

1. 32K baseline is a strong candidate for a low-latency online pool.
2. 1K/512 at rate 8 achieved about `786 output tok/s` with `TTFT p95 171ms`.
3. 8K workloads show more TTFT variability and should be treated separately from short chat.

### 4.2 256K startup context, baseline and MTP

The 256K configuration is the current practical long-context baseline.

Representative baseline results:

| Scenario | Rate | Output tok/s | Req/s | TTFT p95 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1K / 512 | 4 | 516.84 | 1.01 | 2562ms | 45.06ms | 30653ms |
| 1K / 1024 | 4 | 644.32 | 0.63 | 145ms | 45.74ms | 44014ms |
| 8K / 1024 | 2 | 256.34 | 0.25 | 24256ms | 31.95ms | 62788ms |

Representative 256K MTP results:

| Scenario | Rate | Output tok/s | Req/s | TTFT p95 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1K / 512 | 4 | 617.01 | 1.21 | 320ms | 112.24ms | 21044ms |
| 1K / 512 | 8 | 785.79 | 1.53 | 508ms | 143.08ms | 27396ms |
| 1K / 1024 | 4 | 722.13 | 0.71 | 352ms | 109.42ms | 41079ms |
| 1K / 1024 | 8 | 889.80 | 0.87 | 486ms | 139.20ms | 50836ms |
| 8K / 1024 | 2 | 436.34 | 0.43 | 295ms | 82.60ms | 31109ms |

Interpretation:

1. MTP improves output throughput in multiple decode scenarios.
2. MTP often worsens ITL p95, so streaming smoothness may degrade.
3. MTP is better suited for high-throughput or batch-like pools than for all low-latency streaming traffic.

### 4.3 1M startup context

The initial 1M startup test used:

```bash
--max-model-len 1048576
```

Result:

| max-model-len | Ready window | Result |
|---:|---:|---|
| 1048576 | 30 minutes | not ready |

The service process entered loading/compile/warmup states, but `/v1/models` did not become ready within the configured 30-minute window.

Interpretation:

1. This does not prove GLM-5.2-FP8 can never run 1M on this hardware.
2. It proves the initial naive 1M vLLM service configuration is not production-ready.
3. 1M requires a separate experiment with longer startup observation, stricter admission control, likely parameter tuning, and a dedicated ultra-long service pool.

## 5. Cost and QPS interpretation

Observed output throughput for useful online shapes is roughly in the several-hundred to high-hundreds `output tok/s` range.

Examples:

| Shape | Output tok/s | Derived req/s |
|---|---:|---:|
| 1K / 128, 32K baseline r8 | 412 | 3.22 |
| 1K / 512, 32K baseline r8 | 786 | 1.53 |
| 1K / 1024, 256K MTP r8 | 890 | 0.87 |
| 1K / 2048, 256K MTP r4 | 729 | 0.36 |

This shows why QPS decreases as output length grows. The model is not simply low-QPS; each request consumes a different amount of decode capacity.

## 6. Recommended production service pools

| Pool | Suggested context | Suggested config | Workload |
|---|---:|---|---|
| low-latency | 32K | baseline | short chat, small RAG, streaming-sensitive traffic |
| standard / agent | 256K | baseline | RAG, Agent, long document tasks |
| high-throughput | 256K | MTP | long decode, batch generation, non-strict-streaming tasks |
| ultra-long | 1M | experimental | async long-context jobs only |

## 7. Gateway requirements

A production gateway must become a model traffic scheduler, not only an HTTP proxy.

Required capabilities:

| Capability | Purpose |
|---|---|
| input token estimation | classify context length before routing |
| output token budget parsing | estimate decode workload from `max_tokens` |
| context tier classification | short / standard / long / ultra-long |
| decode tier classification | short-output / normal-output / long-output |
| pool routing | route to 32K, 256K, MTP, or 1M pool |
| admission control | limit in-flight token pressure per user and pool |
| async long-context jobs | avoid synchronous 1M requests by default |
| per-pool billing | price 1M and long-decode requests differently |
| per-pool observability | track TTFT, ITL, E2E, token throughput, failures |

Initial routing policy:

```text
input <= 8K and output <= 512      -> 32K low-latency pool
input <= 32K and output <= 1024    -> 32K or 256K standard pool
input 32K-256K                    -> 256K long/agent pool
output >= 2048                    -> high-throughput / long-decode pool
input > 256K                      -> ultra-long pool, async or low concurrency
input close to 1M                 -> experimental 1M pool, separately billed
```

## 8. Final recommendation

For this hardware and model combination:

1. Use `32K baseline` for low-latency online serving.
2. Use `256K baseline` for long-context RAG/Agent workloads.
3. Keep `256K MTP` as a high-throughput option, but do not default it for all streaming requests.
4. Treat `1M` as experimental until it can start reliably and complete staged probes.
5. Build Gateway token-aware routing before exposing multiple context pools to users.

## 9. Supporting reports

Detailed stage reports are kept as appendices:

| File | Scope |
|---|---|
| [01-performance-baseline.md](reports/01-performance-baseline.md) | Installation-era baseline, prefill, mixed traffic, early recipe comparison |
| [02-decode-observability-ab.md](reports/02-decode-observability-ab.md) | Decode matrix, GPU/vLLM observability, MTP A/B |
| [03-context-window-comparison.md](reports/03-context-window-comparison.md) | 32K vs 256K vs 1M startup-context comparison |
