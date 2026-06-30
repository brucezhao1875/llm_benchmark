# GLM-5.2-FP8 on H20-02 性能测试报告

日期：2026-06-30  
机器：h20-02  
模型：GLM-5.2-FP8  
部署方式：vLLM OpenAI API Server / systemd service  
测试目标：验证单机 8 卡 H20 上 GLM-5.2-FP8 的基础服务能力、Decode 吞吐、长上下文 Prefill、混合业务流量，以及 vLLM Recipe/MTP 配方的收益。

## 1. 当前结论

1. 单机 8 卡 H20 可以运行 GLM-5.2-FP8，当前服务已经通过 vLLM systemd 方式启动。
2. 当前服务最大上下文配置为 `262144` tokens，不是最早的 `32768` tokens；因此本报告结果反映的是长上下文配置下的服务表现。
3. Decode 吞吐会随着并发/请求注入速率提升而提升，但单请求延迟、TPOT、ITL、E2E 会变差。
4. 长上下文主要受 Prefill 限制，TTFT 随输入长度近似线性增长；128K/256K 请求不适合与短请求混跑。
5. 混合业务流量下，长请求会显著拖慢短请求；正式上线时建议按输入 token 长度做网关分流。
6. Recipe/MTP 配方在 512 in / 512 out 场景下提升了输出吞吐，但 ITL 变差；它更像是一个需要按业务目标验证的配方，而不是无条件最优配置。
7. 当前服务仍运行 Recipe/MTP 配方；如作为生产基线，需要确认是否保留 MTP，或恢复到非 MTP baseline。

## 2. 硬件和软件环境

### 2.1 硬件

| 项目 | 配置 |
|---|---:|
| GPU | 8 x NVIDIA H20-3e |
| 单卡显存 | 约 143,771 MiB |
| GPU 拓扑 | 任意 GPU 两两之间显示 `NV18` |
| CPU NUMA | GPU0-3 在 NUMA0，GPU4-7 在 NUMA1 |
| 数据盘 | `/data`，ext4，约 11T |

### 2.2 软件

| 项目 | 版本/路径 |
|---|---|
| 模型目录 | `/data/models/GLM-5.2-FP8-ms` |
| Python venv | `/data/venvs/glm52` |
| vLLM | `0.23.0` |
| PyTorch | `2.11.0+cu130` |
| transformers | `5.12.1` |
| API Endpoint | `http://127.0.0.1:8000/v1/chat/completions` |
| served model name | `glm-5.2-fp8` |

## 3. 当前 systemd 启动配置

服务名：`glm52-vllm.service`

当前服务状态：`active`

当前实际启动命令：

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
  --trust-remote-code \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 5 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice
```

systemd 关键环境变量：

```ini
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
Environment=PATH=/data/venvs/glm52/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=/home/server
```

日志文件：

```bash
/data/logs/glm52-vllm.log
```

> 注意：当前命令是 Recipe/MTP 配方，不是最初 baseline。最初 baseline 与当前命令主要区别是没有 `--speculative-config.*`、parser 和 auto tool choice 相关参数。

## 4. 测试工具和数据目录

### 4.1 vLLM bench 命令模板

```bash
/data/venvs/glm52/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model glm-5.2-fp8 \
  --tokenizer /data/models/GLM-5.2-FP8-ms \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len <INPUT_TOKENS> \
  --random-output-len <OUTPUT_TOKENS> \
  --request-rate <REQUEST_RATE> \
  --num-prompts <NUM_PROMPTS> \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result \
  --result-filename <RESULT_JSON>
```

### 4.2 测试结果目录

| 测试组 | 目录 |
|---|---|
| Decode baseline | `/data/bench/glm52/20260630-025746-decode-baseline` |
| Prefill baseline | `/data/bench/glm52/20260630-030252-prefill-baseline` |
| Mixed baseline | `/data/bench/glm52/20260630-031603-mixed-baseline` |
| Recipe/MTP A/B | `/data/bench/glm52/20260630-032529-recipe-ab` |

GPU 监控数据保存在各目录下的 `gpu_monitor.csv`。

## 5. 指标含义

| 指标 | 含义 |
|---|---|
| TTFT | Time To First Token，请求发出到首 token 返回的时间；主要反映排队、Prefill、调度开销 |
| TPOT | Time Per Output Token，平均每个输出 token 的耗时；常用于衡量 decode 稳态速度 |
| ITL | Inter Token Latency，相邻 token 之间的间隔；更贴近流式输出体感 |
| E2EL | End-to-End Latency，请求整体完成时间 |
| Output throughput | 输出 token 吞吐，tok/s；衡量服务整体 decode 产能 |
| Request throughput | 请求吞吐，req/s；受请求长度和输出长度影响很大 |
| P95/P99 | 百分位延迟，P95 表示 95% 请求不超过该值，P99 表示 99% 请求不超过该值 |

## 6. Decode baseline 测试结果

### 6.1 512 input / 512 output

| Request rate | 请求数 | 成功 | 实际 req/s | 输出 tok/s | TTFT p50 ms | TTFT p95 ms | TTFT p99 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 20 | 0.59 | 301.12 | 277.87 | 1763.77 | 1823.92 | 35.38 | 30.57 | 18358.10 |
| 2 | 30 | 30 | 0.73 | 375.69 | 115.37 | 7448.49 | 7667.75 | 57.27 | 43.57 | 29398.12 |
| 4 | 40 | 40 | 1.19 | 611.68 | 120.70 | 648.97 | 682.70 | 50.00 | 51.18 | 25674.74 |

观察：

1. 输出吞吐从 301 tok/s 提升到 612 tok/s，说明多请求 continuous batching 对整体吞吐有明显收益。
2. r2 的 TTFT p95 明显异常高，说明小样本压测中存在排队或调度抖动；不能只看单次 P95，需要多轮或更大样本确认。
3. r4 的整体吞吐最高，但单请求完成时间仍在 25 秒级，说明 512 输出长度下 E2E 主要由 decode 持续时间决定。

### 6.2 2K input / 512 output

| Request rate | 请求数 | 成功 | 实际 req/s | 输出 tok/s | TTFT p50 ms | TTFT p95 ms | TTFT p99 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 20 | 0.52 | 267.16 | 572.04 | 5027.13 | 5040.71 | 51.81 | 37.45 | 27014.85 |
| 2 | 30 | 30 | 0.84 | 432.09 | 118.41 | 1450.54 | 1608.37 | 47.83 | 44.24 | 24564.94 |
| 4 | 40 | 40 | 1.11 | 568.06 | 134.27 | 2915.17 | 3112.74 | 55.31 | 51.72 | 28394.45 |

观察：

1. 输入从 512 增加到 2K 后，输出吞吐下降不大，说明这些场景仍主要受 decode 阶段影响。
2. TTFT 对输入长度更敏感，2K 输入下 p95 会更容易上升。

## 7. 长上下文 Prefill baseline 测试结果

该组使用自定义脚本生成较长输入，请求输出较短，用来观察 TTFT 与上下文长度的关系。

| 上下文档位 | 并发 | 请求数 | 平均 TTFT s | P95 TTFT s | 平均 E2E s | 实际 QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 1 | 2 | 2.33 | 2.35 | 2.58 | 0.387 |
| 8K | 2 | 4 | 3.45 | 4.59 | 4.84 | 0.412 |
| 8K | 4 | 8 | 5.74 | 9.15 | 9.38 | 0.425 |
| 16K | 1 | 2 | 4.66 | 4.66 | 4.90 | 0.204 |
| 16K | 2 | 4 | 6.95 | 9.27 | 9.48 | 0.211 |
| 16K | 4 | 8 | 10.99 | 18.48 | 18.64 | 0.214 |
| 32K | 1 | 2 | 9.49 | 9.50 | 9.72 | 0.103 |
| 32K | 2 | 4 | 14.18 | 18.90 | 19.07 | 0.105 |
| 32K | 4 | 8 | 21.85 | 37.71 | 37.75 | 0.106 |
| 64K | 1 | 2 | 19.74 | 19.75 | 19.94 | 0.050 |
| 64K | 2 | 4 | 29.54 | 39.35 | 39.44 | 0.051 |
| 64K | 4 | 8 | 45.51 | 78.57 | 75.29 | 0.051 |
| 128K | 1 | 1 | 42.88 | 42.88 | 43.08 | 0.023 |
| 128K | 2 | 2 | 64.35 | 85.63 | 84.11 | 0.023 |
| 256K | 1 | 1 | 100.62 | 100.62 | 100.72 | 0.010 |

观察：

1. TTFT 与上下文长度基本近似线性：32K 约 9.5s，64K 约 19.7s，128K 约 42.9s，256K 约 100.6s。
2. 对长上下文而言，提高并发不会线性提升 QPS，反而会显著增加排队和尾延迟。
3. 256K 可用，但不适合作为普通在线请求路径，应作为长文档/离线/低优先级队列处理。

## 8. 综合业务混合流量测试

流量构成：

| 类型 | 占比 | 近似输入 | 近似输出 | 业务含义 |
|---|---:|---:|---:|---|
| short | 60% | 1K | 192 | 普通短问答 |
| rag | 25% | 8K | 384 | RAG/知识库问答 |
| agent | 10% | 16K | 768 | Agent 多轮上下文 |
| long | 5% | 64K | 512 | 长文档分析 |

| 注入速率 req/s | 请求数 | 成功 | 实际 QPS | 输出 tok/s | TTFT p50 s | TTFT p95 s | E2E p50 s | E2E p95 s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 12 | 12 | 0.205 | 49.94 | 0.37 | 4.67 | 3.95 | 13.35 |
| 0.5 | 20 | 20 | 0.312 | 83.09 | 2.33 | 19.97 | 29.48 | 46.59 |
| 1.0 | 30 | 30 | 0.344 | 89.26 | 22.81 | 41.78 | 65.10 | 79.92 |

观察：

1. 当混合流量从 0.2 req/s 提升到 1.0 req/s，实际 QPS 只到 0.344，说明系统进入排队状态。
2. 长上下文请求会拖累短请求，导致短请求 TTFT 和 E2E 明显恶化。
3. 正式部署不建议把 1K、8K、16K、64K 请求无差别进入同一个队列。

建议的生产分流：

| 服务池 | 输入长度 | 建议用途 |
|---|---:|---|
| low-latency pool | <= 4K/8K | 普通在线问答、低延迟请求 |
| agent/rag pool | 8K-32K/64K | RAG、Agent、多轮任务 |
| long-context pool | 64K-256K | 长文档、批处理、低优先级任务 |

## 9. Recipe/MTP 对比测试

当前 Recipe/MTP 参数：

```bash
--speculative-config.method mtp \
--speculative-config.num_speculative_tokens 5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice
```

### 9.1 512 input / 512 output：Baseline vs Recipe/MTP

| Request rate | 配置 | 输出 tok/s | 实际 req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms | MTP 接受率 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Baseline | 301.12 | 0.59 | 1763.77 | 35.38 | 30.57 | 18358.10 | N/A |
| 1 | Recipe/MTP | 326.98 | 0.64 | 4096.88 | 33.75 | 89.59 | 20616.89 | 34.25% |
| 2 | Baseline | 375.69 | 0.73 | 7448.49 | 57.27 | 43.57 | 29398.12 | N/A |
| 2 | Recipe/MTP | 511.39 | 1.00 | 591.09 | 37.01 | 106.03 | 19150.85 | 34.19% |
| 4 | Baseline | 611.68 | 1.19 | 648.97 | 50.00 | 51.18 | 25674.74 | N/A |
| 4 | Recipe/MTP | 655.89 | 1.28 | 1040.00 | 47.69 | 130.16 | 24609.65 | 34.63% |

吞吐提升：

| Request rate | Recipe/MTP 输出吞吐提升 |
|---:|---:|
| 1 | +8.6% |
| 2 | +36.1% |
| 4 | +7.2% |

观察：

1. MTP 的 speculative acceptance rate 约 34%-35%，平均 acceptance length 约 2.7。
2. Recipe/MTP 在该测试中提升了输出吞吐，尤其 r2 提升明显。
3. ITL p95 明显变差，说明 speculative decoding 改善整体产能时，流式 token 间隔不一定更稳定。
4. r2 baseline 的 TTFT p95 明显异常高，因此 r2 的对比收益需要复测确认。

### 9.2 8K input / 1024 output：Recipe/MTP 探索测试

| Request rate | 请求数 | 成功 | 输出 tok/s | 实际 req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms | MTP 接受率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 222.76 | 0.22 | 12547.20 | 39.36 | 60.19 | 43779.38 | 36.13% |
| 2 | 16 | 16 | 335.63 | 0.33 | 10220.13 | 40.39 | 82.37 | 42541.94 | 36.08% |

说明：

1. 本轮没有在完全相同条件下补跑 8K/1024 的非 MTP baseline，因此该组只能说明 Recipe/MTP 自身表现，不能作为严格 A/B。
2. 从结果看，8K/1024 场景下实际 req/s 仍偏低，E2E 在 40 秒量级，主要因为输出长度 1024 且请求排队/批处理时间较长。

## 10. 是否“慢”的判断

基于本轮数据，不能简单用单个 curl 的 4.4 秒判断模型整体慢或快。原因：

1. 单请求短输出主要看 TTFT 和少量 decode，不能反映 continuous batching 下的整体吞吐。
2. 大模型服务的产能指标应主要看 tok/s、TPOT、ITL、E2E、TTFT p95/p99，以及在目标业务混合下的 QPS。
3. 对 GLM-5.2 这种大模型，单请求低并发时 GPU 资源利用不充分；提高并发通常能提升整体 tok/s，但会牺牲单请求延迟和流式平滑度。

## 11. 正式部署建议

### 11.1 服务方式

建议继续使用 systemd 管理 vLLM：

1. `glm52-vllm.service` 作为正式进程管理入口。
2. 标准输出和错误输出写入 `/data/logs/glm52-vllm.log`。
3. 通过 Gateway 暴露服务，不建议业务直接调用 vLLM。
4. Gateway 层负责鉴权、限流、token 估算、请求分流、熔断和计费。

### 11.2 推荐服务池

如果只有 h20-02 单机先上线，建议先用一个保守配置作为在线服务：

```bash
--tensor-parallel-size 8 \
--dtype auto \
--kv-cache-dtype fp8 \
--max-model-len 32768 或 65536 \
--gpu-memory-utilization 0.90
```

如果希望支持长上下文，建议拆成不同服务池：

| 服务池 | max-model-len | 目标 |
|---|---:|---|
| online-short | 32768 | 普通问答，低延迟 |
| agent-rag | 65536 或 131072 | RAG/Agent |
| long-context | 262144 | 长文档，低 QPS，异步/队列化 |

### 11.3 Recipe/MTP 是否保留

建议：

1. 如果目标是最大化整体输出 tok/s，可以继续评估 MTP。
2. 如果目标是流式体验稳定，必须重点看 ITL p95/p99，不能只看 output throughput。
3. 当前数据下，MTP 对 512/512 有吞吐收益，但 ITL 变差；是否保留应由真实业务混合流量再确认。

## 12. 关键风险和限制

1. 本次测试样本较小，P95/P99 只能作为方向性参考，不应作为 SLA。
2. Mixed 测试使用模拟流量，不等价于真实用户 prompt、真实采样参数和真实工具调用链。
3. Recipe/MTP 只做了 512/512 严格 A/B；8K/1024 没有非 MTP baseline 对照。
4. 当前 max-model-len 为 262144，会占用更多 KV cache 预算；如果生产只需要 32K，应重新以 32K 配置测试。
5. GPU monitor 数据已保留，但本报告没有展开逐秒 GPU 利用率、显存、水位和功耗分析。

## 13. 下一轮建议测试

为了形成可用于上线评审的报告，建议下一轮测试按以下方式补强：

1. 每个关键测试点至少 3 轮，报告 mean / p50 / p95 / p99。
2. Decode 场景增加 128、512、1024、2048 输出长度，观察输出长度对 TPOT/ITL 的影响。
3. 使用真实业务 prompt 样本，分 short、rag、agent、long 四类。
4. 对 MTP 做完整 A/B，包括 512/512、2K/512、8K/1024。
5. 对 32K 与 262K 两种 `max-model-len` 分别测试，确认长上下文配置是否影响短请求性能。
6. 增加 GPU 利用率、KV cache 使用率、队列长度、scheduler 状态等服务端监控。
7. 如果 Gateway 上线，要在 Gateway 层同步采集 request id、输入 token、输出 token、排队时间、首 token 时间和完成时间。

## 14. 文件索引

| 文件 | 用途 |
|---|---|
| `/root/demo/docs/glm52-h20-installation-notes.md` | 安装部署手记和知识库 |
| `/root/demo/docs/glm52-vllm-small-benchmark-2026-06-29.md` | 早期小规模测试记录 |
| `/root/demo/docs/glm52-vllm-long-context-benchmark-2026-06-29.md` | 早期长上下文测试记录 |
| `/root/demo/docs/glm52-performance-test-plan.md` | 本轮测试大纲与执行记录 |
| `/root/demo/docs/glm52-performance-final-report-2026-06-30.md` | 本报告 |
