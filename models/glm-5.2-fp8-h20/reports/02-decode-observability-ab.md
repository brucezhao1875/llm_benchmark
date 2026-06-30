# GLM-5.2-FP8 Decode / Observability / MTP A-B 测试报告

日期：2026-06-30  
机器：h20-02  
模型：GLM-5.2-FP8  
测试目标：按新版大纲复测 Decode 输出长度矩阵、服务端观测、Recipe/MTP 与非 MTP baseline 对比。

## 1. 本轮结论

1. 大模型服务不应孤立看 QPS，本轮主要看 `output tokens/s`、`TPOT`、`ITL`、`TTFT` 和 `E2E`。
2. 当前 8 卡 H20 跑 GLM-5.2-FP8，MTP 配置下 1K 输入场景的稳定输出吞吐大致在 `600-900 output tok/s` 区间，具体取决于输出长度和 request rate。
3. request rate 提升可以提高整体 output tok/s，但会增加 E2E 和 ITL；短输出场景在 r8 下 TTFT p95 已明显变差。
4. MTP 在本轮 4 个 A/B 代表点上均提升 output tok/s，提升幅度约 `11%-70%`，但 MTP 的 `ITL p95` 明显高于 baseline，流式体感可能更抖。
5. GPU 观测显示压测期间 GPU SM 利用率中位数达到 `100%`，说明测试期间 GPU 已被充分调动，不是明显的低利用率问题。
6. vLLM metrics 快照中 `waiting requests` 基本为 0，说明本轮矩阵没有形成长期队列堆积；延迟主要来自模型执行、batching、decode 长度和调度策略。
7. 当前测试结束后，服务已切到非 MTP baseline，仍为 `max-model-len=262144`。

## 2. 测试目录

| 阶段 | 配置 | 远端目录 |
|---|---|---|
| 阶段一 | Recipe/MTP | `/data/bench/glm52/20260630-042139-decode-mtp-full` |
| 阶段二 | non-MTP baseline | `/data/bench/glm52/20260630-043738-baseline-ab` |

每个目录包含：

1. vLLM bench JSON 结果。
2. 单点日志文件。
3. `gpu_dmon.csv`。
4. vLLM `/metrics` 快照。
5. systemd service unit 记录。

## 3. 当前最终服务状态

当前服务状态：`active`。

当前最终配置是非 MTP baseline：

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

systemd 环境变量：

```ini
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
Environment=PATH=/data/venvs/glm52/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=/home/server
```

## 4. 测试方法

vLLM bench 命令模板：

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

服务端观测：

```bash
nvidia-smi dmon -s pucmt -d 1
curl -s http://127.0.0.1:8000/metrics
```

## 5. 阶段一：Recipe/MTP Decode 输出长度矩阵

配置：Recipe/MTP，`max-model-len=262144`。

### 5.1 1K input / 128 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms | MTP accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0.89 | 113.64 | 1033.43 | 4 | 171 | 13.67 | 31.78 | 1882 | 30.01% |
| 2 | 24 | 24 | 1.71 | 218.99 | 1991.46 | 9 | 427 | 27.41 | 79.53 | 3831 | 29.68% |
| 4 | 32 | 32 | 2.68 | 342.61 | 3115.63 | 21 | 798 | 40.69 | 92.81 | 5390 | 31.07% |
| 8 | 48 | 48 | 2.86 | 365.44 | 3323.25 | 45 | 5245 | 97.31 | 172.77 | 12693 | 30.15% |

结论：128 输出下 request rate 从 1 到 4 提升明显，r8 开始收益变小且 TTFT/ITL 明显恶化。该场景的可用稳定点更接近 r4，而不是 r8。

### 5.2 1K input / 512 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms | MTP accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0.63 | 321.59 | 972.31 | 9 | 200 | 21.66 | 57.74 | 11247 | 34.36% |
| 2 | 24 | 24 | 0.94 | 480.60 | 1453.06 | 23 | 294 | 32.89 | 90.54 | 16999 | 33.67% |
| 4 | 32 | 32 | 1.21 | 617.01 | 1865.49 | 32 | 320 | 40.63 | 112.24 | 21044 | 33.83% |
| 8 | 48 | 48 | 1.53 | 785.79 | 2375.79 | 48 | 508 | 53.10 | 143.08 | 27396 | 33.79% |

结论：512 输出下 output tok/s 随 request rate 持续提升，r8 达到 786 tok/s，但 ITL p95 上升到 143ms。若追求吞吐可用 r8，若追求流式体感应控制更低并发。

### 5.3 1K input / 1024 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms | MTP accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0.40 | 410.70 | 826.21 | 16 | 250 | 25.38 | 80.64 | 26155 | 37.15% |
| 2 | 24 | 24 | 0.59 | 601.96 | 1210.97 | 24 | 294 | 31.74 | 89.70 | 32681 | 39.00% |
| 4 | 32 | 32 | 0.71 | 722.13 | 1452.72 | 32 | 352 | 39.95 | 109.42 | 41079 | 37.06% |
| 8 | 48 | 48 | 0.87 | 889.80 | 1790.03 | 48 | 486 | 49.44 | 139.20 | 50836 | 38.16% |

结论：1024 输出下吞吐随 request rate 提升明显，r8 达到本轮最高 `889.8 output tok/s`。但 E2E p95 已超过 50s，适合高吞吐长回复，不适合低延迟交互。

### 5.4 1K input / 2048 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms | MTP accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 12 | 0.23 | 477.16 | 718.53 | 12 | 216 | 20.05 | 62.86 | 41223 | 43.94% |
| 2 | 16 | 16 | 0.25 | 517.41 | 779.15 | 16 | 288 | 29.26 | 82.66 | 60054 | 39.23% |
| 4 | 24 | 24 | 0.36 | 729.26 | 1098.17 | 24 | 308 | 30.20 | 90.94 | 62080 | 42.43% |

结论：2048 长输出下 QPS 很低是正常现象，因为每个请求消耗大量 decode token。此时更应该看 output tok/s，而不是孤立 QPS。

### 5.5 8K input / 512 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms | MTP accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 12 | 0.32 | 164.48 | 2800.05 | 12 | 13805 | 61.88 | 61.95 | 34479 | 34.65% |
| 2 | 16 | 16 | 0.56 | 286.34 | 4874.41 | 16 | 6154 | 42.98 | 82.46 | 22180 | 33.45% |
| 4 | 24 | 24 | 0.64 | 330.15 | 5620.20 | 24 | 13792 | 65.43 | 92.56 | 33725 | 34.10% |

结论：8K 输入会显著影响 TTFT，p95 到秒级甚至十秒级。RAG 请求应单独作为中长上下文池管理，不应与短请求混在同一个低延迟池里。

### 5.6 8K input / 1024 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms | MTP accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 12 | 0.37 | 383.40 | 3455.05 | 12 | 228 | 24.12 | 63.46 | 24843 | 35.64% |
| 2 | 16 | 16 | 0.43 | 436.34 | 3932.15 | 16 | 295 | 30.18 | 82.60 | 31109 | 33.41% |
| 4 | 24 | 24 | 0.62 | 635.03 | 5722.72 | 24 | 353 | 33.68 | 91.93 | 34788 | 38.27% |

结论：8K/1024 在 MTP 下表现好于 8K/512 的部分点，说明结果受 batching、请求分布和调度状态影响，需要用真实业务流量再确认。

## 6. 阶段二：Baseline A/B 代表点

配置：非 MTP baseline，`max-model-len=262144`。

| 场景 | Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K/512 | 4 | 32 | 32 | 1.01 | 516.84 | 1562.62 | 2562 | 58.11 | 45.06 | 30653 |
| 1K/1024 | 4 | 32 | 32 | 0.63 | 644.32 | 1296.19 | 145 | 42.90 | 45.74 | 44014 |
| 8K/1024 | 2 | 16 | 16 | 0.25 | 256.34 | 2310.10 | 24256 | 57.99 | 31.95 | 62788 |
| 1K/2048 | 2 | 16 | 16 | 0.23 | 466.10 | 701.89 | 122 | 30.76 | 31.70 | 63060 |

## 7. MTP vs Baseline 对比

| 场景 | Rate | Baseline output tok/s | MTP output tok/s | 提升 | Baseline TTFT p95 | MTP TTFT p95 | Baseline ITL p95 | MTP ITL p95 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1K/512 | 4 | 516.84 | 617.01 | +19.4% | 2562ms | 320ms | 45ms | 112ms | 吞吐和 E2E 改善，流式间隔变差 |
| 1K/1024 | 4 | 644.32 | 722.13 | +12.1% | 145ms | 352ms | 46ms | 109ms | 吞吐改善，TTFT/ITL 变差但 E2E 略好 |
| 8K/1024 | 2 | 256.34 | 436.34 | +70.2% | 24256ms | 295ms | 32ms | 83ms | MTP 显著更好，但 baseline TTFT 异常高，建议复测 |
| 1K/2048 | 2 | 466.10 | 517.41 | +11.0% | 122ms | 288ms | 32ms | 83ms | 吞吐和 E2E 改善，流式体感变差 |

判断：

1. 如果目标是整体吞吐和成本效率，MTP 值得继续评估。
2. 如果目标是稳定流式输出，MTP 需要谨慎，因为 ITL p95 明显变高。
3. 8K/1024 的 A/B 差异过大，baseline 的 TTFT p95 异常高，不能只凭单轮下结论，应复测。
4. 当前更合理的策略是把 MTP 作为高吞吐池配置，而不是直接作为所有业务的默认配置。

## 8. 服务端观测

### 8.1 GPU dmon 汇总

| 阶段 | GPU SM util avg | GPU SM util p50 | GPU SM util max | GPU mem util avg | power avg | power max |
|---|---:|---:|---:|---:|---:|---:|
| MTP Decode 矩阵 | 80.1% | 100.0% | 100.0% | 21.6% | 326.6W | 432W |
| Baseline A/B | 86.1% | 100.0% | 100.0% | 25.4% | 340.7W | 418W |

观察：

1. GPU SM 利用率 p50 为 100%，说明压测期间 GPU 经常被打满。
2. 平均 util 低于 100% 是因为不同测试点之间有间隔、启动 bench 有空档、短输出阶段可能存在调度空隙。
3. GPU memory util 不是显存占用比例，而是 dmon 中的 memory activity/util 口径；不能直接等价为显存 GB 使用率。

### 8.2 vLLM metrics 观察

阶段一 metrics 快照数：182。  
阶段二 metrics 快照数：50。

观察到的典型指标：

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:num_requests_waiting_by_reason{reason="capacity"}
vllm:num_requests_waiting_by_reason{reason="deferred"}
vllm:time_to_first_token_seconds
vllm:request_time_per_output_token_seconds
vllm:request_prefill_time_seconds
vllm:request_decode_time_seconds
```

关键现象：

1. `num_requests_running` 在高压测试中可到 32/48 左右。
2. `num_requests_waiting` 多数快照为 0。
3. 这说明本轮测试更多是在 running batch 内消耗时间，而不是大量请求长期排队。
4. 如果后续真实流量出现 waiting requests 持续 > 0，说明进入容量或调度拥塞，应触发限流或分流。

## 9. QPS 解释

本轮测试再次验证：QPS 是业务画像下的派生指标。

以 MTP 的 1K input 为例：

| 输出长度 | 最优 observed output tok/s | 对应 observed req/s | 说明 |
|---:|---:|---:|---|
| 128 | 365.44 | 2.86 | 短输出 QPS 高，但 r8 延迟恶化 |
| 512 | 785.79 | 1.53 | 普通问答吞吐较好 |
| 1024 | 889.80 | 0.87 | tok/s 高，但 E2E 长 |
| 2048 | 729.26 | 0.36 | QPS 低是正常现象 |

因此不能说“模型 QPS 只有 0.x 或 1.x”。更准确的说法是：

```text
在 1K/512 场景，MTP 下最高观察到约 786 output tok/s，对应约 1.53 req/s。
在 1K/1024 场景，MTP 下最高观察到约 890 output tok/s，对应约 0.87 req/s。
在 1K/2048 场景，MTP 下最高观察到约 729 output tok/s，对应约 0.36 req/s。
```

## 10. 生产配置建议

### 10.1 是否保留 MTP

建议：不要立即把 MTP 作为所有业务默认配置。

更合理的选择：

| 服务池 | 配置建议 | 原因 |
|---|---|---|
| 低延迟流式池 | baseline 或低并发 MTP | 重点控制 ITL 和 TTFT |
| 高吞吐生成池 | MTP | output tok/s 更高，适合长输出或非强流式任务 |
| RAG/Agent 池 | MTP 候选，但需复测 | 8K/1024 本轮收益大，但 baseline 数据需复核 |
| long-context 池 | 单独限流 | 避免拖慢短请求 |

### 10.2 Gateway 策略

Gateway 应至少按以下维度分流：

1. 输入 token 长度。
2. 预期输出 token 长度。
3. 是否 streaming。
4. 是否 Agent/RAG/long-context。
5. 是否高优先级在线请求。

建议初始规则：

| 请求类型 | 条件 | 路由 |
|---|---|---|
| short-chat | input <= 4K，output <= 512 | 低延迟池 |
| normal-chat | input <= 8K，output <= 1024 | 普通池 |
| long-generation | output >= 1024 | 高吞吐池 |
| rag/agent | input 8K-32K | RAG/Agent 池 |
| long-context | input >= 64K | 长上下文队列 |

## 11. 后续建议

1. 对 8K/1024 的 baseline vs MTP 再复测 2-3 轮，因为本轮 baseline TTFT p95 异常高。
2. 将真实业务 prompt 接入测试，而不是只用 random dataset。
3. 如果要做 SLA，单点样本数应提升到 100+，并至少跑 3 轮。
4. 将 vLLM `/metrics` 接入 Prometheus/Grafana，持续观测 waiting requests、running requests、TTFT、TPOT、KV cache usage。
5. 如果要精确核算商业收入，应使用真实业务的平均 input/output tokens，并以 observed output tok/s 而不是孤立 QPS 计算。
