# GLM-5.2-FP8 32K / 256K / 1M 启动上下文对比测试报告

日期：2026-06-30  
机器：h20-02  
模型：GLM-5.2-FP8  
测试目标：验证不同 `max-model-len` 启动配置对启动时间、短请求性能、边界上下文能力、1M 可用性和 Gateway 分池策略的影响。

## 1. 本轮结论

1. `32K` 启动配置成功，`/v1/models` ready 用时 `220s`。
2. `32K` 配置完成了短请求矩阵和 32K 边界探针，所有测试点成功。
3. `1M` 启动配置在 `30 分钟 ready 等待窗口` 内没有 ready，记录为 `1m_start_failed`；后续复测官方缺省 H20/H200 配方也失败。
4. 官方缺省配方不写 `--max-model-len` 时，vLLM 会从模型配置解析出 `1048576`，并在 H20 上因 KV cache 不足失败。
5. `32K` 对普通短请求表现稳定，适合作为低延迟在线池候选。
6. 当前服务已恢复到测试前配置：非 MTP baseline，`max-model-len=262144`。
7. Gateway 必须支持 token-aware routing：普通请求走短上下文池，长上下文走 256K/1M 专用池，1M 请求默认不应进入普通在线同步池。

## 2. 测试目录

远端总目录：

```text
/data/bench/glm52/20260630-061059-context-window-32k-vs-1m
```

子目录：

| 阶段 | 目录 | 结果 |
|---|---|---|
| 32K 启动配置 | `/data/bench/glm52/20260630-061059-context-window-32k-vs-1m/stage32k` | 成功 |
| 1M 启动配置 | `/data/bench/glm52/20260630-061059-context-window-32k-vs-1m/stage1m` | 30 分钟内未 ready |

历史 256K 对照数据来自：

| 配置 | 目录 |
|---|---|
| 256K baseline | `/data/bench/glm52/20260630-043738-baseline-ab` |
| 256K MTP | `/data/bench/glm52/20260630-042139-decode-mtp-full` |

## 3. 启动配置对比

| 启动配置 | max-model-len | ready 结果 | ready 时间 | 说明 |
|---|---:|---|---:|---|
| 32K | 32768 | 成功 | 220s | 完成全部测试 |
| 256K | 262144 | 历史已成功 | 历史数据 | 当前已恢复到该配置 |
| 1M | 1048576 | 未 ready | >1800s | 30 分钟窗口内 API 未 ready，停止后续测试 |
| 官方缺省 H20/H200 配方 | 1048576 | 失败 | 约 170s 后退出 | 未显式设置 `--max-model-len`，vLLM 从 config 读取 1M |

1M 阶段日志要点：

```text
06:25:49 switch_service max_model_len=1048576
06:55:58 service_not_ready max_model_len=1048576
06:55:58 restoring_original_service
06:57:59 context_window_comparison_end
```

诊断过程中看到 1M vLLM 进程曾进入 active/loading/compile/warmup 状态，但 `/v1/models` 在设定 ready 窗口内没有成功返回。因此本报告将其定义为：

```text
1M 配置在当前启动参数下不可作为可用服务配置。
```

不是说模型理论上不能 1M，而是当前这套 vLLM 参数和 30 分钟 ready 策略下没有形成可用 API 服务。

## 4. 当前最终服务状态

测试结束后已恢复原始 service。

当前启动命令：

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

## 5. 32K 启动配置：短请求矩阵

配置：

```bash
--max-model-len 32768
--tensor-parallel-size 8
--dtype auto
--kv-cache-dtype fp8
--gpu-memory-utilization 0.90
```

### 5.1 1K input / 128 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0.83 | 105.88 | 962.82 | 7 | 913 | 33.64 | 24.13 | 4670 |
| 2 | 24 | 24 | 1.58 | 201.63 | 1833.59 | 11 | 428 | 46.83 | 30.12 | 6097 |
| 4 | 32 | 32 | 2.47 | 316.26 | 2875.96 | 22 | 1317 | 49.37 | 37.39 | 6406 |
| 8 | 48 | 48 | 3.22 | 412.01 | 3746.71 | 44 | 3476 | 80.84 | 51.08 | 10398 |

观察：短输出场景 request rate 提升会提高 output tok/s，但 r8 的 TTFT p95 已上升到 3.5s 左右，低延迟池不应无限提高并发。

### 5.2 1K input / 512 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0.55 | 283.34 | 856.66 | 14 | 120 | 28.36 | 30.46 | 14598 |
| 2 | 24 | 24 | 0.87 | 445.07 | 1345.63 | 24 | 136 | 34.47 | 37.55 | 17726 |
| 4 | 32 | 32 | 1.15 | 590.28 | 1784.68 | 32 | 162 | 40.70 | 44.58 | 20940 |
| 8 | 48 | 48 | 1.53 | 785.86 | 2376.00 | 48 | 171 | 50.81 | 54.64 | 26100 |

观察：这是 32K 配置下最稳的一组。r8 达到 `785.86 output tok/s`，TTFT p95 仍只有 `171ms`，但 E2E p95 约 `26.1s`，主要由 512 token decode 时间决定。

### 5.3 1K input / 1024 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0.36 | 368.93 | 742.18 | 16 | 121 | 29.33 | 30.96 | 30096 |
| 2 | 24 | 24 | 0.52 | 528.41 | 1063.00 | 24 | 131 | 35.66 | 37.77 | 36602 |
| 4 | 32 | 32 | 0.63 | 645.89 | 1299.36 | 32 | 157 | 42.71 | 45.35 | 43827 |

观察：1024 输出下 QPS 自然下降，但 output tok/s 仍能到 `645.89 tok/s`。这是典型的“token throughput 稳定，QPS 随输出长度下降”。

### 5.4 8K input / 512 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 12 | 0.30 | 153.65 | 2615.68 | 12 | 13141 | 69.78 | 30.82 | 38497 |
| 2 | 16 | 16 | 0.54 | 276.67 | 4709.90 | 16 | 5866 | 44.97 | 31.58 | 23115 |
| 4 | 24 | 24 | 0.64 | 326.36 | 5555.77 | 24 | 13365 | 66.14 | 37.73 | 34004 |

观察：8K 输入下 TTFT p95 波动较大，说明 RAG 类请求不能只看 output tok/s，应同时看 prefill/TTFT 尾延迟。

### 5.5 8K input / 1024 output

| Rate | 请求数 | 成功 | req/s | output tok/s | total tok/s | max conc | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 12 | 0.30 | 305.91 | 2756.81 | 12 | 189 | 28.83 | 31.08 | 29624 |
| 2 | 16 | 16 | 0.42 | 426.85 | 3846.70 | 16 | 196 | 30.34 | 31.69 | 31185 |

观察：8K/1024 在本轮表现比 8K/512 更平滑，说明小样本下受调度状态影响较大。后续若要做 SLA，应扩大样本量并多轮复测。

## 6. 32K 边界探针

| 输入长度 | 输出长度 | 请求数 | 成功 | TTFT ms | total tok/s | E2E ms | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16K | 64 | 1 | 1 | 2473 | 3544.67 | 3533 | 成功 |
| 24K | 64 | 1 | 1 | 2586 | 5180.71 | 3645 | 成功 |
| 32K | 1 | 1 | 1 | 2602 | 8882.30 | 2602 | 成功 |

结论：32K 启动配置可以完成 32K 边界输入探针。由于样本数为 1，这只证明能力可用，不代表稳定 SLA。

## 7. 服务端观测：32K 阶段

### 7.1 GPU dmon

| 指标 | avg | p50 | max | samples |
|---|---:|---:|---:|---:|
| GPU SM util | 75.6% | 100.0% | 100.0% | 4000 |
| GPU memory activity | 21.8% | 29.0% | 38.0% | 4000 |
| power draw | 305.5W | 353.5W | 422.0W | 4000 |

说明：GPU SM p50 为 100%，说明压测期间 GPU 经常被打满。avg 低于 100% 是因为测试点间存在空档、启动 bench 存在间隔。

### 7.2 空载显存快照

32K ready 后空载显存约：

```text
每卡约 132385 MiB / 143771 MiB
```

注意：这不是 KV cache 已实际使用的 token 数，而是模型权重、runtime、预留/缓存等综合结果。

### 7.3 vLLM metrics 快照

最后快照中：

```text
vllm:num_requests_waiting = 0
vllm:num_requests_running = 8
vllm:kv_cache_usage_perc ≈ 0.0172
```

说明：本轮 32K 测试没有形成持续 waiting queue，KV cache 使用率在最后采样点很低。

## 8. 32K vs 256K 对照

256K 对照来自历史测试，不是同一轮连续 A/B，因此只作为方向性对比。

### 8.1 32K baseline vs 256K baseline

| 场景 | Rate | 32K output tok/s | 256K baseline output tok/s | 差异 | 32K TTFT p95 | 256K TTFT p95 | 32K ITL p95 | 256K ITL p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K/512 | 4 | 590.28 | 516.84 | +14.2% | 162ms | 2562ms | 44.58ms | 45.06ms |
| 1K/1024 | 4 | 645.89 | 644.32 | +0.2% | 157ms | 145ms | 45.35ms | 45.74ms |
| 8K/1024 | 2 | 426.85 | 256.34 | +66.5% | 196ms | 24256ms | 31.69ms | 31.95ms |

解释：

1. 1K/512 下，32K 比 256K baseline 更好，尤其 TTFT p95 明显更低。
2. 1K/1024 下，两者 output tok/s 接近。
3. 8K/1024 下，256K baseline 的 TTFT p95 异常高，之前报告也已标注该点需要复测。因此不能简单断言 32K 对 8K/1024 必然提升 66%。

### 8.2 32K baseline vs 256K MTP

| 场景 | Rate | 32K baseline output tok/s | 256K MTP output tok/s | 差异 | 32K ITL p95 | 256K MTP ITL p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1K/512 | 4 | 590.28 | 617.01 | -4.3% | 44.58ms | 112.24ms |
| 1K/512 | 8 | 785.86 | 785.79 | ~0.0% | 54.64ms | 143.08ms |
| 1K/1024 | 4 | 645.89 | 722.13 | -10.6% | 45.35ms | 109.42ms |
| 8K/1024 | 2 | 426.85 | 436.34 | -2.2% | 31.69ms | 82.60ms |

解释：

1. 256K MTP 在部分场景 output tok/s 更高。
2. 32K baseline 的 ITL p95 明显更低，流式体感更稳定。
3. 如果目标是低延迟流式体验，32K baseline 是更稳的在线池候选。
4. 如果目标是最大化吞吐，MTP 仍值得作为高吞吐池候选。

## 9. 1M 和官方缺省配方测试结论

本轮未获得 1M 请求性能数据，因为 1M 服务没有在 30 分钟内 ready。

结论边界：

```text
本轮只能证明：当前参数下，1M 作为 vLLM systemd 服务不能在 30 分钟窗口内稳定 ready。
不能证明：GLM-5.2-FP8 在硬件上绝对不能跑 1M。
```

可能原因：

1. `max-model-len=1048576` 导致 profile、KV cache 规划、warmup 或 CUDA graph 阶段显著变慢。
2. vLLM 在 1M 配置下可能需要额外参数，例如降低并发、调整 CUDA graph、限制 batch token 或禁用部分 capture。
3. 当前 30 分钟 ready 窗口对 1M 可能不够，但如果服务启动要超过 30 分钟，也不适合作为普通在线池。
4. systemd restart 和旧进程清理在超大配置切换时存在额外耗时。

建议下一次 1M 专项测试不要和完整矩阵混在一起，而是单独做：

```text
1. 单独启动 1M 服务。
2. ready 等待窗口放宽到 60-90 分钟。
3. 先只观察启动日志，不立即压测。
4. 如果 ready 成功，再跑 384K/1 output。
5. 如果 384K 成功，再逐级跑 512K、768K、1M。
```

### 9.1 官方缺省 H20/H200 配方复测

复测时间：2026-06-30  
远端结果目录：

```text
/data/bench/glm52/20260630-084852-default-recipe-startup-test-with-path
```

测试命令：

```bash
PATH=/data/venvs/glm52/bin:$PATH /data/venvs/glm52/bin/vllm serve /data/models/GLM-5.2-FP8-ms \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 5 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --served-model-name glm-5.2-fp8
```

说明：这里用本地模型路径替代 `zai-org/GLM-5.2-FP8`，避免重新下载；测试重点是官方配方没有显式传 `--max-model-len`。

测试结果：

| 项目 | 结果 |
|---|---|
| 是否 ready | 否 |
| 退出时间 | 约 170s |
| vLLM 解析出的上下文 | `1048576` |
| 1M 所需 KV cache | `60.79 GiB` / GPU |
| 当前可用 KV cache | `22.87 GiB` / GPU |
| vLLM 估算最大可承载长度 | `394496` tokens |
| 原 262K 服务恢复 | 成功，约 240s ready |

关键日志：

```text
Using max model len 1048576
Available KV cache memory: 22.87 GiB
ValueError: To serve at least one request with the model's max seq len (1048576),
60.79 GiB KV cache is needed, larger than available 22.87 GiB.
Based on the available memory, the estimated maximum model length is 394496.
```

结论：

```text
H20/H200 standard FP8 配方可以表达“用 FP8 运行 GLM-5.2-FP8”，
但在 H20 上不能省略 max-model-len。
否则 vLLM 会按模型原生 1M 上下文启动，最终因 KV cache 不足失败。
```

因此 H20 生产启动命令必须显式设置：

```bash
--max-model-len 262144
```

如果要探索 H20 的上限，应使用独立实验池从 `384K` 或 `393216` 这类长度开始，而不是直接使用缺省 1M。

### 9.2 每个 token 的 KV cache 显存占用

根据 vLLM 实测错误：

```text
60.79 GiB / 1,048,576 tokens = 60.79 KiB/token/GPU
```

也就是：

```text
约 62,249 bytes/token/GPU
```

这是每张 GPU 上的 KV cache 成本，不是 8 卡合计成本。8 卡总预留量还要乘以 8，但真正启动失败的约束是每张卡自己的可用 KV cache 是否足够。

从 GLM-5.2 的 MLA/DSA 结构做一个下界估算：

```text
num_hidden_layers = 78
kv_lora_rank = 512
qk_rope_head_dim = 64
kv_cache_dtype = fp8 = 1 byte/element

raw MLA KV payload
= 78 * (512 + 64) * 1
= 44,928 bytes/token/GPU
= 43.875 KiB/token/GPU
```

vLLM 实际规划值比 raw payload 大：

```text
60.79 / 43.875 = 约 1.39 倍
```

差异来自具体运行时实现，包括 `fp8_ds_mla` / DSA KV 格式、block/page 分配、alignment、scale/metadata、CUDA graph/profile 相关保留等。因此：

```text
raw 公式用于理解数量级；
生产容量规划必须以 vLLM 启动日志为准。
```

按 vLLM 这次实测成本估算：

| 上下文长度 | 每 GPU KV cache 需求 |
|---:|---:|
| 262,144 | `15.20 GiB` |
| 394,496 | `22.87 GiB` |
| 1,048,576 | `60.79 GiB` |

恢复后的 262K 服务日志显示：

```text
Available KV cache memory: 28.74 GiB
GPU KV cache size: 502,016 tokens
Maximum concurrency for 262,144 tokens per request: 1.92x
```

这说明 262K 的服务配置有足够 KV 空间，而 1M 在当前 H20 上没有。

## 10. 对 Gateway 的要求

本轮结果加强了一个结论：Gateway 不能只是 HTTP 转发层，必须是模型流量调度层。

### 10.1 推荐池化

| 池 | max-model-len | 目标 | 当前依据 |
|---|---:|---|---|
| low-latency | 32K | 普通聊天、低延迟、短 RAG | 32K 测试成功，ITL 稳定 |
| standard/agent | 256K | RAG、Agent、长上下文 | 历史 256K 已可用 |
| ultra-long | 1M | 偶发超长上下文 | 当前参数下未 ready，应作为专项池继续验证 |
| high-throughput | 256K + MTP | 长输出、高吞吐、非强流式 | MTP 吞吐更高但 ITL 更差 |

### 10.2 Gateway 必须新增的能力

| 能力 | 要求 |
|---|---|
| input token 估算 | 请求进入后端前估算输入长度 |
| max output token 读取 | 读取 `max_tokens`，缺省时设置默认预算 |
| context tier | 识别 short / standard / long / ultra-long |
| decode tier | 识别 short-output / long-output |
| pool routing | 按 token 和 SLA 路由到 32K/256K/1M/MTP 池 |
| admission control | 控制每个池的 in-flight token 和 waiting queue |
| async job | 1M 请求默认支持异步任务形式 |
| per-pool billing | 1M 和长输出应单独计价 |
| per-pool observability | 按池统计 TTFT、ITL、E2E、tok/s、失败率 |

### 10.3 初始路由建议

```text
input <= 8K 且 output <= 512      -> 32K low-latency pool
input <= 32K 且 output <= 1024    -> 32K 或 256K standard pool
input 32K-256K                   -> 256K long/agent pool
output >= 2048                   -> high-throughput / long-decode pool
input > 256K                     -> ultra-long pool，默认异步或低并发
input 接近 1M                    -> 1M 专项池，单独限流、单独计费
```

## 11. 最终建议

1. 生产默认不要直接使用 1M 启动配置。
2. 32K 可以作为低延迟在线池候选。
3. 256K 可以作为长上下文/Agent/RAG 池候选。
4. MTP 可以作为高吞吐池候选，但不适合作为所有流式请求默认配置。
5. 1M 应作为独立 ultra-long 实验池继续验证，且默认异步、低并发、单独计费。
6. Gateway 应尽快加入 token-aware routing，否则后端服务池无法被有效运营。

## 12. 文件索引

| 文件 | 用途 |
|---|---|
| `/root/demo/docs/glm52-context-window-comparison-test-plan.md` | 本轮测试大纲 |
| `/root/demo/docs/glm52-context-window-comparison-report-2026-06-30.md` | 本报告 |
| `/root/demo/docs/glm52-decode-observability-ab-report-2026-06-30.md` | 前一轮 Decode/MTP/观测报告 |
| `/root/demo/docs/glm52-performance-test-plan.md` | 综合测试大纲与执行记录 |
