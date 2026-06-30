# GLM-5.2-FP8 性能测试大纲与执行记录

> 版本：v1.0
> 创建时间：2026-06-30
> 测试对象：GLM-5.2-FP8 on h20-02
> 推理框架：vLLM OpenAI-compatible API
> 主测试工具：vLLM bench serve
> 辅助工具：自定义 Python 脚本、nvidia-smi 监控
> 执行约束：避免频繁 SSH 登录，尽量通过单次远端批处理完成测试与监控采集。

## 1. 测试目标

本轮测试目标不是单纯追求最高吞吐，而是形成可指导生产部署的性能画像。

需要回答的问题：

| 问题 | 目标 |
|---|---|
| 当前 baseline 配置性能如何 | 建立当前脚本的基线 |
| vLLM 官方 recipe 是否提升指标 | 做 A/B 对比 |
| 多请求 continuous batching 是否提升整体吞吐 | 找 aggregate TPS 拐点 |
| 不同上下文长度下 TTFT 如何变化 | 指导上下文分层和限流 |
| Decode 阶段真实生成能力如何 | 测 output TPS、TPOT、ITL |
| 真实业务混合流量下能承载多少 | 指导 Gateway 路由、限流和容量规划 |

## 2. 当前服务启动参数记录

当前服务由 systemd 管理：

```text
/etc/systemd/system/glm52-vllm.service
```

服务名：

```text
glm52-vllm.service
```

日志：

```text
/data/logs/glm52-vllm.log
```

当前已使用过的核心启动命令：

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

systemd 关键环境变量：

```ini
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
Environment=PATH=/data/venvs/glm52/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=/home/server
```

注意：

```text
PATH 必须包含 /data/venvs/glm52/bin，否则 vLLM/FlashInfer 初始化时可能找不到 ninja。
```

## 3. 测试工具选择

### 3.1 主工具：vLLM bench serve

用途：

```text
TTFT
TPOT
ITL
E2E latency
request throughput
output token throughput
percentile metrics
request-rate 压测
```

适合测试：

```text
Decode 吞吐专项
并发收益曲线
Prefill 固定长度专项
Baseline / Recipe A/B 对比
```

### 3.2 辅助工具：自定义 Python 脚本

用途：

```text
真实业务混合流量
按业务类型分别统计指标
超长上下文特殊测试
自定义 request-rate 和熔断策略
```

### 3.3 GPU 监控工具

使用：

```bash
nvidia-smi dmon
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

监控原则：

```text
监控进程在 h20-02 本地后台运行并写日志。
不通过频繁 SSH 轮询 GPU 状态。
每个测试阶段生成独立监控日志。
```

## 4. 测试分组

## A. Decode 吞吐专项

目标：验证多个请求通过 continuous batching / batched decoding 是否提高整体服务吞吐。

测试矩阵：

| Input tokens | Output tokens | 并发 | 请求数 |
|---:|---:|---:|---:|
| 512 | 512 | 1 / 2 / 4 / 8 / 16 / 32 | 每组 50 |
| 2K | 512 | 1 / 2 / 4 / 8 / 16 / 32 | 每组 50 |
| 8K | 1024 | 1 / 2 / 4 / 8 / 16 | 每组 30 |
| 32K | 512 | 1 / 2 / 4 / 8 | 每组 10，可选 |

重点指标：

```text
aggregate output TPS
single-request decode TPS
TPOT p50/p90/p95/p99
ITL p50/p90/p95/p99
TTFT p50/p95/p99
E2E latency p50/p95/p99
error rate
timeout rate
GPU util
```

判断逻辑：

```text
并发增加后 aggregate TPS 上升，说明 batching 有收益。
如果 aggregate TPS 不再增长但 p95 latency 快速恶化，说明达到吞吐拐点。
生产推荐点不是最大并发，而是满足 SLA 的最大 goodput。
```

## B. Prefill 长上下文专项

目标：测输入上下文长度对 TTFT 的影响。

测试矩阵：

| Input tokens | Output tokens | 并发 | 轮数 |
|---:|---:|---:|---:|
| 8K | 16 | 1 / 2 / 4 | 3 |
| 16K | 16 | 1 / 2 / 4 | 3 |
| 32K | 16 | 1 / 2 / 4 | 3 |
| 64K | 16 | 1 / 2 / 4 | 3 |
| 128K | 16 | 1 / 2 | 2 |
| 256K | 16 | 1 | 1 |

不再测试：

```text
128K x 并发 8/16
256K x 并发 2/4/8/16
```

原因：

```text
前一轮测试已经证明这些组合主要表现为排队，耗时很长，信息增量低。
```

## C. 综合业务数据模拟测试

目标：模拟真实业务混合流量，判断单实例稳定承载能力。

业务画像 v1：

| 类型 | 占比 | Input tokens | Output tokens | 场景 |
|---|---:|---:|---:|---|
| Short Chat | 60% | 512-2K | 128-256 | 普通问答 |
| RAG | 25% | 4K-16K | 256-512 | 知识库问答 |
| Agent | 10% | 8K-32K | 512-1024 | 工具调用、复杂任务 |
| Long Context | 5% | 64K-128K | 512 | 长文档/仓库级任务 |

压测方式：

```text
按 request-rate 递增，而不是固定并发。
```

压力档位：

```text
0.2 req/s
0.5 req/s
1 req/s
2 req/s
4 req/s
```

每档持续：

```text
5-10 分钟
```

熔断条件：

```text
p95 TTFT > 60s
p95 E2E latency > 120s
error rate > 1%
timeout rate > 1%
队列持续增长
GPU/KV cache 明显耗尽
```

输出要求：

```text
整体指标
按业务类型拆分指标
长上下文请求对短请求的影响
推荐 Gateway 路由和限流策略
```

## D. Baseline vs vLLM Recipe A/B 测试

Baseline：当前启动脚本。

Recipe：在 baseline 基础上增加 vLLM GLM-5.2 recipe 推荐参数：

```bash
--speculative-config.method mtp \
--speculative-config.num_speculative_tokens 5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice
```

视测试目标增加：

```bash
--max-num-seqs
--no-enable-prefix-caching
```

A/B 代表性测试点：

| 场景 | Input | Output | 并发 | 目的 |
|---|---:|---:|---:|---|
| Decode | 512 | 512 | 1 / 4 / 8 / 16 | 看 MTP 对生成吞吐提升 |
| Agent | 8K | 1024 | 1 / 2 / 4 / 8 | 看中上下文 + 长输出 |
| Prefill | 32K | 16 | 1 / 2 / 4 | 看 recipe 是否影响 TTFT |
| Long | 128K | 16 | 1 / 2 | 看长上下文稳定性 |

判断标准：

```text
Decode TPS 提升 > 10%-20%：recipe 有明显价值。
TTFT 不明显恶化：可接受。
p95/p99 不明显恶化：可接受。
error rate 不上升：可接受。
启动稳定：可接受。
```

## 5. 报告输出结构

最终报告建议包含：

```text
1. 测试环境
2. 启动参数
3. 工具与方法
4. Baseline Decode 结果
5. Recipe Decode 结果
6. Prefill 曲线
7. 综合业务混合流量
8. A/B 对比结论
9. 推荐生产配置
10. Gateway 路由、限流、配额建议
11. 测试局限
12. 后续计划
```

## 6. 执行过程记录

### 2026-06-30

计划开始执行：

```text
第一阶段：Decode baseline 小矩阵
工具：vLLM bench serve
同步监控：nvidia-smi 本地后台采样
远端执行方式：单次 SSH 批处理，避免频繁登录
```

预期产物：

```text
/data/bench/glm52/<timestamp>/bench_results
/data/bench/glm52/<timestamp>/gpu_monitor.csv
/data/bench/glm52/<timestamp>/run.log
本地 Markdown 汇总报告
```

### 2026-06-30 执行记录：vLLM bench 首次运行问题

首次运行命令未显式指定本地 tokenizer，`vLLM bench serve` 尝试进行远端 repo file list 检查。

远端日志表现：

```text
Error retrieving file list: [Errno 101] Network is unreachable
Error retrieving file list. Please ensure your model_name_or_path, repo_type, token and revision arguments are correctly set.
```

修正策略：

```text
显式增加 --tokenizer /data/models/GLM-5.2-FP8-ms
显式增加 --trust-remote-code
显式设置 --endpoint /v1/chat/completions
先用小批次验证 bench 可正常完成，再扩大矩阵。
```

本次首次输出目录：

```text
/data/bench/glm52/20260630-023637
```

### 2026-06-30 执行记录：vLLM bench smoke baseline 完成

修正后执行目录：

```text
/data/bench/glm52/20260630-024404
```

产物：

```text
/data/bench/glm52/20260630-024404/run.log
/data/bench/glm52/20260630-024404/gpu_monitor.csv
/data/bench/glm52/20260630-024404/smoke_decode_512_128_r1.log
/data/bench/glm52/20260630-024404/smoke_decode_512_128_r1.json
/data/bench/glm52/20260630-024404/smoke_decode_512_128_r2.log
/data/bench/glm52/20260630-024404/smoke_decode_512_128_r2.json
```

修正后的 vLLM bench 命令模板：

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
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result \
  --result-filename <RESULT_JSON>
```

Smoke 测试结果：

| 测试 | Input | Output | Request Rate | Requests | 成功 | 输出吞吐 | Peak 输出吞吐 | TTFT p95 | TPOT p95 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke r1 | 512 | 128 | 1 req/s | 5 | 5 | 83.74 tok/s | 147 tok/s | 463.30ms | 23.31ms | 19.79ms | 3233.94ms |
| smoke r2 | 512 | 128 | 2 req/s | 8 | 8 | 149.28 tok/s | 238 tok/s | 261.72ms | 27.82ms | 24.68ms | 3607.00ms |

观察：

```text
request-rate 从 1 提升到 2 后，整体 output token throughput 从 83.74 tok/s 提升到 149.28 tok/s。
这验证了多请求 continuous batching 可以提高整体吞吐。
同时 TPOT/ITL 变大，说明单请求 token 生成间隔开始上升。
当前样本很小，只作为工具链跑通和 smoke baseline，不作为最终容量结论。
```

### 2026-06-30 执行记录：Decode baseline 小矩阵完成

远端输出目录：

```text
/data/bench/glm52/20260630-025746-decode-baseline
```

产物：

```text
run.log
gpu_monitor.csv
decode_512_512_r1.json / .log
decode_512_512_r2.json / .log
decode_512_512_r4.json / .log
decode_2k_512_r1.json / .log
decode_2k_512_r2.json / .log
decode_2k_512_r4.json / .log
```

本轮全部 `rc=0`，无失败请求。

已观察到的关键结果摘录：

| 场景 | Request Rate | Prompts | 输出吞吐 | Peak 输出吞吐 | TTFT p95 | TPOT p95 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K in / 512 out | 1 req/s | 20 | 267.16 tok/s | 552 tok/s | 5027ms | 51.81ms | 37.45ms | 27015ms |
| 2K in / 512 out | 2 req/s | 30 | 432.09 tok/s | 706 tok/s | 1451ms | 47.83ms | 44.24ms | 24565ms |
| 2K in / 512 out | 4 req/s | 40 | 568.06 tok/s | 840 tok/s | 2915ms | 55.31ms | 51.72ms | 28394ms |

初步观察：

```text
2K/512 场景下，request-rate 从 1 提升到 4，output throughput 从 267 tok/s 提升到 568 tok/s。
这说明多请求 batching 明显提高整体吞吐。
但 TPOT/ITL 和 E2E 延迟也有上升，说明并发增加后单请求体验开始变差。
```

注意：

```text
本轮是 baseline 小矩阵，不是完整 decode 矩阵。
512/512 三组结果已写入远端 JSON，最终报告会统一解析汇总。
```

### 2026-06-30 执行记录：Prefill baseline 缩减矩阵完成

远端输出目录：

```text
/data/bench/glm52/20260630-030252-prefill-baseline
```

产物：

```text
run.log
gpu_monitor.csv
prefill_bench.py
prefill_bench.log
prefill_summary.json
prefill_<context>_c<concurrency>.json
```

全部完成，`prefill_rc=0`。

单并发 TTFT 曲线：

| Context | 实际 prompt tokens | 并发 | 请求数 | 平均 TTFT | p95 TTFT | 平均延迟 | QPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8166 | 1 | 2 | 2.33s | 2.35s | 2.58s | 0.387 |
| 16K | 16358 | 1 | 2 | 4.66s | 4.66s | 4.90s | 0.204 |
| 32K | 32742 | 1 | 2 | 9.49s | 9.50s | 9.72s | 0.103 |
| 64K | 65510 | 1 | 2 | 19.74s | 19.75s | 19.94s | 0.050 |
| 128K | 131046 | 1 | 1 | 42.88s | 42.88s | 43.08s | 0.023 |
| 256K | 262054 | 1 | 1 | 100.62s | 100.62s | 100.72s | 0.010 |

观察：

```text
Prefill TTFT 随上下文长度近似线性增长。
长上下文下并发增加主要带来排队，QPS 基本不随并发明显提升。
256K 单请求 TTFT 已约 100s，应只作为长任务专用能力，不适合作为普通在线默认服务。
```

### 2026-06-30 执行记录：综合业务混合流量 baseline 完成

远端输出目录：

```text
/data/bench/glm52/20260630-031603-mixed-baseline
```

业务画像：

| 类型 | 占比 | 实际 input tokens | output tokens |
|---|---:|---:|---:|
| short | 60% | 约 961 | 192 |
| rag | 25% | 约 8130 | 384 上限 |
| agent | 10% | 约 16321 | 768 上限 |
| long | 5% | 约 65473 | 512 上限 |

结果摘要：

| Request Rate | Requests | 成功 | 实际 QPS | Output TPS | Overall TTFT p50 | Overall TTFT p95 | Overall E2E p50 | Overall E2E p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 req/s | 12 | 12 | 0.205 | 49.94 | 0.37s | 4.67s | 3.95s | 13.35s |
| 0.5 req/s | 20 | 20 | 0.312 | 83.09 | 2.33s | 19.97s | 29.48s | 46.59s |
| 1.0 req/s | 30 | 30 | 0.344 | 89.26 | 22.81s | 41.78s | 65.10s | 79.92s |

关键观察：

```text
0.2 req/s 表现相对可接受。
0.5 req/s 开始出现明显排队，短请求也被拖慢。
1.0 req/s 下实际 QPS 只有 0.344，说明服务已经无法按 1.0 req/s 消化该混合画像。
在混合流量下，少量 long/agent/rag 请求会显著影响 short 请求体验。
```

短请求在 1.0 req/s 下的表现：

```text
short TTFT p50: 22.31s
short E2E p50: 66.07s
short E2E p95: 76.86s
```

结论：

```text
该混合画像不能直接放在一个未隔离的实例上高强度混跑。
Gateway 应按 input tokens / 请求类型分流，至少将 long-context 请求与 short-chat 请求隔离。
```

## 2026-06-30 最终汇总

最终报告已生成：`/root/demo/docs/glm52-performance-final-report-2026-06-30.md`。

本轮已完成：

1. vLLM bench 工具验证与 tokenizer 本地化修正。
2. Decode baseline：512/512、2K/512，在 request rate 1/2/4 下测试。
3. 长上下文 Prefill baseline：8K、16K、32K、64K、128K、256K 的代表性测试。
4. 综合业务混合流量：short/rag/agent/long 混合比例模拟。
5. Recipe/MTP 对比：512/512 严格 A/B，8K/1024 Recipe 探索测试。

当前服务状态：仍运行 Recipe/MTP 配方，`max-model-len=262144`。

## 下一轮测试大纲修订版：聚焦 Decode 与服务端瓶颈

修订日期：2026-06-30

### 1. 修订原则

下一轮测试不再扩大上下文长度矩阵，重点从“能不能跑长上下文”转向“正式服务边界在哪里”。

核心问题：

1. 不同输出长度下，GLM-5.2-FP8 的稳定 decode 吞吐是多少。
2. 并发提高后，整体 output tok/s 是否明显提升，在哪个点开始变平。
3. 单请求体验如何变化，特别是 TTFT、TPOT、ITL、E2E 的 p95/p99。
4. 当前瓶颈是在 GPU、KV cache、scheduler、请求排队，还是 API/客户端侧。
5. 当前 Recipe/MTP 配方是否值得作为正式配置保留。

### 2. 明确降级：不优先做 32K vs 262K 对比

`32K vs 262K max-model-len` 对比降级为 P2，本轮不优先执行。

原因：

1. 当前服务已经以 `max-model-len=262144` 启动，长上下文能力已经验证过。
2. 32K 配置主要用于低延迟在线服务池优化，不是当前最关键的上线决策点。
3. 切换 `max-model-len` 需要重启服务，可能引入编译、cache、warmup 等额外变量。
4. 如果后续明确要单独建设 32K 低延迟服务池，再针对 32K 单独测试。

### 3. P0 测试一：Decode 输出长度矩阵

目标：确认输出长度变化对整体吞吐、单请求延迟和流式体验的影响。

固定条件：

```text
模型：GLM-5.2-FP8
服务：vLLM OpenAI API Server
max-model-len：262144
tensor-parallel-size：8
kv-cache-dtype：fp8
```

P0 矩阵：

| 输入长度 | 输出长度 | request rate | 目的 |
|---:|---:|---:|---|
| 1K | 128 | 1 / 2 / 4 / 8 | 短输出低延迟场景 |
| 1K | 512 | 1 / 2 / 4 / 8 | 普通问答主场景 |
| 1K | 1024 | 1 / 2 / 4 / 8 | 中长回复场景 |
| 1K | 2048 | 1 / 2 / 4 | 长输出压力场景 |
| 8K | 512 | 1 / 2 / 4 | RAG 场景 |
| 8K | 1024 | 1 / 2 / 4 | Agent/RAG 长回复场景 |

P1 矩阵：

| 输入长度 | 输出长度 | request rate | 目的 |
|---:|---:|---:|---|
| 8K | 128 | 1 / 2 / 4 | RAG 短答场景 |
| 8K | 2048 | 1 / 2 | RAG 长答场景 |
| 16K | 512 | 1 / 2 | Agent 中等输出 |
| 16K | 1024 | 1 / 2 | Agent 长输出 |

每个测试点输出指标：

1. 成功请求数、失败请求数。
2. request throughput，req/s。
3. output throughput，tok/s。
4. TTFT p50/p95/p99。
5. TPOT p50/p95/p99。
6. ITL p50/p95/p99。
7. E2E p50/p95/p99。
8. max concurrent requests。

### 4. P0 测试二：服务端观测增强

目标：把客户端压测结果和服务端状态对应起来，定位瓶颈。

压测期间同步采集：

| 指标 | 来源 | 目的 |
|---|---|---|
| GPU util | `nvidia-smi dmon` / DCGM | 判断 GPU 是否吃满 |
| GPU memory used | `nvidia-smi` | 判断显存水位 |
| power draw | `nvidia-smi dmon` | 判断 GPU 是否处于高负载状态 |
| KV cache usage | vLLM `/metrics` | 判断 KV cache 是否成为限制 |
| running requests | vLLM `/metrics` | 判断正在执行的请求数 |
| waiting requests | vLLM `/metrics` | 判断排队情况 |
| prefill/decode 时间 | vLLM `/metrics` | 区分 Prefill 与 Decode 瓶颈 |
| output tok/s | vLLM bench + metrics | 对齐客户端和服务端吞吐 |

最小采集方式：

```bash
nvidia-smi dmon -s pucmt -d 1
```

推荐同时采集 vLLM metrics：

```bash
curl -s http://127.0.0.1:8000/metrics
```

注意：vLLM metric 名称以当前 `/metrics` 实际输出为准，不在大纲中硬编码。

### 5. P0 测试三：Recipe/MTP 小范围 A/B

目标：判断当前 Recipe/MTP 配方是否适合作为正式启动配置。

A/B 代表点：

| 输入长度 | 输出长度 | request rate | 业务含义 |
|---:|---:|---:|---|
| 1K | 512 | 4 | 普通问答主场景 |
| 1K | 1024 | 4 | 中长输出场景 |
| 8K | 1024 | 2 | RAG/Agent 场景 |
| 1K | 2048 | 2 | 长输出 decode 压力 |

对比配置：

| 配置 | 说明 |
|---|---|
| Baseline | 不启用 MTP/speculative decoding |
| Recipe/MTP | 启用 `--speculative-config.method mtp` 与相关 parser/tool 参数 |

判断规则：

| 结果 | 决策 |
|---|---|
| output tok/s 明显提升，ITL/E2E 可接受 | 可考虑保留 MTP |
| output tok/s 小幅提升，但 ITL p95/p99 明显恶化 | 只适合批量或非流式场景 |
| output tok/s 无明显提升 | 回到 baseline |
| acceptance rate 偏低且波动大 | 不建议作为默认配置 |

### 6. P1 测试：小规模真实业务混合流量

目标：验证短请求是否会被长请求拖慢，并为 Gateway 分流策略提供依据。

业务画像：

| 类型 | 占比 | 输入长度 | 输出长度 |
|---|---:|---:|---:|
| short | 60% | 1K | 128 / 256 |
| rag | 25% | 8K | 512 |
| agent | 10% | 16K | 1024 |
| long | 5% | 64K | 512 |

测试 request rate：

```text
0.2 / 0.5 / 1.0 req/s
```

该组测试目标不是压满机器，而是观察：

1. short 请求的 TTFT p95/p99 是否被拖高。
2. long/agent 请求是否造成 waiting requests 持续累积。
3. 是否必须将 short、rag/agent、long 分成不同服务池。

### 7. 下一轮报告必须新增的内容

1. Decode 输出长度矩阵表。
2. output tok/s 随 request rate 变化的趋势。
3. TPOT/ITL 随输出长度变化的趋势。
4. GPU util 与 output tok/s 的对应关系。
5. KV cache usage 与上下文/并发的对应关系。
6. waiting requests 与 TTFT p95/p99 的对应关系。
7. Recipe/MTP 是否保留的明确建议。
8. 推荐生产启动参数。
9. Gateway 分流建议，包括 short、rag/agent、long 的拆分策略。

## 指标口径修订：以 tokens/s 为主，QPS 为业务画像派生指标

修订日期：2026-06-30

### 1. 为什么不把 QPS 作为主指标

大模型请求不是等价单位。不同请求的输入 token、输出 token、上下文长度差异很大，因此单独说 QPS 容易误导。

例如同样是 `1 QPS`：

| 请求类型 | 输入 tokens | 输出 tokens | 资源压力 |
|---|---:|---:|---|
| 短问答 | 1K | 128 | 较低 |
| 普通问答 | 1K | 512 | 中等 |
| RAG | 8K | 512 / 1024 | Prefill 和 Decode 都较高 |
| Agent | 16K | 1024 | 高 |
| 长文档 | 64K+ | 512+ | Prefill、KV cache、排队压力高 |

因此：

```text
QPS 必须绑定请求画像才有意义。
```

更合理的表达方式是：

```text
在 1K input / 512 output / TTFT p95 < 2s / E2E p95 < 30s 条件下，可承载 X QPS。
```

而不是：

```text
模型 QPS 是 X。
```

### 2. 后续报告的主指标

后续报告应以 token 维度指标为主。

| 指标 | 说明 | 主要反映 |
|---|---|---|
| output tokens/s | 每秒输出 token 数 | Decode 产能，是在线生成能力的核心指标 |
| total tokens/s | input + output 总 token 吞吐 | 整体 token 处理能力 |
| prefill tokens/s | 输入 token 处理速度 | 长上下文首 token 能力 |
| TTFT p50/p95/p99 | 首 token 延迟 | 排队、Prefill、调度开销 |
| TPOT p50/p95/p99 | 每输出 token 平均耗时 | Decode 稳态速度 |
| ITL p50/p95/p99 | 相邻 token 间隔 | 流式输出体感 |
| E2E p50/p95/p99 | 请求端到端完成时间 | 用户完整等待时间 |
| error rate | 失败率 | 稳定性 |

优先级：

```text
output tokens/s > TPOT/ITL > TTFT > E2E > QPS
```

但如果是长上下文场景，优先级需要调整为：

```text
TTFT / prefill tokens/s > KV cache usage > E2E > output tokens/s
```

### 3. 后续报告的辅助指标

QPS、并发和 request rate 仍然需要保留，但必须作为辅助指标，并且必须绑定输入/输出 token 画像。

| 辅助指标 | 说明 |
|---|---|
| request throughput / QPS | 在指定请求画像和 SLA 下的请求吞吐 |
| request rate | 压测注入速率，不等于服务实际可承载 QPS |
| concurrency | 同时在途请求数 |
| average input tokens | 平均输入长度 |
| average output tokens | 平均输出长度 |
| max concurrent requests | 压测过程中观察到的最大在途请求数 |
| KV cache usage | KV cache 水位 |
| running requests | vLLM 正在处理的请求数 |
| waiting requests | vLLM 排队请求数 |
| GPU util | GPU 利用率 |
| GPU memory used | 显存使用量 |

### 4. QPS 的正确计算方式

QPS 应作为 token throughput 在业务画像下的换算结果。

近似公式：

```text
可承载 QPS ≈ 可用 output tokens/s / 平均 output tokens
```

但这个公式只考虑 Decode，不包含 Prefill、排队、KV cache、SLA 约束。

更完整的表达方式：

```text
最大稳定 QPS = 在指定 input/output token 画像下，满足 TTFT/E2E/ITL/error rate SLA 的最大 request throughput。
```

示例：

```text
如果服务稳定 output throughput 为 650 tok/s：

平均输出 128 tokens：理论 decode QPS ≈ 650 / 128 ≈ 5.1 req/s
平均输出 512 tokens：理论 decode QPS ≈ 650 / 512 ≈ 1.27 req/s
平均输出 1024 tokens：理论 decode QPS ≈ 650 / 1024 ≈ 0.63 req/s
平均输出 2048 tokens：理论 decode QPS ≈ 650 / 2048 ≈ 0.32 req/s
```

因此不同输出长度下看到不同 QPS 是正常现象。

### 5. 后续测试报告的推荐表述

后续报告中避免使用孤立 QPS，统一使用以下表述：

```text
场景：<input tokens> input / <output tokens> output
SLA：TTFT p95 < Xs，ITL p95 < Yms，E2E p95 < Zs，error rate < N%
结果：最大稳定 output throughput = A tok/s，最大稳定 request throughput = B req/s
```

示例：

```text
场景：1K input / 512 output
SLA：TTFT p95 < 2s，E2E p95 < 30s，error rate = 0
结果：最大稳定 output throughput = 600 tok/s，最大稳定 request throughput = 1.1 req/s
```

### 6. 对当前报告的解释口径

当前已完成报告中的 QPS 应按以下方式理解：

1. Decode 场景下的 QPS 是在固定输出 token 数下的派生值。
2. 512 output 场景中 QPS 约 1.x，并不代表模型只能服务 1 个短请求每秒。
3. 如果平均输出 token 降到 128，理论请求吞吐会显著提高。
4. 如果平均输出 token 提升到 1024 或 2048，QPS 会自然下降。
5. 对 GLM-5.2-FP8 这种大模型，`output tokens/s` 比孤立 QPS 更能反映硬件和模型组合的服务能力。

## 2026-06-30 新版大纲执行完成记录

本轮已按修订后的大纲完成三阶段测试：

1. 阶段一：当前 Recipe/MTP 配置下 Decode 输出长度矩阵与服务端观测。
2. 阶段二：切换非 MTP baseline，执行 A/B 代表点测试。
3. 阶段三：生成新测试报告。

新报告路径：`/root/demo/docs/glm52-decode-observability-ab-report-2026-06-30.md`。

远端结果目录：

1. `/data/bench/glm52/20260630-042139-decode-mtp-full`
2. `/data/bench/glm52/20260630-043738-baseline-ab`

当前服务最终状态：非 MTP baseline，`max-model-len=262144`。
