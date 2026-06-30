# GLM-5.2-FP8 vLLM 小规模性能压测报告

> 测试时间：2026-06-29 18:29 CST / 2026-06-29 10:29 UTC
> 测试机器：h20-02
> 服务方式：systemd `glm52-vllm.service`
> 模型：`glm-5.2-fp8`
> 模型路径：`/data/models/GLM-5.2-FP8-ms`
> API：`http://127.0.0.1:8000/v1/chat/completions`
> 测试类型：小规模 streaming 压测，不代表极限吞吐

## 1. 测试目标

本次测试关注三个基础指标：

| 指标 | 含义 |
|---|---|
| TTFT | Time To First Token，从请求发出到收到第一个输出 token/chunk 的时间 |
| 吞吐量 | 单请求输出 tokens/s，以及服务整体 aggregate output tokens/s |
| Token 间隔 | streaming 输出中相邻 token/chunk 之间的时间间隔，近似衡量生成平滑度 |

## 2. 测试方法

使用 Python 标准库直接请求 vLLM OpenAI-compatible streaming API。

请求参数：

```json
{
  "model": "glm-5.2-fp8",
  "messages": [
    {
      "role": "user",
      "content": "请用大约120个中文字说明大模型推理服务中TTFT、吞吐量和token间隔分别代表什么。"
    }
  ],
  "temperature": 0,
  "max_tokens": 128,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

测试档位：

| 档位 | 并发 | 请求数 | 单请求输出上限 |
|---|---:|---:|---:|
| A | 1 | 3 | 128 tokens |
| B | 2 | 4 | 128 tokens |

说明：

```text
本次只做轻量测试，目的是确认服务基础性能和流式指标，不做容量上限判断。
```

## 3. 汇总结果

| 并发 | 请求数 | 成功数 | 失败数 | 平均 TTFT | p50 TTFT | p95 TTFT | 平均延迟 | p95 延迟 | 单请求端到端输出 TPS | 单请求 decode TPS | aggregate output TPS | QPS | 平均 token 间隔 | p95 token 间隔 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 0 | 0.081s | 0.053s | 0.138s | 2.256s | 2.315s | 56.76 tok/s | 58.86 tok/s | 56.71 tok/s | 0.443 | 17.54ms | 17.46ms |
| 2 | 4 | 4 | 0 | 0.124s | 0.100s | 0.260s | 2.358s | 2.455s | 54.34 tok/s | 57.37 tok/s | 108.11 tok/s | 0.845 | 18.01ms | 18.01ms |

## 4. 单请求明细

### 4.1 并发 1

| 请求 | HTTP | TTFT | 总耗时 | completion tokens | 端到端 TPS | decode TPS | 平均 token 间隔 | p95 token 间隔 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.138s | 2.315s | 128 | 55.30 | 58.82 | 17.55ms | 17.39ms |
| 1 | 200 | 0.053s | 2.224s | 128 | 57.55 | 58.96 | 17.51ms | 17.45ms |
| 2 | 200 | 0.052s | 2.229s | 128 | 57.44 | 58.81 | 17.55ms | 17.46ms |

### 4.2 并发 2

| 请求 | HTTP | TTFT | 总耗时 | completion tokens | 端到端 TPS | decode TPS | 平均 token 间隔 | p95 token 间隔 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.061s | 2.420s | 128 | 52.89 | 54.27 | 19.02ms | 18.01ms |
| 1 | 200 | 0.260s | 2.455s | 128 | 52.15 | 58.31 | 17.70ms | 17.74ms |
| 2 | 200 | 0.077s | 2.280s | 128 | 56.14 | 58.11 | 17.76ms | 17.55ms |
| 3 | 200 | 0.100s | 2.278s | 128 | 56.18 | 58.77 | 17.57ms | 17.55ms |

## 5. 指标解释

### 5.1 TTFT

本次 TTFT 很低：

```text
并发 1 平均 TTFT：0.081s
并发 2 平均 TTFT：0.124s
```

这说明在短 prompt、低并发、模型已热启动的情况下，首 token 返回很快。

需要注意：

```text
本次 prompt 只有 36 tokens。
如果 prompt 增加到 4K、16K、32K，TTFT 会明显增加，因为 prefill 计算量会变大。
```

### 5.2 吞吐量

本次单请求 decode TPS 约：

```text
57-59 tokens/s
```

服务整体 aggregate output TPS：

```text
并发 1：约 56.71 tokens/s
并发 2：约 108.11 tokens/s
```

这说明并发从 1 提升到 2 后，整体输出吞吐接近翻倍，低并发下 GPU batching 仍有提升空间。

### 5.3 Token 间隔

本次平均 token 间隔：

```text
并发 1：约 17.54ms
并发 2：约 18.01ms
```

近似对应 decode 速度：

```text
1 / 0.0175 ≈ 57 tokens/s
```

这与 decode TPS 结果一致。

## 6. 初步结论

1. 当前 systemd 启动的 vLLM 服务可正常响应 streaming API。
2. 在短 prompt、低并发、128 tokens 输出下，TTFT 表现很好，约 0.05-0.26 秒。
3. 单请求 decode 速度约 57-59 tokens/s。
4. 并发 2 下 aggregate output TPS 约 108 tokens/s，说明低并发下吞吐可随并发提升。
5. Token 间隔约 17-18ms，输出节奏稳定。
6. 本次测试请求量很小，只能作为热启动低并发基线，不能代表最大并发或生产容量。

## 7. 局限性

本次测试不覆盖：

```text
长 prompt prefill 性能
32K / 64K / 128K 上下文
高并发排队
p95 / p99 稳定性
长时间运行稳定性
不同 max_tokens 输出长度
真实业务 prompt 分布
Gateway 转发开销
跨机器负载均衡
```

因此不能用本次数据直接声明生产 QPS 上限。

## 8. 后续建议测试矩阵

建议下一步按业务形态做小矩阵：

| 场景 | input tokens | output tokens | 并发 |
|---|---:|---:|---|
| 短问答 | 512 | 128 | 1 / 2 / 4 / 8 |
| 普通 RAG | 2K | 256 | 1 / 2 / 4 / 8 |
| Agent | 8K | 512 | 1 / 2 / 4 |
| 长上下文 | 32K | 512 | 1 / 2 |

每组记录：

```text
TTFT
平均延迟
p50 / p95 / p99 延迟
单请求 decode TPS
aggregate output TPS
token 间隔 p50 / p95
失败率
GPU util
KV cache 使用情况
```

## 9. 本次测试结论一句话

在模型已热启动、短 prompt、低并发条件下，GLM-5.2-FP8 on 8xH20 的 vLLM 服务表现正常：TTFT 低，单请求 decode 约 58 tok/s，并发 2 时整体吞吐约 108 tok/s。

## 10. 补充：多并发吞吐测试

> 补充测试时间：2026-06-29 18:31 CST / 2026-06-29 10:31 UTC
> 目的：观察并发提升时 aggregate output TPS、QPS、TTFT、延迟和 token 间隔的变化。
> 说明：仍然是小规模测试，每档请求数较少，不代表最终容量上限。

### 10.1 测试配置

请求参数：

```json
{
  "model": "glm-5.2-fp8",
  "messages": [
    {
      "role": "user",
      "content": "请用大约100个中文字解释为什么大模型服务压测需要同时观察TTFT、延迟和吞吐量。"
    }
  ],
  "temperature": 0,
  "max_tokens": 128,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

并发档位：

| 并发 | 请求数 | 单请求输出 tokens |
|---:|---:|---:|
| 1 | 4 | 128 |
| 2 | 6 | 128 |
| 4 | 8 | 128 |
| 8 | 16 | 128 |

### 10.2 多并发结果

| 并发 | 请求数 | 成功数 | 失败数 | QPS | aggregate output TPS | 平均 TTFT | p95 TTFT | 平均延迟 | p95 延迟 | 单请求 decode TPS | 单请求端到端 TPS | 平均 token 间隔 | p95 token 间隔 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 0 | 0.447 | 57.23 | 0.064s | 0.097s | 2.235s | 2.271s | 58.96 | 57.27 | 17.37ms | 17.40ms |
| 2 | 6 | 6 | 0 | 0.873 | 111.70 | 0.089s | 0.119s | 2.285s | 2.300s | 58.28 | 56.01 | 17.57ms | 17.75ms |
| 4 | 8 | 8 | 0 | 1.513 | 193.71 | 0.183s | 0.317s | 2.637s | 2.742s | 52.23 | 48.61 | 19.51ms | 19.48ms |
| 8 | 16 | 16 | 0 | 2.486 | 318.17 | 0.168s | 0.247s | 3.206s | 3.273s | 42.15 | 39.93 | 24.27ms | 24.05ms |

### 10.3 观察结论

并发提升后，整体吞吐明显上升：

```text
并发 1：约 57 tok/s
并发 2：约 112 tok/s
并发 4：约 194 tok/s
并发 8：约 318 tok/s
```

同时，单请求体验开始下降：

```text
单请求 decode TPS：58.96 -> 58.28 -> 52.23 -> 42.15 tok/s
平均 token 间隔：17.37ms -> 17.57ms -> 19.51ms -> 24.27ms
平均延迟：2.24s -> 2.29s -> 2.64s -> 3.21s
```

说明当前低并发区间里，vLLM batching 可以显著提升整体吞吐；但随着并发增加，单请求 decode 速度下降，token 间隔变长，用户侧体感会逐步变慢。

### 10.4 初步判断

在当前短 prompt、128 token 输出的 workload 下：

```text
并发 1-2：单请求体验最好，吞吐较低。
并发 4：吞吐提升明显，单请求延迟仍可接受。
并发 8：aggregate TPS 最高，但单请求 TPS 已明显下降。
```

这组数据还没有看到明显的系统饱和拐点，因为：

```text
失败数为 0
TTFT 仍然低于 0.4s
p95 延迟仍在 3.3s 内
```

但已经能看到并发增大带来的单请求生成速度下降。

### 10.5 下一步建议

如果要找真正容量边界，下一步应继续测试：

```text
并发 16
并发 32
```

但建议把请求数和测试时长控制好，并同时观察：

```text
GPU util
GPU 显存
vLLM 日志
p95/p99 延迟
失败率
timeout
```

同时还需要测试更接近真实业务的输入长度：

```text
512 in / 128 out
2K in / 256 out
8K in / 512 out
32K in / 512 out
```

仅短 prompt 的并发吞吐不能代表 Agent 或长上下文场景。
