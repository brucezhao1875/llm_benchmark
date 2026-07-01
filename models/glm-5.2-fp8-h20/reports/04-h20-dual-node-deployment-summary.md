# GLM-5.2-FP8 H20 双机部署与测试总结报告

日期：2026-07-01  
范围：h20-01 / h20-02 两台 8 卡 H20 机器  
模型：GLM-5.2-FP8  
运行方式：vLLM OpenAI-compatible API server  
当前推荐服务形态：`256K context + FP8 KV cache + TP=8 + tool calling + MTP`

## 1. 总结结论

两台 H20 机器已经完成 GLM-5.2-FP8 部署路径验证。

当前状态：

| 机器 | 状态 | 说明 |
|---|---|---|
| h20-02 | 已部署并运行 | 先行完成模型下载、vLLM 运行、service 化、256K 配方、tool calling、MTP |
| h20-01 | 已部署并运行 | 已完成 RAID5、模型同步、venv 同步、service 部署、chat/tool call 验证 |

核心结论：

1. GLM-5.2-FP8 可以在单台 8 x H20 上以 `TP=8` 方式运行。
2. 当前 H20 单机的推荐上下文不是模型原生 1M，而是显式限制到 `256K`。
3. 官方 H20/H200 FP8 recipe 不能直接省略 `--max-model-len` 在 H20 上生产启动，因为 vLLM 会读取模型原生 `1048576` 上下文，导致 KV cache 不足。
4. 当前默认 service 已采用“官方 recipe + 256K + tool calling + MTP”的完整服务形态。
5. 当前 256K MTP/tool 配方下，每卡可用 KV cache 约 `24.26 GiB`，GPU KV cache capacity 为 `418,368 tokens`，满 256K 请求的理论并发为 `1.60x`。
6. 对普通短上下文请求，KV cache 够用；对多个接近 256K 的超长请求，KV cache 不够支撑高并发。
7. Gateway 必须做 token-aware routing，不能把所有请求都打到同一个 256K MTP/tool 池。

## 2. 相关测试报告引用

本报告引用以下已有文档和测试结果：

| 文档 | 内容 |
|---|---|
| [../report.md](../report.md) | GLM-5.2-FP8 on 8 x NVIDIA H20 unified benchmark report |
| [01-performance-baseline.md](01-performance-baseline.md) | 安装期 baseline、prefill、mixed traffic、早期 recipe 对比 |
| [02-decode-observability-ab.md](02-decode-observability-ab.md) | Decode matrix、GPU/vLLM 观测、MTP A/B |
| [03-context-window-comparison.md](03-context-window-comparison.md) | 32K / 256K / 1M 启动上下文对比、官方缺省配方验证、KV cache 计算 |
| [../notes/glm52-h20-installation-notes.md](../notes/glm52-h20-installation-notes.md) | H20 上 GLM-5.2-FP8 安装手记和知识库 |

## 3. 双机环境概况

### 3.1 GPU 环境

两台机器均为 8 卡 H20：

```text
GPU: 8 x NVIDIA H20-3e
单卡显存: 143,771 MiB 级别
运行方式: tensor parallel size = 8
```

h20-02 已确认 GPU 拓扑为全互联 `NV18`：

```text
nvidia-smi topo -m
GPU0-GPU7 之间均显示 NV18
```

含义：

```text
8 张 GPU 之间通过 NVLink / NVSwitch 体系互联
适合 TP=8 的单机张量并行
```

### 3.2 软件环境

当前运行环境：

| 项目 | 值 |
|---|---|
| vLLM | `0.23.0` |
| PyTorch | `2.11.0+cu130` |
| CUDA toolkit | `/usr/local/cuda-13.1` |
| Python venv | `/data/venvs/glm52` |
| 模型目录 | `/data/models/GLM-5.2-FP8-ms` |
| API 服务 | vLLM OpenAI-compatible API server |
| 服务端口 | `8000` |

service 环境变量已经在 h20-01 / h20-02 两台机器保持一致：

```text
PATH=/data/venvs/glm52/bin:/usr/local/cuda-13.1/bin:...
LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64
CUDA_HOME=/usr/local/cuda-13.1
CUDA_PATH=/usr/local/cuda-13.1
PYTHONUNBUFFERED=1
```

这个 CUDA 环境变量很关键。h20-01 首次启动时曾因为 systemd service 没有看到完整 CUDA toolkit 路径，在 FlashInfer / DeepGEMM kernel 编译或加载阶段失败：

```text
RuntimeError: Assertion failed: !cubin.empty() || isPathValid(path_)
```

修复方式：

```text
1. 停止 service
2. 清理失败的 ~/.cache/vllm 和 ~/.cache/flashinfer
3. 在 systemd unit 中加入 CUDA_HOME / CUDA_PATH / LD_LIBRARY_PATH / CUDA bin PATH
4. 重新启动，让 vLLM 重新编译缓存
```

修复后 h20-01 成功 ready。

## 4. 存储与模型同步情况

### 4.1 h20-02

h20-02 是先行部署机器，已具备：

```text
/data/models/GLM-5.2-FP8-ms
/data/venvs/glm52
/etc/systemd/system/glm52-vllm.service
```

模型大小：

```text
模型目录: 704G
模型文件总大小: 755,663,671,119 bytes
文件数量: 153 个，其中 regular files 150 个
```

### 4.2 h20-01 RAID5

h20-01 原始状态：

```text
4 块 3.5T NVMe 未做 RAID，未挂载
/data 只是根盘上的普通空目录
```

已完成操作：

```text
/dev/nvme0n1
/dev/nvme1n1
/dev/nvme2n1
/dev/nvme3n1
=> /dev/md0 RAID5
=> ext4
=> mounted on /data
```

RAID5 创建后容量：

```text
/dev/md0 ext4 约 11T
挂载点: /data
```

RAID resync 在后台进行。作用是初始化和校验 RAID5 的 parity，不影响 `/data` 使用，但会消耗磁盘 IO。

部署期间观察到的 resync 状态示例：

```text
md0 : active raid5 nvme3n1[3] nvme2n1[2] nvme1n1[1] nvme0n1[0]
resync = 34.8%
speed ~= 200MB/s
```

说明：

```text
resync 会扫描整个 RAID5 地址空间，不只是已写入的模型文件。
正式 FIO 或存储性能测试应等 resync 完成后再做。
```

### 4.3 h20-02 到 h20-01 模型同步

同步路径：

```text
h20-02:/data/models/GLM-5.2-FP8-ms
=> h20-01:/data/models/GLM-5.2-FP8-ms

h20-02:/data/venvs/glm52
=> h20-01:/data/venvs/glm52
```

实际网络：

```text
h20-01 内网 IP: 172.16.8.103
h20-02 内网 IP: 172.16.8.104
网卡协商速率: 1000Mb/s
```

因此模型同步速度主要受 1Gbps 链路限制。单流 rsync 约 `100-112MB/s`，后续采用 4 路分片 rsync，但总吞吐仍接近 1Gbps 物理上限。

最终对齐结果：

```text
模型:
Number of files: 153
Total file size: 755,663,671,119 bytes
Number of regular files transferred: 0

venv:
Number of files: 82,267
Total file size: 9,496,231,543 bytes
Number of regular files transferred: 0
```

这说明最终 h20-01 与 h20-02 的模型目录和 venv 已对齐。

## 5. 当前 vLLM service 配方

两台机器当前 service 使用同一套配方。

unit 路径：

```text
/etc/systemd/system/glm52-vllm.service
```

核心启动命令：

```bash
/data/venvs/glm52/bin/vllm serve /data/models/GLM-5.2-FP8-ms \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 5 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --served-model-name glm-5.2-fp8 \
  --trust-remote-code
```

这个配方的含义：

| 参数 | 作用 |
|---|---|
| `--tensor-parallel-size 8` | 单机 8 卡 TP 并行 |
| `--kv-cache-dtype fp8` | KV cache 使用 FP8，降低显存占用 |
| `--max-model-len 262144` | 显式限制单请求最大上下文为 256K |
| `--gpu-memory-utilization 0.90` | vLLM 规划使用 90% GPU 显存 |
| `--speculative-config.method mtp` | 启用 MTP speculative decoding |
| `--speculative-config.num_speculative_tokens 5` | 使用 5 个 speculative tokens |
| `--tool-call-parser glm47` | 启用 GLM tool calling parser |
| `--reasoning-parser glm45` | 启用 GLM reasoning parser |
| `--enable-auto-tool-choice` | 支持自动工具选择 |

这套配置可以理解为：

```text
官方 H20/H200 FP8 recipe
+ 本地模型路径
+ 显式 256K 上下文限制
+ tool calling
+ reasoning parser
+ MTP
+ systemd service 化
```

不能省略 `--max-model-len 262144`。原因见 [03-context-window-comparison.md](03-context-window-comparison.md)。

## 6. 启动与兼容性验证

### 6.1 h20-02

h20-02 当前已运行完整配方 service。

已验证：

```text
/v1/models ready
普通 chat HTTP 200
tool call HTTP 200
```

tool call 测试结果：

```text
tool_calls_count=1
tool_call_name=get_current_weather
tool_call_arguments={"city": "上海"}
```

### 6.2 h20-01

h20-01 首次启动经历了两阶段：

| 阶段 | 结果 | 原因 |
|---|---|---|
| 第一次启动 | 失败 | FlashInfer / DeepGEMM cubin 生成或加载失败 |
| 修复 CUDA service 环境后重启 | 成功 | 重新生成编译缓存 |

成功启动结果：

```text
ready_after_seconds=920
约 15 分 20 秒
```

启动日志关键指标：

```text
Available KV cache memory: 24.26 GiB
GPU KV cache size: 418,368 tokens
Maximum concurrency for 262,144 tokens per request: 1.60x
```

普通 chat 测试：

```text
HTTP 200
finish_reason=stop
content=我是由Z.ai训练的GLM大语言模型。
```

tool call 测试：

```text
HTTP 200
finish_reason=stop
tool_calls_count=1
tool_call_name=get_current_weather
tool_call_arguments={"city": "上海"}
```

## 7. 压测数据摘要

完整数据见：

```text
../report.md
01-performance-baseline.md
02-decode-observability-ab.md
03-context-window-comparison.md
```

### 7.1 32K baseline

32K 启动配置适合作为低延迟在线池候选。

代表性结果：

| 场景 | Rate | Output tok/s | Req/s | TTFT p95 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1K / 128 | 8 | 412.01 | 3.22 | 3476ms | 51.08ms | 10398ms |
| 1K / 512 | 4 | 590.28 | 1.15 | 162ms | 44.58ms | 20940ms |
| 1K / 512 | 8 | 785.86 | 1.53 | 171ms | 54.64ms | 26100ms |
| 1K / 1024 | 4 | 645.89 | 0.63 | 157ms | 45.35ms | 43827ms |
| 8K / 512 | 4 | 326.36 | 0.64 | 13365ms | 37.73ms | 34004ms |
| 8K / 1024 | 2 | 426.85 | 0.42 | 196ms | 31.69ms | 31185ms |

解释：

```text
短上下文下，output tokens/s 是主要指标。
QPS 会随着输出 token 数增加而自然下降。
```

### 7.2 256K baseline

256K baseline 是长上下文/Agent/RAG 池的基础参考。

代表性结果：

| 场景 | Rate | Output tok/s | Req/s | TTFT p95 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1K / 512 | 4 | 516.84 | 1.01 | 2562ms | 45.06ms | 30653ms |
| 1K / 1024 | 4 | 644.32 | 0.63 | 145ms | 45.74ms | 44014ms |
| 8K / 1024 | 2 | 256.34 | 0.25 | 24256ms | 31.95ms | 62788ms |

说明：

```text
256K baseline 可用，但在部分长输入场景 TTFT p95 波动较大。
长上下文池需要单独观察 TTFT 和 waiting queue，不能只看 output tok/s。
```

### 7.3 256K MTP

MTP 对 decode 吞吐有明显帮助，但会提高 ITL p95，流式体感可能变差。

代表性结果：

| 场景 | Rate | Output tok/s | Req/s | TTFT p95 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1K / 512 | 4 | 617.01 | 1.21 | 320ms | 112.24ms | 21044ms |
| 1K / 512 | 8 | 785.79 | 1.53 | 508ms | 143.08ms | 27396ms |
| 1K / 1024 | 4 | 722.13 | 0.71 | 352ms | 109.42ms | 41079ms |
| 1K / 1024 | 8 | 889.80 | 0.87 | 486ms | 139.20ms | 50836ms |
| 8K / 1024 | 2 | 436.34 | 0.43 | 295ms | 82.60ms | 31109ms |

说明：

```text
MTP 更适合高吞吐池、批量生成、非强流式体验场景。
如果目标是最低 ITL/更平滑 streaming，应保留一个非 MTP baseline 池。
```

### 7.4 当前默认 service 的 KV 容量

当前默认 service 是 256K + MTP + tool calling。

启动日志：

```text
Available KV cache memory: 24.26 GiB
GPU KV cache size: 418,368 tokens
Maximum concurrency for 262,144 tokens per request: 1.60x
```

含义：

```text
如果每个请求都打满 256K，上不了 2 个完整并发。
如果平均上下文是 32K，则 KV cache 角度理论容量约为 12.8 个请求。
如果平均上下文是 8K，则 KV cache 角度理论容量约为 51 个请求。
```

所以不能用单一 QPS 判断模型服务能力，必须按 token 形态描述：

```text
Under <input tokens>/<output tokens> and SLA <TTFT/E2E/ITL/error rate>,
the service sustains <X output tokens/s> and <Y req/s>.
```

## 8. 1M 上下文结论

GLM-5.2-FP8 模型原生支持 1M context，但当前 H20 单机不适合直接跑 1M 服务。

官方缺省 H20/H200 recipe 如果不写 `--max-model-len`，vLLM 会读取模型 config 中的 `1048576`。

实测结果见 [03-context-window-comparison.md](03-context-window-comparison.md)：

```text
Using max model len 1048576
Available KV cache memory: 22.87 GiB
1M required KV cache: 60.79 GiB / GPU
vLLM estimated maximum model length: 394496 tokens
```

结论：

```text
H20 上必须显式写 --max-model-len 262144。
1M 不应作为默认服务池。
如果要支持 1M，需要独立 ultra-long pool，并优先考虑更大显存硬件。
```

## 9. Gateway 与生产部署建议

当前两台 H20 应作为两个独立的 8 卡 vLLM backend 节点使用。

建议的 Gateway 路由策略：

| 请求类型 | 推荐池 |
|---|---|
| 纯短 chat，输入 <= 8K，输出 <= 512 | 32K baseline 低延迟池 |
| 普通 RAG / Agent，输入 <= 32K | 32K 或 256K 标准池 |
| 长上下文，输入 32K-256K | 256K long-context 池 |
| tool calling / agent function call | 当前 256K tool/MTP 池 |
| 长输出、批量生成 | 256K MTP 高吞吐池 |
| 接近 1M context | 不进入默认在线池，走单独实验池或异步任务 |

必须在 Gateway 层实现：

```text
1. input token 预估
2. max_tokens / output budget 解析
3. pool routing
4. per-pool admission control
5. in-flight token accounting
6. TTFT / ITL / E2E / output tokens/s 观测
7. per-pool billing
```

当前不建议把所有流量都打到一个 `256K + MTP + tool` 池。

更合理的生产形态：

```text
32K baseline pool: 低延迟、普通 chat
256K baseline pool: 长上下文、Agent、RAG
256K MTP/tool pool: tool calling、高吞吐、长输出
1M pool: 暂不默认提供，作为后续专项实验
```

## 10. 运维注意事项

### 10.1 查看服务状态

```bash
systemctl status glm52-vllm --no-pager
```

### 10.2 查看启动日志

```bash
journalctl -u glm52-vllm -b --no-pager | grep -Ei "Using max model len|KV cache|Maximum concurrency|Started server|ERROR|Traceback"
```

### 10.3 查看 API 是否 ready

```bash
curl -s http://127.0.0.1:8000/v1/models
```

### 10.4 查看 RAID resync

```bash
cat /proc/mdstat
```

### 10.5 查看 GPU

```bash
nvidia-smi
nvidia-smi topo -m
```

### 10.6 注意首次启动耗时

h20-01 修复环境后首次成功启动用时：

```text
920s
```

原因：

```text
模型加载
Torch compile
FlashInfer / DeepGEMM kernel 编译
CUDA graph capture
warmup
RAID resync 期间磁盘 IO 干扰
```

后续重启如果 cache 完整，理论上会快于首次启动，但仍应预留数分钟级启动窗口。

## 11. 最终状态

截至本报告：

| 项目 | h20-02 | h20-01 |
|---|---|---|
| RAID / data disk | 已有 `/data` | 已创建 RAID5 并挂载 `/data` |
| 模型 | 已安装 | 已从 h20-02 同步完成 |
| venv | 已安装 | 已从 h20-02 同步完成 |
| systemd service | 已配置 | 已配置 |
| CUDA service env | 已补齐 | 已补齐 |
| vLLM ready | 是 | 是 |
| 普通 chat | 通过 | 通过 |
| tool call | 通过 | 通过 |
| 当前配方 | 256K + MTP + tool | 256K + MTP + tool |

最终判断：

```text
GLM-5.2-FP8 在两台 H20 单机 8 卡环境上已经形成可复制部署方案。
当前生产可用的基础配置是 256K context。
后续重点不再是“能否安装”，而是 Gateway 分池、token-aware routing、正式压测和 SLA 定义。
```
