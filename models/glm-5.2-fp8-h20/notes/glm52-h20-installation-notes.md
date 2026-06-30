# GLM-5.2 在 H20 双机环境的安装部署手记与知识库

> 文档版本：v0.1
> 更新时间：2026-06-29
> 基于环境：h20-01 / h20-02，8 x NVIDIA H20-3e，GLM-5.2-FP8，vLLM OpenAI-compatible API
> 文档定位：安装手记、部署知识库、运维排障记录、后续压测与生产化参考
> 注意：本文记录的是当前环境和当前部署过程中的事实、判断和建议。实际生产指标必须以本机压测结果为准。

## 1. 背景与目标

本次工作的目标是确认 GLM-5.2 是否可以在现有两台 H20 服务器上部署运行，并形成后续正式部署、压测、服务化和网关接入的基础方案。

当前机器：

| 主机 | 角色建议 | 说明 |
|---|---|---|
| h20-01 | 独立 GLM-5.2-FP8 实例 / 后续压测实例 / 长上下文实验实例 | 不建议与 h20-02 跨机 TP 组成一个模型实例 |
| h20-02 | 当前已完成下载和 vLLM 启动验证的实例 | 当前模型文件位于 `/data/models/GLM-5.2-FP8-ms` |

当前已完成事项：

| 项目 | 状态 |
|---|---|
| 确认 H20 机器硬件情况 | 已完成 |
| 下载 GLM-5.2-FP8 模型 | 已完成 |
| 验证模型文件完整性 | 已完成 |
| 创建 Python/vLLM 运行环境 | 已完成 |
| 使用 vLLM 启动 OpenAI-compatible API | 已完成 |
| 本地 curl 调用验证 | 已完成 |
| 初步讨论上下文、KV cache、TP、NVLink/NVSwitch | 已完成 |
| systemd 正式服务化 | 待完成 |
| 标准化压测报告 | 待完成 |
| Gateway 接入、鉴权、计费、限流 | 待完成 |

## 2. 机器与硬件环境

### 2.1 GPU 与主机资源

两台机器均为 H20 GPU 服务器，核心特征如下：

| 项目 | 规格 |
|---|---|
| GPU | 8 x NVIDIA H20-3e |
| 单卡显存 | 约 143,771 MiB，约 141 GiB |
| 总 GPU 显存 | 约 1.1 TiB 级别 |
| 系统内存 | 约 1 TiB |
| 操作系统 | Ubuntu 22.04.5 |
| 数据盘 | h20-02 `/data` 挂载在 `/dev/md0`，约 11T |

h20-02 内网地址曾确认：

```text
172.16.8.104
```

h20-02 外网 SSH 端口映射发生过变更：

```text
旧：<redacted-host>:11104
新：<redacted-host>:11105
```

由于此前频繁 SSH 可能触发端口限制或防火墙策略，后续运维建议：

```text
减少短时间内频繁 SSH 登录
优先让长任务在远端后台运行
使用日志文件和监控文件观察状态
必要时通过 h20-01 跳板访问 h20-02
```

### 2.2 GPU 拓扑

h20-02 的 GPU 拓扑输出：

```text
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18    0-47,96-143     0               N/A
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18    0-47,96-143     0               N/A
GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18    0-47,96-143     0               N/A
GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18    0-47,96-143     0               N/A
GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18    48-95,144-191   1               N/A
GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18    48-95,144-191   1               N/A
GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18    48-95,144-191   1               N/A
GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X      48-95,144-191   1               N/A
```

关键结论：

```text
GPU0-GPU7 任意两张之间都是 NV18
这说明单机 8 卡之间具备高带宽 NVLink/NVSwitch 互联
该机器非常适合单机 TP=8
```

常见拓扑标记含义：

| 标记 | 含义 | 对 TP 的影响 |
|---|---|---|
| X | 自己到自己 | 不适用 |
| NV# | 通过 NVLink/NVSwitch 互联，# 为链路级别 | 最优 |
| PIX | 同一个 PCIe switch 下 | 较好 |
| PXB | 跨多个 PCIe bridge/switch | 中等 |
| PHB | 跨 PCIe host bridge | 较差 |
| NODE | 跨 host bridge 但在同一 NUMA node | 较差 |
| SYS | 跨 CPU socket / 系统互联 | 最差 |

### 2.3 NUMA 的含义

NUMA 是 Non-Uniform Memory Access，表示 CPU 访问本地内存和访问远端 CPU 插槽下的内存速度不同。

当前拓扑显示：

```text
GPU0-GPU3 靠近 NUMA 0
GPU4-GPU7 靠近 NUMA 1
```

对应 CPU affinity：

```text
GPU0-GPU3: CPU 0-47,96-143
GPU4-GPU7: CPU 48-95,144-191
```

这通常说明机器是双路 CPU 结构。

对 vLLM 的影响：

```text
核心推理计算在 GPU 上
GPU-GPU 通信走 NVLink/NVSwitch
NUMA 主要影响 CPU 侧 tokenizer、网络、数据拷贝、进程调度
第一阶段不必过早做复杂绑核优化
```

### 2.4 NVLink 与 NVSwitch

NVLink 是 NVIDIA 的 GPU-GPU 高速互联。NVSwitch 是 GPU 间的高速交换芯片，可以理解为专门服务 GPU 的交换机。

当前看到 `NV18`，可以理解为：

```text
每张 GPU 有 18 条 NVLink 级别的接入能力
8 张 GPU 通过 NVSwitch fabric 实现高速全互联
```

需要注意：

```text
NV18 不表示 GPU0 到 GPU1 独占 18 条物理线
更常见结构是 GPU -> NVSwitch -> GPU
```

Hopper 代际常见 NVLink 口径：

```text
每 GPU 最大 NVLink 带宽约 900 GB/s 双向
18 条 link
每条 link 约 50 GB/s 双向，约 25 GB/s 单向 + 25 GB/s 反向
```

所以也经常看到两个数字：

```text
900 GB/s：每 GPU 双向总带宽
450 GB/s：每 GPU 单向总带宽
```

与 PCIe / IB 的关系：

| 类型 | 连接对象 | 主要用途 | 对本项目意义 |
|---|---|---|---|
| PCIe | CPU-GPU / 设备总线 | 通用设备通信 | 比 NVLink 慢，不适合高频 GPU-GPU 通信 |
| NVLink | GPU-GPU | 单机多 GPU 高速通信 | TP=8 的关键基础 |
| NVSwitch | 多 GPU 交换网络 | 单机多 GPU全互联 | 当前机器拓扑优秀 |
| IB | 服务器-服务器 | 多机训练/推理/RDMA | 只有跨机器并行才关键 |
| Ethernet | 服务器-服务器 | 普通网络 | 不适合跨机 TP |

结论：

```text
单机内部适合 TP=8
两台机器之间不建议做跨机 TP=16，除非确认有高性能 IB/RDMA 并完成 NCCL 压测
```

## 3. 模型选择与大小

### 3.1 GLM-5.2 基本判断

GLM-5.2 是智谱/Z.ai 近期开放的强模型，不是 5.2B 小模型。公开资料显示其为大规模 MoE 模型，参数量约 743B，active 参数约 39B 级别，支持长上下文，最长可到 1M tokens。

本次选择的是：

```text
GLM-5.2-FP8
```

选择 FP8 的原因：

```text
模型总参数量非常大
BF16/FP16 权重约 1.4TB 以上，仅权重就超过单机 8xH20 的舒适承载范围
FP8 权重约 700GB 级别，更适合单机 8xH20 部署
官方/原生 FP8 通常比事后量化更可靠
```

### 3.2 FP8、FP16、BF16 对比

| 类型 | 每参数字节 | 优点 | 缺点 | 当前建议 |
|---|---:|---|---|---|
| FP8 | 1 byte | 显存占用低，可部署更大模型，吞吐更好 | 精度低于 BF16/FP16 | 当前首选 |
| FP16 | 2 bytes | 精度较高，生态成熟 | 指数范围较小，容易溢出/下溢 | 不建议当前单机部署 GLM-5.2 全量 |
| BF16 | 2 bytes | 指数范围接近 FP32，LLM 训练/推理更稳定 | 显存占用大 | 理论质量好，但当前成本高 |

FP16 与 BF16 的核心差异：

```text
FP16：尾数更多，指数范围小
BF16：尾数少一些，指数范围大，更稳定
```

对 LLM 来说，BF16 通常比 FP16 更稳。

FP8 与 BF16/FP16 的效果差异：

```text
常规问答、摘要、工具调用：通常差异较小
复杂数学、精细代码、长上下文推理：可能出现差异
最终应以业务测试集评估，而不是只看理论
```

当前部署建议：

```text
生产优先 GLM-5.2-FP8
除非质量评测明显不满足，再考虑 BF16/FP16 或更复杂的多机方案
```

## 4. 模型下载过程

### 4.1 模型目录

h20-02 模型最终下载目录：

```text
/data/models/GLM-5.2-FP8-ms
```

模型目录大小约：

```text
704G
```

模型 shard 数：

```text
141 个 safetensors shard
```

完整性检查结果：

```text
present=141
partial=0
complete=141/141
safetensors_count=141
zero_safetensors=0
aria2_count=0
```

说明：

```text
141 个模型分片均已下载完成
没有 .aria2 临时文件
没有 0 字节 safetensors
```

### 4.2 关键元数据文件

模型目录中的重要文件：

| 文件 | 观测大小 | 作用 |
|---|---:|---|
| config.json | 29K | 模型结构配置，包含层数、hidden size、MoE、上下文长度等 |
| generation_config.json | 194B | 默认生成参数，如 temperature、top_p、eos 等 |
| model.safetensors.index.json | 11M | 权重分片索引，记录每个参数在哪个 safetensors 文件里 |
| tokenizer.json | 20M，约 20,217,442 bytes | tokenizer 主体词表和分词规则 |
| tokenizer_config.json | 761B | tokenizer 额外配置，如特殊 token、chat template 等 |

解释：

```text
config.json 决定模型如何被加载
model.safetensors.index.json 决定 141 个 shard 如何映射到参数
safetensors 文件保存实际权重
tokenizer.json 决定文本如何切 token
generation_config.json 是默认生成策略，不是强约束
```

### 4.3 下载机制

本次最终采用 ModelScope 源和 aria2 并发下载，而不是单线程串行下载。

核心脚本：

```text
/data/scripts/download_glm52_remaining_aria2_parallel.sh
```

aria2 input 文件：

```text
/data/models/glm52-aria2-input.txt
```

下载日志：

```text
/data/models/glm52-aria2-parallel-download.log
```

监控日志：

```text
/data/models/glm52-download-monitor.log
```

最终稳定参数：

```bash
aria2c -c \
  -j32 \
  -x4 \
  -s4 \
  -k1M \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=true \
  --retry-wait=10 \
  --max-tries=0 \
  --timeout=60 \
  --connect-timeout=30 \
  --summary-interval=15
```

参数解释：

| 参数 | 含义 |
|---|---|
| `-c` | 断点续传 |
| `-j32` | 最多 32 个任务并发下载 |
| `-x4` | 单文件最多 4 个连接 |
| `-s4` | 单文件分 4 段下载 |
| `-k1M` | 分片大小 1MiB |
| `--file-allocation=none` | 不预分配文件空间，避免大文件预分配慢 |
| `--auto-file-renaming=false` | 不自动重命名，避免产生重复文件 |
| `--allow-overwrite=true` | 允许覆盖目标文件，便于恢复 |
| `--retry-wait=10` | 失败后 10 秒重试 |
| `--max-tries=0` | 无限重试 |
| `--timeout=60` | 传输超时 |
| `--connect-timeout=30` | 连接超时 |
| `--summary-interval=15` | 每 15 秒输出一次摘要 |

### 4.4 aria2 日志解读

典型日志：

```text
[DL:8.4MiB][#97ffe1 4.6GiB/4.9GiB(93%)][#338b36 2.4GiB/4.9GiB(48%)](+27)
[NOTICE] CUID#140 - Redirecting to https://cdn-lfs-cn-1.modelscope.cn/...
```

含义：

| 片段 | 含义 |
|---|---|
| `DL:8.4MiB` | 当前总下载速度约 8.4 MiB/s |
| `#97ffe1` | aria2 内部任务 ID |
| `4.6GiB/4.9GiB(93%)` | 某个分片下载进度 |
| `(+27)` | 还有 27 个任务未在当前摘要中展开 |
| `Redirecting to cdn-lfs-cn-1.modelscope.cn` | ModelScope 正常重定向到 CDN 下载地址 |

结论：

```text
Redirecting 是正常行为，不代表错误
看到多个 #task 同时推进，说明是并发下载
DL 速度是总下载速度，不是单文件速度
```

### 4.5 中断后的影响

aria2 使用 `.aria2` 临时状态文件保存断点。

中断后：

```text
已完成文件保持不变
未完成文件保留 .aria2 状态
重新执行同一下载任务会继续断点续传
如果 URL auth_key 过期，需要重新生成下载 URL
```

风险点：

```text
不要随意删除 .aria2 文件，否则可能丢失断点信息
不要开启自动重命名，否则可能出现重复文件
下载完成后 .aria2 文件应消失
```

### 4.6 查看下载进度的命令

在 h20-02 上查看：

```bash
tail -f /data/models/glm52-aria2-parallel-download.log
```

如果想看模型目录增长：

```bash
watch -n 10 'du -sh /data/models/GLM-5.2-FP8-ms; find /data/models/GLM-5.2-FP8-ms -name "*.aria2" | wc -l'
```

完整性检查：

```bash
MODEL_DIR=/data/models/GLM-5.2-FP8-ms
present=$(find "$MODEL_DIR" -maxdepth 1 -type f -name "model-*.safetensors" | wc -l)
partial=$(find "$MODEL_DIR" -maxdepth 1 -name "*.aria2" | wc -l)
echo "present=$present partial=$partial complete=$((present-partial))/141 size=$(du -sh "$MODEL_DIR" | cut -f1)"
```

## 5. vLLM 运行环境

### 5.1 Python 虚拟环境

虚拟环境路径：

```text
/data/venvs/glm52
```

已观察到的组件版本：

| 组件 | 版本 |
|---|---|
| vLLM | 0.23.0 |
| torch | 2.11.0+cu130 |
| transformers | 5.12.1 |

说明：

```text
版本会随环境变化而变化，正式文档中应记录 pip freeze 或 requirements lock
```

### 5.2 当前启动命令

当前使用过的 vLLM 启动命令：

```bash
/data/venvs/glm52/bin/python /data/venvs/glm52/bin/vllm serve /data/models/GLM-5.2-FP8-ms \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --served-model-name glm-5.2-fp8 \
  --trust-remote-code
```

核心参数解释：

| 参数 | 含义 |
|---|---|
| `--host 0.0.0.0` | 监听所有网卡 |
| `--port 8000` | OpenAI-compatible API 端口 |
| `--tensor-parallel-size 8` | 单模型实例切到 8 张 GPU |
| `--dtype auto` | 权重 dtype 自动根据模型配置处理 |
| `--kv-cache-dtype fp8` | KV cache 使用 FP8，降低显存占用 |
| `--max-model-len 32768` | 服务实例最大上下文长度 32K |
| `--gpu-memory-utilization 0.90` | vLLM 可使用约 90% GPU 显存做规划 |
| `--served-model-name glm-5.2-fp8` | API 调用中的 model 名称 |
| `--trust-remote-code` | 允许加载模型仓库中的自定义代码 |

### 5.3 首次启动慢的原因

首次启动慢是正常现象，可能包括：

```text
加载 700GB 级权重
初始化 8 卡 tensor parallel
初始化 NCCL 通信
编译或加载 CUDA kernel
ptxas 编译
预分配或规划 KV cache
```

曾观察到 `ptxas` 进程，说明存在 CUDA kernel 编译。部分编译缓存后，后续启动可能变快，但不能保证每次都快：

```text
驱动变更
CUDA 版本变更
vLLM 版本变更
模型参数变更
缓存清理
不同 kernel path
```

都可能导致重新编译或慢启动。

## 6. OpenAI-compatible API 调用

### 6.1 模型列表

```bash
curl http://127.0.0.1:8000/v1/models
```

### 6.2 正确的 chat/completions 请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-fp8",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

错误示例：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

这个请求会 Bad Request，因为缺少：

```text
model
messages
```

### 6.3 非流式请求延迟

曾测试：

```bash
curl -s -o /tmp/glm_resp.json \
  -w 'connect=%{time_connect}\nstarttransfer=%{time_starttransfer}\ntotal=%{time_total}\n' \
  http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-fp8",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

结果：

```text
connect=0.000172
starttransfer=4.408103
total=4.408253
```

解释：

```text
connect 很低，说明本机网络连接不是问题
非流式请求的 starttransfer 接近 total，表示服务等完整答案生成后才返回
4.4s 不能直接等于模型整体性能，只是单请求端到端耗时
```

### 6.4 流式请求

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-fp8",
    "messages": [
      {"role": "user", "content": "用一句话介绍你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 128,
    "stream": true
  }'
```

流式请求更适合观察：

```text
TTFT，首 token 延迟
inter-token latency，token 间隔
用户主观体感
```

### 6.5 查看 usage 和 token 数

```bash
python3 - <<'PY'
import json
r=json.load(open('/tmp/glm_resp.json'))
print(r.get('usage'))
print(r['choices'][0]['message']['content'][:500])
PY
```

TPS 计算：

```text
单请求输出 TPS = completion_tokens / total_time
```

示例：

```text
如果 completion_tokens=100，total=4.4s，则约 22.7 tokens/s
如果 completion_tokens=200，total=4.4s，则约 45.5 tokens/s
```

## 7. 日志与进程排查

### 7.1 查看 vLLM 进程

```bash
ps -eo pid,ppid,etime,cmd | egrep "vllm|VLLM|python.*serve" | grep -v grep
```

### 7.2 查看端口

API 端口：

```bash
ss -ntlp | grep ':8000'
```

内部 worker 端口：

```bash
ss -ntlp | grep VLLM
```

如果看到很多 `VLLM::Worker_TP` 相关端口，这是正常现象。TP=8 会启动多个 worker，vLLM/PyTorch/NCCL 需要内部通信端口。

### 7.3 查看 stdout/stderr 去向

曾观察到：

```text
/proc/420489/fd/1 -> /dev/pts/1
/proc/420489/fd/2 -> /dev/pts/1
```

这说明该进程的标准输出和错误输出都写到了启动它的终端，不在日志文件里。

如果终端关闭，日志不可追踪，进程也可能受影响。因此生产不应前台启动。

### 7.4 建议日志方式

临时后台方式：

```bash
mkdir -p /data/logs
nohup /data/venvs/glm52/bin/vllm serve /data/models/GLM-5.2-FP8-ms \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --served-model-name glm-5.2-fp8 \
  --trust-remote-code \
  > /data/logs/glm52-vllm.log 2>&1 &
```

查看日志：

```bash
tail -f /data/logs/glm52-vllm.log
```

正式建议使用 systemd。

## 8. 上下文长度与 max-model-len

### 8.1 模型理论上下文与服务上下文

GLM-5.2 理论支持长上下文，最长可到 1M tokens。模型配置中通常会包含类似：

```json
"max_position_embeddings": 1048576
```

如果 vLLM 启动时不指定：

```bash
--max-model-len
```

则 vLLM 会读取模型配置，自动推导模型最大上下文长度。对 GLM-5.2 这可能意味着按 1M 规划。

但生产服务通常不应直接开放 1M。

当前设置：

```bash
--max-model-len 32768
```

含义：

```text
模型能力：理论可到 1M
当前服务实例开放：最多 32K
```

这不改变模型权重，只限制该 vLLM 实例接受和规划的最大序列长度。

### 8.2 不同业务的建议上下文长度

| 场景 | 推荐 max-model-len |
|---|---:|
| 普通在线问答 | 16K-32K |
| 企业知识库/RAG | 32K-64K |
| Agent / 工具调用 | 64K-128K |
| Coding agent 多文件任务 | 128K-256K |
| 长文档/仓库级任务 | 256K-1M |
| 高并发在线 API | 16K-32K |

当前建议：

```text
通用生产实例：32K 或 64K
Agent 增强实例：64K 或 128K
长上下文实验实例：256K 起逐步测试到 1M
```

不建议：

```text
把唯一正式服务默认设置成 1M
```

原因：

```text
首 token 延迟会上升
KV cache 压力上升
并发能力下降
异常请求风险变大
p95/p99 延迟可能恶化
```

## 9. KV cache 估算与配置

### 9.1 KV cache 不是固定配置值

vLLM 中通常不是直接配置“KV cache 多大”，而是通过以下参数间接决定：

```bash
--max-model-len
--gpu-memory-utilization
--kv-cache-dtype
```

KV cache 的实际容量由 vLLM 根据：

```text
模型权重占用
可用显存
KV dtype
block size
max_model_len
runtime buffer
```

自动规划。

### 9.2 KV cache 与 token 的关系

核心公式：

```text
KV cache 总需求 = 所有正在运行请求的 token 总量 × 每 token KV 显存
```

更具体：

```text
总 KV token = Σ(每个请求的输入 token + 已生成 token)
```

因此高并发要按“总在途 token”计算，不是只看请求数。

例如：

```text
1 个请求 x 1M tokens = 1M KV tokens
10 个请求 x 100K tokens = 1M KV tokens
100 个请求 x 10K tokens = 1M KV tokens
```

这三种情况下 KV cache 总量是同一量级。

### 9.3 KV cache 与上下文长度比例

KV cache 与上下文长度基本是线性关系：

```text
KV cache ∝ token 数
```

相对关系：

| 上下文长度 | 相对 32K 的 KV 占用 |
|---:|---:|
| 32K | 1x |
| 64K | 2x |
| 128K | 4x |
| 256K | 8x |
| 512K | 16x |
| 1M | 32x |

### 9.4 GLM-5.2 1M KV cache 量级估算

GLM-5.2 不是普通 MHA 全量 KV 结构，其长上下文依赖压缩 KV / MLA 类似机制。估算中用到的关键参数：

```text
num_hidden_layers = 78
kv_lora_rank = 512
qk_rope_head_dim = 64
```

粗略公式：

```text
KV cache/token ≈ layers × (kv_lora_rank + qk_rope_head_dim) × bytes
```

代入：

```text
78 × (512 + 64) = 44,928 elements/token
```

1M tokens：

```text
FP8: 44,928 × 1 byte × 1,000,000 ≈ 44.9GB
BF16/FP16: 44,928 × 2 bytes × 1,000,000 ≈ 89.9GB
```

工程预算：

```text
FP8 KV cache：约 55GB-70GB 总预算
BF16/FP16 KV cache：约 110GB-140GB 总预算
```

注意：

```text
这是整个 TP=8 模型实例的总 KV 量级，不是每张卡 70GB
平均到 8 张卡，1M FP8 KV 理论约 5.6GB/卡，加上运行余量可按 7GB-10GB/卡理解
```

### 9.5 高并发示例

如果要支持：

```text
并发 16
平均每请求 32K tokens
```

则：

```text
总在途 token = 16 × 32K = 512K tokens
```

按 1M FP8 KV 约 45GB 估算：

```text
512K ≈ 1M 的一半
KV cache 总量约 22.5GB
```

如果要支持：

```text
并发 16
每请求 1M tokens
```

则：

```text
总在途 token = 16M tokens
KV cache 总量 ≈ 45GB × 16 = 720GB
平均每卡约 90GB
```

这对当前每卡已经有 90 多 GB 权重占用的场景基本不现实。

### 9.6 推荐 KV 相关参数

通用生产初始值：

```bash
--max-model-len 32768 \
--gpu-memory-utilization 0.90 \
--kv-cache-dtype fp8
```

如果偏稳定：

```bash
--max-model-len 32768 \
--gpu-memory-utilization 0.85 \
--kv-cache-dtype fp8
```

如果偏 Agent：

```bash
--max-model-len 65536 \
--gpu-memory-utilization 0.90 \
--kv-cache-dtype fp8
```

如果偏长上下文实验：

```bash
--max-model-len 131072 或更高 \
--gpu-memory-utilization 0.90-0.95 \
--kv-cache-dtype fp8
```

建议逐档测试：

```text
32K -> 64K -> 128K -> 256K -> 512K -> 1M
```

每档观察：

```text
是否成功启动
GPU KV cache size
Maximum concurrency
TTFT
p95 latency
OOM/timeout
GPU 利用率
```

## 10. TP、EP 与并行部署策略

### 10.1 TP：Tensor Parallel

当前使用：

```bash
--tensor-parallel-size 8
```

含义：

```text
一个模型实例切到 8 张 GPU 上
每个请求都会使用全部 8 张 GPU
```

TP 常见切分对象：

```text
Attention projection
Linear 权重矩阵
MLP projection
```

通信模式：

```text
all-reduce
all-gather
reduce-scatter
broadcast
```

优点：

```text
能加载单卡放不下的大模型
单机 8 卡 NVSwitch 场景非常适合
```

缺点：

```text
每个请求都占用所有 TP GPU
强依赖 GPU-GPU 通信
不等于 8 个独立副本
```

### 10.2 EP：Expert Parallel

EP 主要用于 MoE 模型。

含义：

```text
不同 expert 分布在不同 GPU 上
token 根据 gating 路由到部分 expert
```

通信模式：

```text
token dispatch
all-to-all
combine
```

优点：

```text
更适合 MoE 大模型
可能提升吞吐和 expert 权重分布效率
```

缺点：

```text
实现复杂
依赖框架支持
all-to-all 对通信拓扑敏感
需要验证 GLM-5.2 与当前 vLLM 版本兼容性
```

### 10.3 TP + EP

TP+EP 通常是：

```text
Dense/Attention 部分使用 TP
MoE expert 部分使用 EP
```

对 GLM-5.2 这种 MoE 模型，理论上 TP+EP 可能更合理。但当前第一阶段使用纯 TP=8 更稳妥。

后续可以探索：

```text
vLLM 是否稳定支持 GLM-5.2 expert parallel
SGLang 是否对 GLM-5.2 MoE 有更好的 EP 支持
NCCL all-to-all 性能
DeepEP 或相关通信优化是否适用
```

### 10.4 双机部署建议

不建议：

```text
h20-01 + h20-02 跨机器 TP=16
```

除非同时满足：

```text
两机之间有高性能 IB/RDMA
NCCL 跨机通信稳定
框架支持跨节点推理并行
压测证明收益大于复杂度
```

推荐：

```text
h20-01：GLM-5.2-FP8，TP=8，独立实例
h20-02：GLM-5.2-FP8，TP=8，独立实例
Gateway：负载均衡、鉴权、限流、计费、熔断、观测
```

这样更适合生产：

```text
高可用更简单
容量线性扩展更清晰
故障隔离更好
运维复杂度更低
```

## 11. 并发、QPS 与压测方法

### 11.1 不能只看 QPS

大模型服务中，QPS 不能脱离 token 长度单独讨论。

真实压力取决于：

```text
并发请求数 × 输入 token × 输出 token
```

或者：

```text
总在途 token = Σ(input_tokens + generated_tokens)
```

同样并发 16：

| 并发 | 输入 | 输出 | 压力 |
|---:|---:|---:|---|
| 16 | 512 | 128 | 轻 |
| 16 | 8K | 512 | 中 |
| 16 | 32K | 1K | 重 |
| 16 | 128K | 2K | 极重 |

### 11.2 已知单请求结果

当前已知：

```text
非流式单请求 total = 4.408s
```

这只能推导单客户端串行 QPS：

```text
QPS ≈ 1 / 4.408 ≈ 0.23 req/s
```

这不是整体模型最大 QPS。

### 11.3 关键指标

| 指标 | 含义 |
|---|---|
| QPS | 每秒完成请求数 |
| 并发度 | 同时在处理/排队的请求数 |
| TTFT | time to first token，首 token 延迟 |
| ITL | inter-token latency，token 间隔 |
| output tokens/s | 单请求输出速度 |
| aggregate tokens/s | 服务整体输出吞吐 |
| p50/p95/p99 latency | 延迟分位数 |
| failure rate | 失败率 |
| GPU util | GPU 利用率 |
| KV cache usage | KV 使用情况 |

### 11.4 Workload 分层

建议定义三类测试负载。

普通问答：

```text
512 in / 128 out
2K in / 256 out
4K in / 512 out
```

Agent / 工具调用：

```text
4K in / 512 out
8K in / 1K out
16K in / 1K out
32K in / 2K out
```

长文档 / 仓库级任务：

```text
32K in / 1K out
64K in / 2K out
128K in / 2K out
256K in / 4K out
```

### 11.5 并发阶梯

每类 workload 逐档测试：

```text
并发 1
并发 2
并发 4
并发 8
并发 16
并发 32
```

判断拐点：

```text
QPS 不再明显增长
aggregate tokens/s 不再增长
p95 latency 突然上升
TTFT 明显拉长
失败率上升
请求 timeout
GPU 已接近满载
```

### 11.6 推荐压测工具

优先级：

| 工具 | 适用场景 |
|---|---|
| vLLM benchmark_serving.py | 当前 vLLM OpenAI-compatible API 最直接 |
| NVIDIA GenAI-Perf | 适合正式报告，指标更完整 |
| 自写 aiohttp 脚本 | 快速探索、临时验证 |
| Triton perf_analyzer | 更适合 Triton/TensorRT-LLM，不是当前首选 |

压测报告必须记录：

```text
模型版本
vLLM 版本
GPU 型号和数量
max_model_len
kv_cache_dtype
gpu_memory_utilization
输入 token 分布
输出 token 分布
并发
QPS
TTFT
p50/p95/p99 latency
aggregate tokens/s
GPU util
失败率
```

## 12. Agent 与上下文规划

### 12.1 32K 是否足够

32K 对普通 agent 是够用起点，但对复杂 coding agent、长文档 agent、仓库级 agent 偏小。

| Agent 类型 | 典型上下文需求 | 32K 是否够 |
|---|---:|---|
| 普通聊天 agent | 4K-16K | 够 |
| 工具调用 agent | 8K-32K | 基本够 |
| RAG/知识库 agent | 16K-64K | 看检索片段数量 |
| Coding agent 单文件任务 | 16K-64K | 勉强够 |
| Coding agent 多文件任务 | 64K-200K | 偏小 |
| 仓库级 agent | 200K-1M | 不够 |
| 长文档分析 agent | 128K-1M | 不够 |

### 12.2 Agent 消耗上下文的来源

```text
system prompt
工具定义
用户需求
历史对话
检索文档
代码片段
执行日志
错误堆栈
中间计划
模型输出预算
```

如果 agent 做了良好的上下文管理：

```text
按需检索
历史摘要
日志截断
状态外置
只保留相关文件片段
```

则 32K 能跑很多任务。

如果 agent 依赖把所有轨迹塞进 prompt，32K 会很紧。

### 12.3 推荐服务分层

```text
在线通用 agent：32K
高级 coding / RAG agent：64K 或 128K
长上下文代码仓库 / 长文档 agent：256K+
1M：专用长任务实例，不作为默认在线服务
```

## 13. 生产化部署方案

### 13.1 部署方式变更

本次部署方式已从命令行手工启动调整为 systemd service 启动。

旧方式：

```text
在 shell 中直接执行 vllm serve
stdout/stderr 指向当前终端
进程生命周期依赖启动终端或 nohup 管理
```

新方式：

```text
systemd 管理 glm52-vllm.service
固定日志路径 /data/logs/glm52-vllm.log
支持 systemctl start/stop/restart/status
支持失败自动重启
服务随系统启动自动拉起
```

命令行手工启动存在问题：

```text
终端关闭可能影响进程
日志没有固定文件
无法自动重启
无法统一健康检查
无法被运维系统管理
```

正式运行应改为 systemd 或容器编排。

当前阶段采用：

```text
systemd service
```

### 13.2 systemd 服务定义

服务文件：

```text
/etc/systemd/system/glm52-vllm.service
```

服务内容：

```ini
[Unit]
Description=GLM-5.2 FP8 vLLM Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=server
WorkingDirectory=/data/models/GLM-5.2-FP8-ms
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
Environment=PATH=/data/venvs/glm52/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=/home/server
ExecStart=/data/venvs/glm52/bin/vllm serve /data/models/GLM-5.2-FP8-ms \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --served-model-name glm-5.2-fp8 \
  --trust-remote-code
Restart=on-failure
RestartSec=10
StandardOutput=append:/data/logs/glm52-vllm.log
StandardError=append:/data/logs/glm52-vllm.log
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

关键注意事项：

```text
systemd 默认 PATH 不会自动包含 Python venv 的 bin 目录。
vLLM/FlashInfer 初始化时可能调用 ninja。
如果不设置 PATH，即使 /data/venvs/glm52/bin/ninja 已安装，也会报 No such file or directory: 'ninja'。
```

已验证的修复方式：

```ini
Environment=PATH=/data/venvs/glm52/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=/home/server
```

### 13.3 标准部署步骤

部署顺序：

```text
1. 修改部署方案文档
2. 停止旧的命令行 vLLM 进程
3. 写入 /etc/systemd/system/glm52-vllm.service
4. systemctl daemon-reload
5. enable 服务
6. start 服务
7. 通过 status、端口和 /v1/models 做最小检查
```

标准启停命令：

```bash
sudo systemctl daemon-reload
sudo systemctl enable glm52-vllm
sudo systemctl start glm52-vllm
sudo systemctl status glm52-vllm
sudo systemctl restart glm52-vllm
sudo systemctl stop glm52-vllm
```

日志：

```bash
tail -f /data/logs/glm52-vllm.log
journalctl -u glm52-vllm -f
```

注意：

```text
systemd 配置上线前应先在非生产窗口验证
如果 User=server，则要确认 server 用户有模型目录、venv、日志目录权限
```

### 13.4 回滚到命令行方式

如果 systemd 启动失败，可以回滚为临时命令行启动：

```bash
sudo systemctl stop glm52-vllm
mkdir -p /data/logs
nohup /data/venvs/glm52/bin/vllm serve /data/models/GLM-5.2-FP8-ms \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --served-model-name glm-5.2-fp8 \
  --trust-remote-code \
  > /data/logs/glm52-vllm.log 2>&1 &
```

### 13.5 Gateway 接入建议

两台 H20 更适合做两个独立实例：

```text
h20-01: http://h20-01:8000/v1
h20-02: http://h20-02:8000/v1
Gateway: 对外统一 API
```

Gateway 应负责：

```text
鉴权
计费
限流
配额
请求体大小限制
max_tokens 限制
模型路由
健康检查
熔断
重试
审计日志
用量统计
```

不要让外部用户直接访问 vLLM 端口。

## 14. 监控与运维检查

### 14.1 GPU 状态

```bash
nvidia-smi
```

更详细：

```bash
nvidia-smi dmon
```

关注：

```text
显存占用
GPU util
功耗
温度
ECC 错误
进程列表
```

### 14.2 拓扑与链路

GPU 拓扑：

```bash
nvidia-smi topo -m
```

NVLink 状态：

```bash
nvidia-smi nvlink --status
```

PCIe 状态：

```bash
nvidia-smi -q -d PCI | egrep "GPU|Current Link|Max Link"
```

实际 GPU-GPU 带宽：

```bash
find /usr/local/cuda* -name p2pBandwidthLatencyTest 2>/dev/null
```

找到后执行。

### 14.3 API 健康检查

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

简单推理：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-fp8",
    "messages": [{"role": "user", "content": "ping"}],
    "temperature": 0,
    "max_tokens": 8
  }'
```

### 14.4 常见问题

#### curl 连接失败

可能原因：

```text
vLLM 还在加载模型
CUDA kernel 还在编译
API server 尚未监听 8000
进程启动失败
端口被占用
```

检查：

```bash
ps -eo pid,ppid,etime,cmd | egrep "vllm|VLLM|python.*serve" | grep -v grep
ss -ntlp | grep ':8000'
tail -f /data/logs/glm52-vllm.log
```

#### Bad Request

通常是请求体缺少：

```text
model
messages
```

#### 端口很多

TP=8 下看到很多本地端口通常是正常现象，属于 vLLM worker / PyTorch / NCCL 内部通信。

#### 非流式请求感觉慢

非流式请求要等完整输出生成后才返回。应使用 stream 模式观察真实首 token 体感。

#### 显存没有用满是否浪费

不是。剩余显存用于：

```text
KV cache
并发请求
CUDA workspace
通信 buffer
vLLM runtime buffer
```

不要把显存压到 100%。生产上保留余量更重要。

## 15. 当前推荐部署形态

### 15.1 第一阶段：稳定在线服务

每台机器一个实例：

```bash
vllm serve /data/models/GLM-5.2-FP8-ms \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --served-model-name glm-5.2-fp8 \
  --trust-remote-code
```

适合：

```text
普通问答
基础 agent
RAG 短片段
低风险上线
```

### 15.2 第二阶段：Agent 增强实例

可考虑：

```bash
--max-model-len 65536
```

或：

```bash
--max-model-len 131072
```

适合：

```text
复杂工具调用
多轮 agent
代码任务
较长 RAG
```

### 15.3 第三阶段：长上下文实验实例

逐档测试：

```text
256K
512K
1M
```

不建议默认对外开放，适合：

```text
长文档分析
仓库级 agent
能力展示
专用低并发任务
```

### 15.4 双机最终建议

推荐：

```text
h20-01：TP=8，独立 GLM-5.2-FP8 服务
h20-02：TP=8，独立 GLM-5.2-FP8 服务
Chancloud Gateway：统一入口、负载均衡、计费、限流、审计
```

不推荐默认：

```text
两台机器组成一个跨节点 TP=16 服务
```

## 16. 后续待办

### 16.1 服务化

```text
创建 systemd 服务
固定日志路径
设置自动重启
设置健康检查
限制外部直接访问 vLLM
```

### 16.2 压测

```text
使用 vLLM benchmark_serving.py 或 NVIDIA GenAI-Perf
按短问答、Agent、长上下文三类 workload 测试
输出 QPS、TTFT、p95、tokens/s、失败率
```

### 16.3 观测

```text
接入 Prometheus / Grafana
采集 GPU util、显存、请求数、tokens、延迟、失败率
Gateway 记录 tenant/model/user 维度用量
```

### 16.4 安全与运维

```text
避免频繁 SSH 导致端口限制
不要把密码写入脚本或日志
禁止外部直接访问 8000 推理端口
通过 Gateway 做认证和计费
对 max_tokens、输入长度、并发数做限制
```

### 16.5 模型质量评估

```text
构建业务测试集
比较 FP8 与可能的 BF16/FP16 或其他模型
覆盖问答、代码、推理、长上下文、工具调用
输出质量和性能权衡报告
```

## 17. 快速命令索引

查看模型目录：

```bash
cd /data/models/GLM-5.2-FP8-ms
ls -lh
ls -lh config.json generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json
ls model-*.safetensors | wc -l
find . -name "*.aria2"
du -sh .
```

查看 vLLM：

```bash
ps -eo pid,ppid,etime,cmd | egrep "vllm|VLLM|python.*serve" | grep -v grep
ss -ntlp | grep ':8000'
```

查看 GPU：

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink --status
nvidia-smi -q -d PCI | egrep "GPU|Current Link|Max Link"
```

测试 API：

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl -s -o /tmp/glm_resp.json \
  -w 'connect=%{time_connect}\nstarttransfer=%{time_starttransfer}\ntotal=%{time_total}\n' \
  http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-fp8",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

查看 usage：

```bash
python3 - <<'PY'
import json
r=json.load(open('/tmp/glm_resp.json'))
print(r.get('usage'))
print(r['choices'][0]['message']['content'][:500])
PY
```

流式输出：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-fp8",
    "messages": [
      {"role": "user", "content": "用一句话介绍你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 128,
    "stream": true
  }'
```

## 18. 核心结论

1. GLM-5.2 不是 5.2B 小模型，而是 743B 级 MoE 大模型，当前选择 FP8 是合理优先项。
2. h20-02 已完成 GLM-5.2-FP8 下载，模型目录为 `/data/models/GLM-5.2-FP8-ms`，141 个 shard 完整。
3. 当前机器 GPU 拓扑为 8 卡全 `NV18`，非常适合单机 `TP=8`。
4. 不建议两台 H20 跨机器组成 TP=16，除非确认有高性能 IB/RDMA 并完成跨机 NCCL 压测。
5. 当前最合理部署是两台机器各跑一个独立 `TP=8` 实例，前面由 Gateway 做负载均衡、计费和限流。
6. `max-model-len=32K` 是稳定在线服务的合理起点，Agent 场景建议准备 64K 或 128K 实例。
7. 1M context 是模型能力上限，不应作为默认在线服务配置，适合长上下文专用实例逐档压测。
8. KV cache 与总在途 token 数近似线性相关，高并发能力必须结合 input/output token 分布评估。
9. 单次 curl 的 4.4s 只能说明单请求非流式延迟，不能代表整体 QPS，需要标准压测矩阵。
10. 下一步重点是 systemd 服务化、标准压测、Gateway 接入和监控指标建设。
