# nanoDiffusionLab

**A compact laboratory for autoregressive and diffusion language models.**<br>
**一个用于公平比较自回归与扩散语言模型的精简实验平台。**

[中文详细说明](README.zh-CN.md) · [Architecture / 架构](docs/architecture.md) ·
[Experiment report / 实验报告](reports/tinystories_106m.md)

![Autoregressive decoding reveals tokens from left to right; masked diffusion reveals multiple
positions in parallel.](docs/assets/decoding_comparison.gif)

## Why this project? / 为什么做这个项目？

nanoDiffusionLab borrows nanoGPT's small-and-readable philosophy while keeping its implementation
independent. A single Transformer implementation supports two runnable objectives, making it
possible to hold the backbone, data, context length, and input-token budget fixed.

nanoDiffusionLab 借鉴 nanoGPT 代码紧凑、易读的理念，但保持独立实现。同一套 Transformer
目前支持两种训练目标，因此可以固定主干网络、数据、上下文长度与输入 token 预算，进行尽可能
公平的比较。

| Objective / 目标 | Attention / 注意力 | Training signal / 训练信号 | Decoding / 解码 |
|---|---|---|---|
| Autoregressive (AR) / 自回归 | Causal / 因果 | Predict the next token / 预测下一 token | Sequential, left to right / 从左到右串行生成 |
| Masked diffusion (MDLM) / 掩码扩散 | Bidirectional / 双向 | Restore randomly masked tokens / 恢复随机掩码 token | Iterative parallel unmasking / 迭代并行去掩码 |

Block diffusion / 块扩散是下一阶段目标，目前尚未实现。

## Completed 106M experiment / 已完成的 106M 实验

Two independent seeds were trained on the pinned TinyStories dataset with GPT-2 BPE. Each
objective received **2.000B input tokens** on **4× NVIDIA A40 48 GB** GPUs using PyTorch DDP.

我们使用固定 revision 的 TinyStories 与 GPT-2 BPE 完成了两个独立随机种子实验。每个目标均在
**4× NVIDIA A40 48 GB** 上通过 PyTorch DDP 训练 **2.000B 输入 token**。

![Final metrics for two independent seeds](docs/assets/replication_summary.png)

| Result / 结果 | AR seed 1337 | AR seed 2027 | MDLM seed 1337 | MDLM seed 2027 |
|---|---:|---:|---:|---:|
| Parameters / 参数量 | 106.28M | 106.28M | 106.61M | 106.61M |
| Validation loss / 验证损失 | 1.2368 | 1.2409 | 1.9635 | 1.9705 |
| Perplexity / 困惑度 | 3.4446 | 3.4586 | — | — |
| Masked accuracy / 掩码准确率 | — | — | 59.08% | 59.02% |
| Training throughput / 训练吞吐 | 242.9K tok/s | 244.8K tok/s | 287.1K tok/s | 289.4K tok/s |

> AR cross-entropy and MDLM denoising loss are objective-specific. They must not be interpreted as
> the same likelihood metric. / AR 交叉熵与 MDLM 去噪损失的定义不同，不能作为同一种似然指标
> 直接比较。

![Validation curves across two random seeds](docs/assets/training_curves.png)

The near-overlapping curves and close final metrics show that both runs are reproducible across the
two tested seeds. The included fixed-seed samples are currently more coherent for AR; MDLM sampling
quality remains an active research target.

两组曲线几乎重合，最终指标也十分接近，说明当前配置在两个测试 seed 上具有良好复现性。从固定
seed 样例看，当前 AR 的故事连贯性更好；提升 MDLM 采样质量仍是后续重点。

Detailed outputs / 详细结果：

- [seed 1337 comparison / 对比报告](reports/tinystories_106m.md)
- [seed 2027 comparison / 对比报告](reports/tinystories_106m_seed2027.md)
- [implementation snapshot / 实现与早期快照](reports/initial_run_report.html)

## What works / 当前能力

- One Transformer for AR and masked diffusion / 同一 Transformer 支持 AR 与掩码扩散
- Character smoke tests and memory-mapped token shards / 字符级测试与内存映射 token 分片
- Causal and bidirectional SDPA attention / 因果与双向 SDPA 注意力
- Masked-only output projection for lower MDLM memory / 仅投影掩码位置以降低 MDLM 显存
- Time/noise-level conditioning and parallel unmasking / 时间条件与并行去掩码
- CPU, single-GPU, and DDP training / CPU、单卡与 DDP 训练
- BF16/FP16, gradient accumulation, checkpointing, and resume / 混合精度、梯度累积与断点恢复
- Atomic checkpoints, JSONL metrics, metadata, and source snapshots / 原子 checkpoint、指标与源码快照
- Fixed-token-budget comparison and deterministic evaluation / 固定 token 预算与确定性评估

## Quick start / 快速开始

Requires Python 3.10+ and PyTorch 2.3+. / 需要 Python 3.10+ 与 PyTorch 2.3+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

mkdir -p data/tinyshakespeare
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o data/tinyshakespeare/input.txt

python train.py --config configs/shakespeare_char.py
python sample.py --checkpoint out/shakespeare-mdlm/best.pt --show-steps
```

Add `--max-iters 10` for a plumbing check; meaningful samples require a real training run.<br>
添加 `--max-iters 10` 可快速检查流程；有意义的生成样例需要完整训练。

Switch to AR without changing the shared backbone / 切换到 AR 且保持共享主干不变：

```bash
python train.py --config configs/shakespeare_char.py \
  --objective autoregressive --out-dir out/shakespeare-ar
```

## Reproduce the TinyStories experiment / 复现 TinyStories 实验

The paired models use 12 layers, width 576, 9 heads, a 1024-token context, and approximately 106M
parameters. / 成对模型使用 12 层、576 hidden size、9 个 attention heads、1024 token context，
参数量约 106M。

```bash
pip install -e ".[data,dev]"
python scripts/prepare_tinystories.py
bash scripts/run_tinystories_pair.sh
```

To launch one objective manually on four GPUs / 在四张 GPU 上手动启动单个目标：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/tinystories_106m.py \
  --objective autoregressive \
  --out-dir out/tinystories-106m-ar
```

Each run writes its resolved configuration, environment metadata, JSONL metrics, atomic resumable
checkpoints, milestone checkpoints, samples, and final summary. Use `--resume` to continue from
`last.pt`. / 每次运行都会保存解析后的配置、环境信息、JSONL 指标、可恢复 checkpoint、生成
样例与最终摘要；使用 `--resume` 从 `last.pt` 继续。

Regenerate the README figures from completed local runs / 从本地完整运行重新生成 README 图表：

```bash
pip install -e ".[viz]"
python scripts/render_readme_assets.py --refresh-data
```

## Hardware notes / 硬件说明

For 100M–350M models on four PCIe/PHB A40s without NVLink / 对于四张无 NVLink、PHB 拓扑的
A40：

- prefer DDP over tensor parallelism, FSDP, or ZeRO-3 at this scale / 当前规模优先 DDP；
- keep per-GPU micro-batches large enough to amortize all-reduce / 使用足够大的每卡 batch；
- use `no_sync()` during gradient accumulation / 梯度累积期间避免冗余通信；
- benchmark NCCL on the actual server before claiming scaling / 扩展效率必须以真实测量为准。

`configs/fineweb_350m.py` is a model target, not yet a production FineWeb data recipe. / 该配置目前
只是 350M 模型目标，尚未包含生产级 FineWeb 数据流程。

## Project map / 项目结构

```text
model.py                    shared Transformer and AR sampler / 共享主干与 AR 采样
diffusion.py                corruption, denoising loss, sampler / 掩码破坏、去噪与采样
train.py                    token-budget training, eval, DDP / 训练、评估与 DDP
data.py                     character and token-shard loaders / 字符与 token 分片加载
experiment.py               atomic artifacts and metadata / 原子产物与实验元数据
compare_runs.py             paired Markdown reports / 成对实验报告
sample.py                   checkpoint loading and generation / checkpoint 与生成
configs/                    runnable experiment configurations / 实验配置
scripts/                    data, training, and figure workflows / 数据、训练与绘图流程
tests/                      behavioral tests / 行为测试
```

## Scope / 范围

The codebase optimizes for clarity and a correct experimental skeleton. Exact MDLM likelihood
weighting, remasking, block diffusion, KV caching, and pretrained checkpoints are not yet included.

当前版本强调代码清晰与实验骨架正确性，尚未实现精确 MDLM likelihood weighting、remasking、
Block Diffusion、KV cache 与预训练 checkpoint。

## License / 许可证

MIT
