# nanoDiffusionLab

[English](README.md) | **简体中文**

一个从零构建、用于学习、训练和比较自回归语言模型与扩散语言模型的精简 PyTorch
实验平台。

项目借鉴 nanoGPT 代码紧凑、易读的理念，但实现保持独立。目前同一套 Transformer
支持两种可运行目标：

- **自回归（AR）**：因果注意力和 next-token prediction；
- **掩码扩散（Masked Diffusion）**：双向注意力、随机掩码破坏和迭代并行解码。

Block Diffusion 是下一阶段架构目标，目前仅有设计说明，尚未实现。

## 当前能力

- 字符级 smoke test 和 memory-mapped token shards 训练；
- 仅对掩码位置执行输出投影，降低扩散模型 logits 显存；
- 时间/噪声等级条件；
- 基于置信度的并行去掩码和保持 prompt 不变的 infill；
- 使用同一 Transformer blocks 的 AR baseline；
- CPU、单 GPU 和带梯度累积的 DDP 训练；
- BF16/FP16 autocast、gradient checkpointing、SDPA 和可选 `torch.compile`；
- 破坏过程、注意力模式、采样、分片数据和实验记录测试。

## 快速开始

需要 Python 3.10+ 和 PyTorch 2.3+。

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

可添加 `--max-iters 10` 做短流程检查；生成合理样本需要完整训练。

## 切换到 AR baseline

可以通过命令行覆盖目标，同时保持其余配置一致：

```bash
python train.py --config configs/shakespeare_char.py \
  --objective autoregressive --out-dir out/shakespeare-ar
```

公平比较时应固定模型主干、数据和输入 token 预算，并单独披露有效监督 token 数和实际
计算开销。

## 4×A40 TinyStories 106M 实验

首先使用 GPT-2 BPE 准备固定 revision 的 TinyStories：

```bash
pip install -e ".[data,dev]"
python scripts/prepare_tinystories.py
```

成对模型使用 12 层、576 hidden size、9 attention heads、1024 token context，参数量约
106M。两种目标使用相同的 2B 输入 token 预算：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/tinystories_106m.py --objective autoregressive \
  --out-dir out/tinystories-106m-ar

python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/tinystories_106m.py --objective masked_diffusion \
  --out-dir out/tinystories-106m-mdlm
```

每次运行都会保存解析后的配置、环境信息、源码快照、JSONL 指标、原子 checkpoint、
token 里程碑和最终摘要。使用 `--resume` 可从该运行的 `last.pt` 精确恢复。

完整的顺序实验、固定种子采样和 Markdown 对比报告可通过以下命令运行：

```bash
bash scripts/run_tinystories_pair.sh
```

## 实验报告

- [106M 实现与首轮 AR 运行报告](reports/initial_run_report.html) — 固定在 step 250 的阶段性
  快照，成对训练仍在进行。

## 4×A40 PCIe/PHB 说明

本项目在约 100M–350M 参数规模优先使用 DDP。对于没有 NVLink、GPU 间均为 PHB 的
服务器：

- 优先采用 DDP，而不是 tensor parallel、FSDP 或 ZeRO-3；
- 使用足够大的每卡 micro-batch 来摊薄 all-reduce 成本；
- 梯度累积期间使用 `no_sync()`，避免冗余通信；
- 扩展效率结论必须来自目标服务器上的真实测量。

`configs/fineweb_350m.py` 目前仍只是模型目标配置，还没有实现生产级 FineWeb 数据获取与
清洗流程。

## 项目结构

```text
model.py                       共享 Transformer 和 AR sampler
diffusion.py                   掩码破坏、去噪 loss 和并行 sampler
train.py                       token-budget 训练、评估、checkpoint 和 DDP
data.py                        字符与 memory-mapped token-shard loaders
experiment.py                  原子产物和本地实验元数据
compare_runs.py                从完成的运行生成 Markdown 对比报告
sample.py                      checkpoint 加载和文本生成
config.py                      类型化模型与训练配置
configs/shakespeare_char.py    可运行的约 10M 字符模型
configs/tinystories_106m.py    可运行的 TinyStories 成对配置
configs/fineweb_350m.py        4×A40 目标配置
tests/                         行为和基础设施测试
docs/architecture.md           设计选择与路线图
```

## 范围与限制

当前版本强调代码清晰和正确的实验骨架。尚未实现精确 MDLM likelihood weighting、
remasking、Block Diffusion、KV cache 和预训练 checkpoint。AR validation perplexity 与
MDLM denoising loss 属于不同目标，不能当作同一个 likelihood 指标直接比较。

更多设计说明见 [docs/architecture.md](docs/architecture.md)。

## License

MIT
