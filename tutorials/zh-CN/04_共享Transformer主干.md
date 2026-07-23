# 04｜共享 Transformer 主干

这一章沿着 `model.py` 的前向传播追踪 tensor shape。AR 和 MDLM 使用同一个 `Transformer`
类，差异由配置中的注意力方向、mask embedding 与时间条件控制。

## 正式一亿参数配置

`configs/tinystories_106m.py` 的主要尺寸是：

| 字段 | 数值 | 含义 |
|---|---:|---|
| `vocab_size` | 50,257 | GPT-2 正常输出词表 |
| `block_size` | 1,024 | 最大上下文长度 |
| `n_layer` | 12 | Transformer block 数量 |
| `n_head` | 9 | 每层注意力头数 |
| `n_embd` | 576 | 每个位置的隐藏维度 |

每个注意力头维度为 `576 / 9 = 64`。`n_embd` 必须能被 `n_head` 整除，代码会主动检查。

AR 关闭 mask token 和时间条件，共 106,276,032 参数。MDLM 的输入 embedding 多一个 mask
位置，并增加时间 MLP，共 106,610,112 参数。

## 第一步：词元与位置 embedding

输入 `idx` 形状为 `[B, T]`。每个词元编号查表得到 `[B, T, C]`，每个位置编号查表得到
`[T, C]`：

```python
x = token_embedding(idx) + position_embedding(positions)
```

位置 embedding 通过广播加到 batch 中的每一条序列。模型使用学习式绝对位置，因此输入长度
不能超过 `block_size`。

MDLM 还会把每条序列的噪声等级 `t: [B]` 输入两层 MLP，得到 `[B, C]`，再广播到序列中的每个
位置：

```python
x = x + time_mlp(t[:, None])[:, None, :]
```

这告诉模型当前序列大约被破坏了多少。AR 不创建 `time_mlp`。

## 第二步：多头自注意力

输入 `x: [B, T, C]` 先经过一个线性层，同时得到 query、key 和 value：

```text
[B, T, C] --qkv linear--> [B, T, 3C]
                         ├── q [B, T, C]
                         ├── k [B, T, C]
                         └── v [B, T, C]
```

每个张量重排成 `[B, H, T, D]`：

- `H = n_head`；
- `D = C / H`；
- 正式模型中为 `[B, 9, T, 64]`。

代码调用 PyTorch 的 `scaled_dot_product_attention`，让后端根据设备选择合适的 SDPA 实现。
最关键的开关是 `is_causal`：

- AR 为 `True`，位置 `i` 看不到 `i+1` 及更右侧；
- MDLM 为 `False`，每个位置可以利用左右两侧的可见词元。

多头结果再合并回 `[B, T, C]`，经过输出投影与 dropout。

## 第三步：残差与 MLP

每个 `Block` 使用 pre-norm 结构：

```python
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

MLP 先把隐藏维度从 `C` 扩展到 `4C`，使用 GELU，再投影回 `C`。残差路径让每个 block 学习对
已有表示的增量，而不是从头重建整个表示。

12 个 block 之后再做一次 LayerNorm，得到最终 hidden states `[B, T, C]`。

## 第四步：输出到词表

`lm_head` 把每个隐藏向量从 `C` 投影到 `V`：

```text
[B, T, C] --lm_head--> [B, T, V]
```

`[B, T, 50,257]` 可能很大。MDLM 训练只需要 masked positions 的 logits，因此
`Transformer.forward` 接受布尔 `output_mask: [B, T]`，先选择 hidden states：

```text
[B, T, C] --布尔选择--> [M, C] --lm_head--> [M, V]
```

`M` 是当前 batch 中被掩码的位置总数。先选择 hidden 再做词表投影，可以显著降低训练和采样
显存。注意：Transformer block 仍处理完整序列，因为双向上下文正是 MDLM 所需的信息。

## 用小模型观察 shape

```bash
python - <<'PY'
import torch
from config import ModelConfig
from model import Transformer

ar_config = ModelConfig(
    vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16,
    objective="autoregressive", mask_token_id=None, time_conditioning=False,
)
ar = Transformer(ar_config)
tokens = torch.randint(20, (3, 8))
print("AR input:", tuple(tokens.shape))
print("AR logits:", tuple(ar(tokens).shape))

mdlm_config = ModelConfig(
    vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16,
    objective="masked_diffusion", mask_token_id=20, time_conditioning=True,
)
mdlm = Transformer(mdlm_config)
mask = torch.zeros((3, 8), dtype=torch.bool)
mask[:, ::2] = True
noisy = tokens.clone()
noisy[mask] = 20
t = mask.float().mean(dim=1)
print("MDLM full logits:", tuple(mdlm(noisy, t).shape))
print("MDLM masked logits:", tuple(mdlm(noisy, t, mask).shape))
PY
```

预期 shape：

```text
AR input: (3, 8)
AR logits: (3, 8, 20)
MDLM full logits: (3, 8, 20)
MDLM masked logits: (12, 20)
```

3 条序列各选择 4 个位置，所以 masked logits 第一维是 12。

## KV cache 为什么只属于 AR

AR 每次生成一个新词元，历史位置的 key/value 不会变化，可以缓存每层的 `[B, H, T, D]`，
后续只计算新位置。

MDLM 每一轮会把多个 mask 替换成新词元，所有位置下一轮看到的上下文都可能变化。普通 AR 式
KV cache 不能直接复用，否则缓存中仍是旧 mask 状态。块扩散可能在受限注意力结构下设计不同的
缓存策略，但不属于当前基线。

## 完成检查

- 你能从 `[B, T]` 追踪到 `[B, H, T, D]` 再回到 `[B, T, V]`；
- 你知道 AR/MDLM 的注意力差异由 `is_causal` 控制；
- 小模型脚本得到 `(12, 20)` 的 masked logits；
- 你能解释 masked-only output projection 为什么省显存，但不省 Transformer 全序列计算。

[上一章：从文本到训练批次](03_从文本到训练批次.md) ·
[下一章：自回归语言模型](05_自回归语言模型.md)

