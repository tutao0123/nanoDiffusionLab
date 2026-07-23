# 08｜准备 TinyStories

这一章下载固定版本的 TinyStories，使用 GPT-2 BPE 分词，并写成可以 memory-map 的二进制
分片。完整数据准备主要消耗网络、CPU 和约 1 GB 磁盘，不需要 GPU。

## 安装数据依赖

```bash
python -m pip install -e ".[data,dev]"
```

确认命令可用：

```bash
python scripts/prepare_tinystories.py --help
```

## 先准备一个小型子集

第一次运行建议写到独立目录：

```bash
python scripts/prepare_tinystories.py \
  --output data/tinystories-gpt2-tutorial-mini \
  --max-train-documents 1000 \
  --max-validation-documents 200 \
  --shard-tokens 1000000
```

这个子集只用于检查下载、tokenizer、分片与 loader。不要把它产生的 loss 或生成质量与正式报告
比较。

查看结果：

```bash
find data/tinystories-gpt2-tutorial-mini -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
python -m json.tool data/tinystories-gpt2-tutorial-mini/manifest.json | head -80
```

## 完整准备命令

```bash
python scripts/prepare_tinystories.py
```

默认输出为 `data/tinystories-gpt2`。脚本固定：

- dataset：`roneneldan/TinyStories`；
- revision：`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`；
- tokenizer：GPT-2 BPE；
- dtype：`uint16`；
- 默认每个分片最多 100,000,000 tokens。

参考完整产物约 914 MB。实际下载缓存还会占用 Hugging Face cache 空间，准备前应留出额外磁盘。

## 为什么固定 revision

只写数据集名称不能保证未来下载到完全相同的内容。revision 是 Git commit 风格的不可变标识。
正式报告记录它，别人才能复现同一份 parquet，而不是某个时间点的最新版本。

## 为什么使用 GPT-2 BPE

字符模型适合展示位置，但一亿参数实验需要更合理的词元粒度。GPT-2 tokenizer 有 50,257 个
词元，公开、稳定且能直接处理 TinyStories 英文文本。

不要把 BPE token 数量当作字符数或单词数。`Once upon a time` 可能分成若干子词与空格前缀，
具体边界由 tokenizer 决定。

观察编码：

```bash
python - <<'PY'
import tiktoken

encoding = tiktoken.get_encoding("gpt2")
text = "Once upon a time, there was a little fox."
tokens = encoding.encode_ordinary(text)
print(tokens)
print([encoding.decode([token]) for token in tokens])
print(encoding.decode(tokens))
PY
```

最后一行应当恢复原文本。

## 分片写入过程

脚本批量从 parquet 读取 `text`，调用 `encode_ordinary_batch`，并在每篇文档末尾追加 GPT-2
end-of-text token。然后 `ShardWriter` 按 token 数切分 `.bin`。

每个分片关闭时计算 SHA-256，并写入 manifest：

```json
{
  "path": "train/00000.bin",
  "tokens": 100000000,
  "sha256": "..."
}
```

这允许你发现不完整复制或存储损坏。训练 loader 当前信任 manifest，不会每次启动都重算全部
哈希，因为扫描近 1 GB 文件会拖慢启动；交付数据时应单独验证。

## 验证分片大小

`uint16` 每个 token 占 2 bytes，因此文件大小应约等于 `tokens × 2`：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data/tinystories-gpt2")
manifest = json.loads((root / "manifest.json").read_text())
for split, info in manifest["splits"].items():
    print(split)
    for shard in info["shards"]:
        size = (root / shard["path"]).stat().st_size
        expected = shard["tokens"] * 2
        print(shard["path"], size, "OK" if size == expected else f"expected {expected}")
PY
```

## 从分片读取一个 batch

```bash
python - <<'PY'
import torch
from data import TokenizedData

data = TokenizedData(
    "data/tinystories-gpt2/manifest.json",
    block_size=32,
    device=torch.device("cpu"),
)
batch = data.batch("validation", 2, generator=torch.Generator().manual_seed(42))
print("shape:", tuple(batch.shape))
print("dtype:", batch.dtype)
print("min/max:", int(batch.min()), int(batch.max()))
print(data.description())
PY
```

应得到 `(2, 32)` 的 `torch.int64`。最大值应小于 50,257；正常训练窗口不会把额外 mask token
写入原始数据。

## 不要覆盖正式数据做小实验

如果 `data/tinystories-gpt2` 已经包含完整数据，不要用带 `--max-train-documents` 的命令指向同一
目录。教程子集使用 `data/tinystories-gpt2-tutorial-mini`，就是为了避免分片与 manifest 混合。

## 常见问题

### 下载很慢或 Xet 异常

脚本默认设置 `HF_HUB_DISABLE_XET=1`，使用普通 HTTPS 更适合参考服务器。仍然失败时检查代理、
DNS、Hugging Face 可访问性和 cache 目录空间。

### `No space left on device`

同时检查项目数据目录和 Hugging Face cache：

```bash
df -h .
du -sh ~/.cache/huggingface 2>/dev/null || true
```

不要在不确认用途时直接删除共享服务器的 cache。

### 中途失败后如何继续

当前准备脚本不是面向分片级断点续传设计。先检查已生成目录，不要把残留分片误认为完整数据。
最安全做法是换一个新输出目录重新运行，成功后再原子地切换训练配置所指向的目录。

## 完成检查

- 小型子集包含 train、validation 分片与 manifest；
- 正式复现时，manifest 的 dataset revision 与 tokenizer 正确；
- 分片字节数等于记录 token 数的两倍；
- `TokenizedData` 能读取 `(2, 32)` batch。

[上一章：字符级训练与采样](07_字符级训练与采样.md) ·
[下一章：训练一亿参数模型](09_训练一亿参数模型.md)
