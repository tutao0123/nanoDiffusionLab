# 10｜四卡 DDP 与断点恢复

这一章解释 `torchrun`、DDP、梯度累积和精确恢复如何在当前训练循环中配合。目标不是掌握所有
分布式训练方法，而是能够判断四卡任务是否真的按预期运行，以及中断后能否安全继续。

## 为什么这个规模选择 DDP

约一亿参数模型可以完整放进每张 A40。DDP 在每张 GPU 保存一份模型副本，每个 rank 读取不同
随机 batch，反向传播时同步梯度：

```text
rank 0: model copy + batch A ─┐
rank 1: model copy + batch B ─┼─ all-reduce gradients ─ optimizer step
rank 2: model copy + batch C ─┤
rank 3: model copy + batch D ─┘
```

没有 NVLink 的 PHB 拓扑会让 GPU 通信经过 PCIe Host Bridge，但一亿参数仍适合数据并行。FSDP
和 ZeRO-3 的参数分片能节省模型状态显存，却增加通信与实现复杂度；Tensor Parallel 则让每层
前向都产生跨卡通信。在模型完整可放入单卡时，它们不是这里的优先方案。

## `torchrun` 做了什么

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py ...
```

它启动四个 Python 进程，并为每个进程设置：

- `RANK`：全局进程编号 0 到 3；
- `LOCAL_RANK`：本机显卡编号 0 到 3；
- `WORLD_SIZE`：总进程数 4。

`train.py` 读取这些环境变量，调用 `torch.cuda.set_device(local_rank)`，再初始化 NCCL process
group。没有这些变量时，同一个文件自动退化为单进程运行。

## 为什么只有 rank 0 写文件

四个进程同时写 `metrics.jsonl` 或 checkpoint 会互相覆盖。代码定义 `master = rank == 0`，只有
master 创建 logger、写 JSON、打印主日志和保存最终文件。

但 checkpoint 中的随机状态不能只保存 rank 0。每个 rank 会收集：

- Python random state；
- CPU torch RNG state；
- 训练数据 generator state；
- 对应 CUDA device RNG state。

`all_gather_object` 把四份状态交给 rank 0，一起写入 checkpoint。恢复时每个 rank 取回自己的
状态，从而继续相同的数据与噪声随机流。

## 梯度累积期间为什么使用 `no_sync`

参考配置每个 optimizer step 有 2 个 micro-step。如果每次 backward 都同步梯度，会进行两次
all-reduce。代码在非最后一个 micro-step 使用 DDP `no_sync()`：

```text
micro-step 1: backward locally, no all-reduce
micro-step 2: backward and all-reduce accumulated gradients
optimizer step
```

这样每个 optimizer step 只同步一次，减少 PHB 拓扑上的通信开销。loss 在 backward 前除以
梯度累积次数，使累积梯度对应 micro-batch loss 的平均。

## 在 tmux 中启动长任务

创建会话：

```bash
tmux new -s nanodiffusion-train
```

在会话中启动：

```bash
bash scripts/run_tinystories_pair.sh
```

按 `Ctrl-b`，松开后按 `d`，可以 detach 而不中断训练。重新进入：

```bash
tmux attach -t nanodiffusion-train
```

列出会话：

```bash
tmux list-sessions
```

如果 attach 后看似空白，先按回车、向上滚动，或从另一个终端检查真实进程与日志：

```bash
ps -eo pid,etimes,cmd | rg 'train.py|torch.distributed'
tail -30 out/tinystories-106m-ar/console.log
```

不要仅根据 tmux pane 是否有当前输出判断训练是否存在。

## 监控四张卡

```bash
watch -n 2 nvidia-smi
```

或使用紧凑输出：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

稳定训练时四张卡通常都有相近显存与较高利用率。短暂下降可能来自 evaluation、checkpoint 或
数据切换。某一张卡长期为 0%，而进程仍宣称 world size 4，才需要检查 rank 是否崩溃。

## 精确恢复命令

输出目录包含 `last.pt` 后：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/tinystories_106m.py \
  --objective autoregressive \
  --out-dir out/tinystories-106m-ar \
  --resume
```

也可以指定路径：

```bash
... --resume out/tinystories-106m-ar/tokens-001000000000.pt
```

省略路径时读取当前 `out_dir/last.pt`。

## 恢复时必须保持什么一致

代码明确检查：

- checkpoint 中的模型配置与当前模型配置完全相同；
- `world_size` 完全相同。

优化器、scaler、step、tokens seen、best val、累计训练时间与每个 rank RNG 都从 checkpoint
恢复。若原来四卡、现在单卡，不能称为精确恢复。可以另写权重加载或迁移流程，但那属于新的
实验条件，应使用新目录并记录。

## checkpoint 保存时机

训练在每次 evaluation 时保存 `last.pt`，验证改善时保存 `best.pt`。跨过配置中的 token
milestone 时还保存：

```text
tokens-000500000000.pt
tokens-001000000000.pt
tokens-002000000000.pt
```

如果进程在两次 evaluation 之间被强制终止，只能从上一个 `last.pt` 恢复，中间尚未保存的 step
会重跑。不要把 `kill -9` 当作正常暂停方式。

## 成对脚本如何自动恢复

`scripts/run_tinystories_pair.sh` 检查目标目录是否已有 `last.pt`。存在时自动添加 `--resume`，
不存在时从零开始。完成 AR 后才会进入 MDLM。

第二 seed 使用：

```bash
SEED=2027 bash scripts/run_tinystories_replication.sh
```

这个脚本写到带 seed 后缀的独立目录，避免两个随机种子互相覆盖。

## DDP 不是四倍加速保证

实际加速受以下因素影响：

- GPU 之间的 NVLink/PCIe/PHB/SYS 拓扑；
- 梯度张量大小与 all-reduce 带宽；
- 每卡 batch 是否足以让计算覆盖通信；
- 数据读取与 CPU 调度；
- evaluation 和 checkpoint 频率；
- PyTorch、CUDA、NCCL 版本。

因此报告“4 卡 289K tok/s”时必须同时说明单机、A40、PHB、BF16 和具体 batch。只写 GPU 数量
不足以比较系统效率。

## 完成检查

- 四个 rank 分别绑定到四张目标 GPU；
- rank 0 日志显示 `world_size=4` 和正确的 tokens per step；
- `nvidia-smi` 中四张卡都有训练进程与相近占用；
- `last.pt` 包含 4 份 rank state；
- 使用相同 world size 的 `--resume` 能从已有 step 继续，而不是回到 step 0。

[上一章：训练一亿参数模型](09_训练一亿参数模型.md) ·
[下一章：公平评测生成质量与速度](11_公平评测生成质量与速度.md)
