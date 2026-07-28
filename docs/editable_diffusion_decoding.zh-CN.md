# 可修正扩散解码

本文记录一个受
[LLaDA2.2 技术报告](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf)
启发、但刻意保持最小范围的实验。它**不是** LLaDA2.2 复现。本仓库的主问题仍然是：在相同
Transformer、数据、tokenizer、训练预算、提示和评测协议下，公平比较自回归（AR）与掩码扩散
（MDLM）目标。

## 最值得迁移的思想

LLaDA2.2 最可迁移的启发不是 100B MoE、128K context 或 agentic RL，而是“diffusion
decoding 不应过早把局部预测变成不可修改的硬约束”。LLaDA2.2 用 Levenshtein editing 实现
这一原则：模型预测 KEEP、SUBSTITUTE、DELETE 和 INSERT，监督标签根据最长公共子序列
（LCS）动态构造，之后再用 L-EBPO 结合环境奖励优化块级编辑决策。

本实验只问一个更窄的问题：固定长度的 MDLM 采样器，能否偶尔把已揭示且置信度最低的 token
重新变成 mask？

## 实现内容

`sample_masked` 新增三个可选控制项：

- `remask_fraction`：每一步最多回退的可编辑已揭示位置比例；
- `remask_decay_start`：从哪个进度开始将回退比例线性衰减到零；
- `protected_token_ids`：GPT-2 EOT 等不能再次被 mask 的结构 token。

Prompt 始终固定。排序使用每个位置被揭示时记录的概率，而不是事后重新计算的分数。最后一步
不会增加 mask。`remask_fraction=0` 与原采样器逐 token 一致，也是公共默认值。

示例：

```bash
python sample.py \
  --checkpoint out/tinystories-106m-mdlm-seed1337/best.pt \
  --length 160 --steps 64 \
  --remask-fraction 0.10 \
  --remask-decay-start 0.75 \
  --protect-eot
```

## 解码器对比

| 属性 | Cached AR | 原始 MDLM | Remask MDLM | LLaDA2.2 Levenshtein editing |
|---|---|---|---|---|
| 注意力与条件依赖 | 因果注意力；下一个 token 依赖固定前缀 | 对 prompt、mask 与已揭示 token 双向建模 | 与原 MDLM 完全相同 | Block diffusion 条件与编辑动作 |
| 已生成 token 能否修改 | 不能 | 不能 | 能；低置信度位置可重新 mask | 能；KEEP/SUBSTITUTE/DELETE/INSERT |
| 插入、删除与动态长度 | 不支持 | 不支持；固定输出画布 | 不支持；固定输出画布 | 支持显式插入和删除 |
| 本实验的网络更新次数 | 128 次/128-token continuation | 64 | 64 | 取决于架构和编辑轨迹 |
| 并行性 | 每次 cached update 新增一个 token | 每次可揭示多个位置 | 并行揭示，但部分位置可能回退 | 并行块去噪加编辑决策 |
| 主要风险 | 左到右错误累积 | 早期错误按任意揭示顺序固化 | 重复计算、后期结构不稳定 | 编辑 credit assignment 更难；INSERT 比 DELETE 更难学习 |

因此，remask 更接近采样器侧回滚，而不是一个学习得到的编辑器；它不改变训练目标和 checkpoint。

## 第一阶段：100-prompt 筛选

筛选复用了固定 prompts 和两个 106M MDLM 训练 seed，每行包含 200 条 continuation。

| 方案 | EOT 完成率 | Distinct-3 | Repeated 4-grams | Gate |
|---|---:|---:|---:|---|
| 原始 MDLM | 23.0% | 0.832 | 0.0963 | 基线 |
| 固定 5% | 17.0% | 0.863 | 0.0749 | 失败：EOT |
| 固定 10% | 18.0% | 0.888 | 0.0544 | 失败：EOT |
| 10% + EOT 保护 | 19.5% | 0.888 | 0.0545 | 失败：EOT |
| 5% + EOT 保护 + 最后 25% 衰减 | 22.0% | 0.848 | 0.0842 | 失败：重复率 |
| 10% + EOT 保护 + 最后 25% 衰减 | 22.5% | 0.862 | 0.0704 | **通过** |

进入正式实验要求 repeated-4gram 相对基线至少降低 20%，EOT 完成率下降不超过 2 个百分点，
distinct-3 下降不超过 0.01。只有最后一种方案通过：重复率相对降低 26.9%，EOT 仅下降 0.5
个百分点。

## 第二阶段：1,000-prompt 正式评测

最佳候选共生成 2,000 条 continuation：两个训练 seed 各使用 1,000 个 prompt。实验直接复用
已有 2,000 条 cached AR 与 2,000 条不可逆 64-step MDLM 输出。DeepSeek V4 Flash 评审全部
4,000 个隐藏身份、位置平衡的新 pair；DeepSeek V4 Pro 对每组确定性抽取 100 对复核。

![TinyStories 106M 可修正解码质量–延迟前沿](assets/remask_quality_latency_frontier.png)

| 解码器 | EOT 完成率 | Distinct-3 | Repeated 4-grams |
|---|---:|---:|---:|
| Cached AR | 25.20% | 0.950 | 0.0195 |
| 原始 MDLM | 26.50% | 0.832 | 0.0951 |
| Remask MDLM | 28.25% | 0.867 | 0.0672 |

透明指标确实改善：相对原 MDLM，remask 将 repeated-4gram 降低 29.4%，distinct-3 提高
0.034，而且没有损害 EOT 完成率。但盲评否定了“代理指标改善等于真实质量提升”这一诱人结论。

| 评审与对比 | Pair 数 | Remask 胜 | 平 | Remask 负 | Remask 效用 |
|---|---:|---:|---:|---:|---:|
| Flash：remask vs AR | 2,000 | 64 | 17 | 1,919 | 0.036 |
| Flash：remask vs 原 MDLM | 2,000 | 356 | 714 | 930 | 0.357 |
| Pro 抽查：remask vs AR | 100 | 6 | 0 | 94 | 0.060 |
| Pro 抽查：remask vs 原 MDLM | 100 | 27 | 6 | 67 | 0.300 |

效用将 remask 胜、平、负分别记为 1、0.5、0。两种评审都在直接对比中更偏好原始 MDLM，
即使多样性和重复率指标朝预期方向变化。合理解释是：揭示时的低置信度在周围上下文变化后，
未必仍是可靠编辑目标；同时在固定 64 次 update 内，重新采样旧位置也会挤占只解决新位置的预算。

这是有价值的负面结果。实验采样器继续保留供研究使用，但公共默认值仍是不可逆 MDLM。

同一次运行的性能测量也给出相同工程结论。batch 1 时，remask MDLM 为 0.410 秒，原 MDLM
为 0.386 秒，cached AR 为 0.678 秒；batch 32 时，remask 为 3.492 秒，原 MDLM 为 2.570
秒。Revisability 保持 64 次网络更新，却需要再次投影和采样更多位置，因此并不是免费改进。

## 范围边界与下一步

本次实现不包含 Block Diffusion、学习得到的 KEEP/SUBSTITUTE/DELETE/INSERT、LCS 标签、MoE
routing 或 L-EBPO。这些能力会改变模型、训练数据构造、目标函数或优化流程，应作为独立实验。

下一轮 revisability 实验应使用能反映当前上下文的置信度或 disagreement 信号，显式核算更新
预算，并将 learned editing 与 block diffusion 分开验证。任何候选仍需保留相同的 AR 对照和
盲评审计；repeated n-gram 不能单独作为选择目标。

完整测量表与溯源见
[`reports/tinystories_106m_remask_generation.md`](../reports/tinystories_106m_remask_generation.md)，
筛选记录见 [`reports/remask_validation_100.md`](../reports/remask_validation_100.md)。
