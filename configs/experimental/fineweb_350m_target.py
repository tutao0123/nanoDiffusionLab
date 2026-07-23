from config import ModelConfig, TrainConfig

# Architecture and hardware target only. The repository does not yet include the
# FineWeb-Edu tokenization and shard-manifest recipe referenced by data_path.
config = TrainConfig(
    model=ModelConfig(
        vocab_size=32_000,
        block_size=1_024,
        n_layer=24,
        n_head=16,
        n_embd=1_024,
        dropout=0.0,
        objective="masked_diffusion",
        mask_token_id=32_000,
    ),
    data_format="tokenized",
    data_path="data/fineweb-edu/manifest.json",
    out_dir="out/fineweb-350m-target",
    batch_size=4,
    gradient_accumulation_steps=8,
    max_iters=200_000,
    eval_interval=1_000,
    eval_iters=100,
    learning_rate=3e-4,
    warmup_iters=2_000,
    dtype="bfloat16",
    compile=False,
    gradient_checkpointing=True,
    backend="nccl",
)
