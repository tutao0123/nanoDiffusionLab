from config import ModelConfig, TrainConfig

# Target configuration for 4x A40. A sharded-token data loader is a planned milestone.
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
    data_path="data/fineweb/input.txt",
    out_dir="out/fineweb-350m",
    batch_size=4,
    gradient_accumulation_steps=8,
    max_iters=200_000,
    eval_interval=1_000,
    eval_iters=100,
    learning_rate=3e-4,
    warmup_iters=2_000,
    dtype="bfloat16",
    compile=True,
    gradient_checkpointing=True,
    backend="nccl",
)
