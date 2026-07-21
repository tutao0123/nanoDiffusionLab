from config import ModelConfig, TrainConfig

config = TrainConfig(
    model=ModelConfig(
        vocab_size=65,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        objective="masked_diffusion",
        mask_token_id=65,
        time_conditioning=True,
    ),
    data_path="data/tinyshakespeare/input.txt",
    out_dir="out/shakespeare-mdlm",
    batch_size=32,
    gradient_accumulation_steps=1,
    max_iters=5_000,
    eval_interval=250,
    learning_rate=3e-4,
)
