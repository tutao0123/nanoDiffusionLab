from config import ModelConfig, TrainConfig

# CharacterData is the runnable first data path. Tokenized TinyStories support is
# intentionally left for the next milestone; these dimensions define the target model.
config = TrainConfig(
    model=ModelConfig(
        vocab_size=16_000,
        block_size=256,
        n_layer=12,
        n_head=8,
        n_embd=512,
        dropout=0.1,
        objective="masked_diffusion",
        mask_token_id=16_000,
    ),
    data_format="tokenized",
    data_path="data/tinystories/input.txt",
    out_dir="out/tinystories-50m",
    batch_size=16,
    gradient_accumulation_steps=8,
    max_iters=100_000,
    eval_interval=1_000,
    eval_iters=100,
    learning_rate=3e-4,
    warmup_iters=1_000,
)
