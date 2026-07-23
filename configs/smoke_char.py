"""Tiny CPU configuration used by CI and installation smoke tests."""

from config import ModelConfig, TrainConfig

config = TrainConfig(
    model=ModelConfig(
        vocab_size=65,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        objective="masked_diffusion",
        mask_token_id=65,
        time_conditioning=True,
    ),
    data_format="char",
    data_path="tests/fixtures/tiny_corpus.txt",
    out_dir="out/cpu-smoke",
    batch_size=2,
    gradient_accumulation_steps=1,
    max_iters=2,
    eval_interval=1,
    eval_iters=1,
    log_interval=1,
    learning_rate=1e-3,
    min_lr=1e-4,
    warmup_iters=1,
    dtype="float32",
    compile=False,
    gradient_checkpointing=False,
    always_save_checkpoint=True,
)
