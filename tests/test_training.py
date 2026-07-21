import argparse

from config import ModelConfig, TrainConfig
from model import Transformer
from train import apply_overrides, learning_rate


def arguments(**overrides: object) -> argparse.Namespace:
    values = {
        "data": None,
        "out_dir": None,
        "run_name": None,
        "objective": None,
        "max_iters": None,
        "max_tokens": None,
        "batch_size": None,
        "gradient_accumulation_steps": None,
        "eval_interval": None,
        "eval_iters": None,
        "seed": None,
        "no_compile": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_objective_override_sets_all_model_fields() -> None:
    config = TrainConfig(out_dir="out/example-mdlm")
    apply_overrides(
        config,
        arguments(objective="autoregressive", max_tokens=1000, seed=2027),
    )

    assert config.model.objective == "autoregressive"
    assert config.model.mask_token_id is None
    assert not config.model.time_conditioning
    assert config.out_dir == "out/example-ar"
    assert config.max_tokens == 1000
    assert config.seed == 2027


def test_106m_architecture_parameter_counts() -> None:
    common = dict(
        vocab_size=50_257,
        block_size=1_024,
        n_layer=12,
        n_head=9,
        n_embd=576,
        dropout=0.0,
    )
    ar = Transformer(
        ModelConfig(
            **common,
            objective="autoregressive",
            mask_token_id=None,
            time_conditioning=False,
        )
    )
    mdlm = Transformer(
        ModelConfig(
            **common,
            objective="masked_diffusion",
            mask_token_id=50_257,
            time_conditioning=True,
        )
    )

    assert ar.parameter_count() == 106_276_032
    assert mdlm.parameter_count() == 106_610_112


def test_learning_rate_warms_up_and_decays() -> None:
    config = TrainConfig(warmup_iters=10, learning_rate=1e-3, min_lr=1e-4)
    assert learning_rate(0, 100, config) == 1e-4
    assert learning_rate(9, 100, config) == 1e-3
    assert learning_rate(100, 100, config) == 1e-4
