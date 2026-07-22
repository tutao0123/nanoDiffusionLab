"""Build README figures from committed experiment data.

Pass ``--refresh-data`` on the training machine to extract compact, shareable
evaluation traces from the ignored run directories before rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "reports/data/tinystories_106m_results.json"
ASSET_DIR = ROOT / "docs/assets"
RUNS = {
    "ar_seed1337": ROOT / "out/tinystories-106m-ar",
    "mdlm_seed1337": ROOT / "out/tinystories-106m-mdlm",
    "ar_seed2027": ROOT / "out/tinystories-106m-ar-seed2027",
    "mdlm_seed2027": ROOT / "out/tinystories-106m-mdlm-seed2027",
}
BLUE = "#2563eb"
ORANGE = "#ea580c"
INK = "#172033"
MUTED = "#667085"
GRID = "#dfe3ea"
PAPER = "#f8fafc"


def refresh_data() -> None:
    payload: dict[str, object] = {
        "experiment": "TinyStories 106M, 2B input tokens per objective",
        "seeds": [1337, 2027],
        "runs": {},
    }
    runs = payload["runs"]
    assert isinstance(runs, dict)
    for name, run_dir in RUNS.items():
        summary = json.loads((run_dir / "summary.json").read_text())
        evaluations = []
        with (run_dir / "metrics.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("event") == "eval":
                    evaluations.append(
                        {
                            "step": row["step"],
                            "tokens_seen": row["tokens_seen"],
                            "val_loss": row["val_loss"],
                        }
                    )
        runs[name] = {
            "objective": summary["objective"],
            "parameter_count": summary["parameter_count"],
            "tokens_seen": summary["tokens_seen"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "mean_training_tokens_per_second": summary["mean_training_tokens_per_second"],
            "peak_memory_mb": summary["peak_memory_mb"],
            "final_metrics": summary["final_metrics"],
            "evaluations": evaluations,
        }
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def render_curves(data: dict[str, object]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True)
    panels = (
        ("ar", "Autoregressive validation loss", "Next-token cross-entropy"),
        ("mdlm", "Masked diffusion validation loss", "Mean over mask ratios 10–90%"),
    )
    runs = data["runs"]
    assert isinstance(runs, dict)
    for axis, (prefix, title, subtitle) in zip(axes, panels, strict=True):
        for seed, color, style in ((1337, BLUE, "-"), (2027, ORANGE, "--")):
            run = runs[f"{prefix}_seed{seed}"]
            evaluations = run["evaluations"]
            x = [point["tokens_seen"] / 1e9 for point in evaluations]
            y = [point["val_loss"] for point in evaluations]
            axis.plot(
                x,
                y,
                color=color,
                linestyle=style,
                linewidth=2.3,
                label=f"seed {seed}",
            )
        axis.set_title(title, loc="left", fontsize=14, weight="bold", color=INK, pad=17)
        axis.text(0, 1.015, subtitle, transform=axis.transAxes, color=MUTED, fontsize=10)
        axis.set_xlabel("Input tokens (billions)", color=MUTED)
        axis.set_ylabel("Validation loss", color=MUTED)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#aab2c0")
        axis.legend(frameon=False, loc="upper right")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.1f}B"))
    fig.suptitle(
        "TinyStories 106M training curves",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.02,
        "Two independent seeds; each objective receives the same 2B input-token budget. "
        "Losses are objective-specific and should not be compared across panels.",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.09, 0.99, 0.88), w_pad=3.2)
    fig.savefig(ASSET_DIR / "training_curves.png", dpi=180, facecolor="white")
    plt.close(fig)


def render_summary(data: dict[str, object]) -> None:
    runs = data["runs"]
    assert isinstance(runs, dict)
    metrics = [
        ("AR perplexity", "ar", "val_perplexity", "lower is better"),
        ("MDLM masked accuracy", "mdlm", "val_masked_accuracy", "higher is better"),
        ("AR throughput", "ar", "mean_training_tokens_per_second", "input tokens/s"),
        ("MDLM throughput", "mdlm", "mean_training_tokens_per_second", "input tokens/s"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for axis, (title, prefix, field, note) in zip(axes, metrics, strict=True):
        values = []
        for seed in (1337, 2027):
            run = runs[f"{prefix}_seed{seed}"]
            if field in run:
                value = run[field]
            else:
                value = run["final_metrics"][field]
            values.append(float(value))
        if "throughput" in title.lower():
            labels = [f"{value / 1000:.1f}K" for value in values]
        elif "accuracy" in title.lower():
            labels = [f"{value * 100:.2f}%" for value in values]
        else:
            labels = [f"{value:.4f}" for value in values]
        center = sum(values) / 2
        spread = max(abs(values[0] - values[1]) * 2.5, abs(center) * 0.01)
        axis.set_xlim(center - spread, center + spread)
        axis.scatter(values, [1, 0], s=100, c=[BLUE, ORANGE], edgecolor="white", linewidth=1.5)
        axis.plot(values, [1, 0], color="#98a2b3", linewidth=1.2, zorder=0)
        for value, y, label in zip(values, [1, 0], labels, strict=True):
            axis.annotate(
                label,
                (value, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                weight="bold",
            )
        axis.set_yticks([1, 0], ["seed 1337", "seed 2027"])
        axis.set_xticks([])
        axis.set_ylim(-0.45, 1.55)
        axis.set_title(title, loc="left", fontsize=12, weight="bold", color=INK)
        axis.text(0, 0.93, note, transform=axis.transAxes, fontsize=9, color=MUTED)
        axis.spines[:].set_visible(False)
        axis.tick_params(axis="y", length=0, colors=MUTED)
    fig.suptitle(
        "Final metrics are stable across seeds",
        x=0.04,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.02,
        "Focused scales with exact labels; connecting lines show seed-to-seed movement, "
        "not a time trend.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.08, 0.99, 0.88), w_pad=2.6)
    fig.savefig(ASSET_DIR / "replication_summary.png", dpi=180, facecolor="white")
    plt.close(fig)


def token_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def render_decoding_gif() -> None:
    width, height = 1200, 610
    tokens = ["Once", "upon", "a", "time", "there", "was", "a", "fox"]
    mdlm_order = [[0, 7], [3, 5], [1, 6], [2, 4]]
    frames = []
    for frame_index in range(9):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text(
            (60, 35), "One Transformer, two decoding paths", font=token_font(34, True), fill=INK
        )
        draw.text(
            (60, 83),
            "Sequential next-token generation vs. iterative parallel unmasking",
            font=token_font(19),
            fill=MUTED,
        )
        draw.rounded_rectangle(
            (45, 135, 1155, 315), radius=22, fill="#eff6ff", outline="#bfdbfe", width=2
        )
        draw.text((75, 158), "AUTOREGRESSIVE", font=token_font(20, True), fill=BLUE)
        draw.text(
            (75, 193),
            f"step {min(frame_index, 8)}/8  |  reveal left to right",
            font=token_font(17),
            fill=MUTED,
        )
        draw.rounded_rectangle(
            (45, 350, 1155, 530), radius=22, fill="#fff7ed", outline="#fed7aa", width=2
        )
        mdlm_step = min((frame_index + 1) // 2, 4)
        draw.text((75, 373), "MASKED DIFFUSION", font=token_font(20, True), fill=ORANGE)
        draw.text(
            (75, 408),
            f"denoise step {mdlm_step}/4  |  reveal multiple positions",
            font=token_font(17),
            fill=MUTED,
        )
        for token_index, token in enumerate(tokens):
            x = 75 + token_index * 130
            ar_visible = token_index < frame_index
            ar_text = token if ar_visible else "..."
            ar_fill = "white" if ar_visible else "#dbeafe"
            draw.rounded_rectangle(
                (x, 240, x + 112, 290), radius=10, fill=ar_fill, outline=BLUE, width=2
            )
            draw.text(
                (x + 56, 265),
                ar_text,
                anchor="mm",
                font=token_font(17, ar_visible),
                fill=INK if ar_visible else BLUE,
            )
            revealed = {position for group in mdlm_order[:mdlm_step] for position in group}
            md_visible = token_index in revealed
            md_text = token if md_visible else "[MASK]"
            md_fill = "white" if md_visible else "#ffedd5"
            draw.rounded_rectangle(
                (x, 455, x + 112, 505), radius=10, fill=md_fill, outline=ORANGE, width=2
            )
            draw.text(
                (x + 56, 480),
                md_text,
                anchor="mm",
                font=token_font(15, md_visible),
                fill=INK if md_visible else ORANGE,
            )
        draw.text(
            (60, 565),
            "AR uses causal attention  •  MDLM uses bidirectional attention",
            font=token_font(17),
            fill=MUTED,
        )
        frames.append(image)
    frames[0].save(
        ASSET_DIR / "decoding_comparison.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[900] + [650] * 7 + [1500],
        loop=0,
        optimize=True,
    )


def render_character_diffusion_gif() -> None:
    """Render a character-cell explanation of training and iterative sampling."""
    width, height = 1200, 650
    target = "a tiny fox finds a red star."
    order = sorted(range(len(target)), key=lambda index: (index * 17) % len(target))
    frames = []
    steps = 7
    for step in range(steps + 1):
        image = Image.new("RGB", (width, height), "#0b1220")
        draw = ImageDraw.Draw(image)
        draw.text(
            (55, 32),
            "MASKED DIFFUSION — CHARACTER VIEW",
            font=token_font(31, True),
            fill="#f8fafc",
        )
        draw.text(
            (55, 77),
            "Train by reconstructing random masks; generate by iterative parallel unmasking",
            font=token_font(17),
            fill="#94a3b8",
        )

        cards = (
            (50, "1  CORRUPT", "clean x₀ → random masks at level t"),
            (425, "2  PREDICT", "bidirectional Transformer(xₜ, t)"),
            (800, "3  REVEAL", "keep high-confidence predictions"),
        )
        for x, title, body in cards:
            draw.rounded_rectangle(
                (x, 125, x + 350, 225),
                radius=16,
                fill="#111c30",
                outline="#334155",
                width=2,
            )
            draw.text((x + 20, 147), title, font=token_font(18, True), fill="#fb923c")
            draw.text((x + 20, 183), body, font=token_font(14), fill="#cbd5e1")
        draw.text((405, 164), "→", font=token_font(28, True), fill="#64748b")
        draw.text((780, 164), "→", font=token_font(28, True), fill="#64748b")

        draw.rounded_rectangle(
            (50, 260, 1150, 545),
            radius=20,
            fill="#0f1a2c",
            outline="#334155",
            width=2,
        )
        draw.text((75, 285), "SAMPLING", font=token_font(19, True), fill="#60a5fa")
        draw.text(
            (230, 286),
            f"reverse step {step}/{steps}   mask ratio {(steps - step) / steps:0.2f}",
            font=token_font(16),
            fill="#94a3b8",
        )
        visible_count = round(len(target) * step / steps)
        visible = set(order[:visible_count])
        previous_count = round(len(target) * max(step - 1, 0) / steps)
        newly_visible = set(order[previous_count:visible_count])
        cell_width = 34
        cell_gap = 3
        start_x = 75
        for index, character in enumerate(target):
            x = start_x + index * (cell_width + cell_gap)
            is_visible = index in visible
            if not is_visible:
                fill, outline, text_fill, display = "#431407", "#ea580c", "#fb923c", "?"
            elif index in newly_visible:
                fill, outline, text_fill = "#052e2b", "#2dd4bf", "#ccfbf1"
                display = "·" if character == " " else character
            else:
                fill, outline, text_fill = "#172554", "#3b82f6", "#dbeafe"
                display = "·" if character == " " else character
            draw.rounded_rectangle(
                (x, 350, x + cell_width, 400),
                radius=6,
                fill=fill,
                outline=outline,
                width=2,
            )
            draw.text(
                (x + cell_width / 2, 375),
                display,
                anchor="mm",
                font=token_font(18, is_visible),
                fill=text_fill,
            )
        recovered = "".join(
            character if index in visible else "·" for index, character in enumerate(target)
        )
        draw.text((75, 435), recovered, font=token_font(25, True), fill="#f8fafc")
        action = (
            "initialize every position as [MASK]"
            if step == 0
            else "predict all masks in parallel → reveal the most confident subset"
        )
        if step == steps:
            action = "complete sequence — no masked positions remain"
        draw.text((75, 485), action, font=token_font(16), fill="#cbd5e1")
        draw.text(
            (55, 585),
            "Conceptual character-level view. The 106M TinyStories experiment "
            "uses GPT-2 BPE tokens.",
            font=token_font(15),
            fill="#94a3b8",
        )
        frames.append(image)
    frames[0].save(
        ASSET_DIR / "character_diffusion.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[1200] + [750] * (steps - 1) + [1800],
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    if args.refresh_data:
        refresh_data()
    data = json.loads(DATA_PATH.read_text())
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    render_curves(data)
    render_summary(data)
    render_decoding_gif()
    render_character_diffusion_gif()


if __name__ == "__main__":
    main()
