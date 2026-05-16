"""
make_fig_training_curve.py — training reward curves for the §4 method
disclosure (or a supplementary).

Reads the TensorBoard `episode/return_mean` scalar from each run that
has a tb_logs directory and plots them on a single panel, rolling-mean
smoothed. Matches the §5 figure styling (Nature structure + MDPI
serif).

Output: draft/figures/section5/fig_training_curves.{pdf,svg}

Usage:
  python -m cpo_thermal_v2.scripts.make_fig_training_curve
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
from cpo_thermal_v2.scripts.make_paper_figures_final import set_paper_style, PAL


OUT_DIR = Path("draft/figures/section5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# (tb_run_dir, display_label, palette_key, scalar_tag)
RUNS = [
    ("tb_logs/stage1_auto_only_N17",   "Ours-auto_only",  "auto_only", "episode/return_mean"),
    ("tb_logs/stage2_hybrid_N17",      "Ours-hybrid",     "hybrid",    "episode/return_mean"),
    ("tb_logs/ours_no_rc_edge_N17",    "Ours-NoRCEdge",   "norcedge",  "episode/return_mean"),
    # NOT included:
    #  - Decima-vanilla: REINFORCE trainer only writes ep_return at
    #    sparse 50-step intervals; <5 datapoints total across 4
    #    re-runs, not enough for a curve.
    #  - HGATE-PPO: Wu 2025 trainer logs loss/{entropy,policy,value}
    #    only — no return scalar.
]


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().values


def _save_pdf_and_svg(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.svg")
    print(f"  wrote {OUT_DIR / stem}.{{pdf,svg}}")


def main() -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.3))

    for tb_dir, label, pal_key, scalar_tag in RUNS:
        reader = SummaryReader(tb_dir)
        scalars = reader.scalars
        ret = scalars[scalars.tag == scalar_tag]
        if len(ret) == 0:
            print(f"  WARN: no {scalar_tag} in {tb_dir}")
            continue
        ret = ret.sort_values("step")
        x = ret.step.values
        y = ret.value.values
        y_smooth = _rolling_mean(y, window=50)
        # Raw trace as a thin faded line + smoothed bold line on top
        ax.plot(x, y, color=PAL[pal_key], linewidth=0.5, alpha=0.18)
        ax.plot(x, y_smooth, color=PAL[pal_key], linewidth=1.4,
                label=label)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Episode return (rolling-50 mean)")
    ax.set_title("PPO training progress at $N{=}17$",
                 fontsize=9.5, pad=6)
    ax.grid(True, axis="both", linestyle=":",
            linewidth=0.4, alpha=0.30)
    ax.set_xlim(0, 3_100_000)
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              ncol=3, columnspacing=1.8)

    # Vertical curriculum markers — only relevant to Ours-auto_only
    # (Stage-1 curriculum cold->warm at 4e5, warm->hot at 1.5e6)
    ax.axvline(0.4e6, color="#888888", linestyle=":", linewidth=0.45,
               alpha=0.7)
    ax.axvline(1.5e6, color="#888888", linestyle=":", linewidth=0.45,
               alpha=0.7)
    ax.text(0.4e6, ax.get_ylim()[1] * 0.95,
            "  cold$\\to$warm", fontsize=7, color="#888888",
            ha="left", va="top", rotation=90)
    ax.text(1.5e6, ax.get_ylim()[1] * 0.95,
            "  warm$\\to$hot", fontsize=7, color="#888888",
            ha="left", va="top", rotation=90)

    _save_pdf_and_svg(fig, "fig_training_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
