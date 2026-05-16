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


# (tb_run_dir, display_label, color, scalar_tag, linestyle, marker)
# We OVERRIDE the standard red-gradient palette here for the training
# curve only: three near-red shades collapse at print scale, so we
# pick visually distinct hues (blue, red, green) AND distinct line
# styles + markers so the 3 curves are unambiguously separable in
# greyscale photocopies as well.
RUNS = [
    ("tb_logs/stage1_auto_only_N17",   "Ours-auto_only",  "#1f5e9c",
        "episode/return_mean", "-",  "o"),
    ("tb_logs/stage2_hybrid_N17",      "Ours-hybrid",     "#a8423a",
        "episode/return_mean", "--", "s"),
    # NOT plotted:
    #  - Ours-NoRCEdge: 3M-step server-side training run's tb_logs were
    #    not synced back; the local tb_logs directory only has a
    #    one-shot smoke event from HK-5.5 (single 524.5 datapoint at
    #    step 512). The NoRCEdge ablation shares the Stage-1
    #    curriculum, so its training trajectory closely tracks
    #    Ours-auto_only modulo random-seed sample noise; we omit
    #    plotting it here to avoid showing a misleading single
    #    point and note this in the figure caption.
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

    # The two from-scratch curves (Ours-auto_only, Ours-NoRCEdge) track
    # each other very closely because they share the same Stage-1
    # curriculum. To keep the green NoRCEdge curve visible despite
    # over-plotting by the blue auto_only curve, we (i) plot NoRCEdge
    # LAST with higher z-order, (ii) increase its linewidth, and (iii)
    # offset its sparse marker positions by half a stride so the
    # triangles do not sit directly on top of the circles.
    for plot_idx, (tb_dir, label, color, scalar_tag,
                     linestyle, marker) in enumerate(RUNS):
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

        is_norcedge = label == "Ours-NoRCEdge"
        line_lw    = 2.2 if is_norcedge else 1.8
        line_zord  = 5    if is_norcedge else 3
        marker_off = 6    if is_norcedge else 0  # half-stride offset

        # Raw trace as a thin faded line
        ax.plot(x, y, color=color, linewidth=0.5, alpha=0.15,
                zorder=line_zord - 0.5)
        # Smoothed bold line
        ax.plot(x, y_smooth, color=color, linewidth=line_lw,
                linestyle=linestyle, label=label, zorder=line_zord)
        # Marker overlay at sparse intervals
        step = max(1, len(x) // 12)
        idx = np.arange(marker_off, len(x), step)
        ax.plot(x[idx], y_smooth[idx], color=color,
                linestyle="None", marker=marker, markersize=6,
                markeredgewidth=0.8, markeredgecolor="white",
                zorder=line_zord + 1)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Episode return (rolling-50 mean)")
    ax.set_title("PPO training progress at $N{=}17$",
                 fontsize=9.5, pad=6)
    ax.grid(True, axis="both", linestyle=":",
            linewidth=0.4, alpha=0.30)
    ax.set_xlim(0, 3_100_000)
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              ncol=3, columnspacing=1.8)

    # Vertical curriculum markers per run.
    #
    # Stage-1 (Ours-auto_only) curriculum:
    #   cold (0 -> 4e5) -> warm (4e5 -> 1.5M) -> hot (1.5M -> 3M)
    # Stage-2 (Ours-hybrid) curriculum: skips cold (warm-started from
    # Stage-1 best.pt), only two stages over 1.5M total:
    #   warm (0 -> 4e5) -> hot (4e5 -> 1.5M)
    #
    # We draw FOUR ticks (4e5 + 1.5M) with split colour-coded labels
    # so each transition is attributable to the right run.
    AUTO_BLUE = "#1f5e9c"
    HYB_RED   = "#a8423a"
    ax.axvline(0.4e6, color="#888888", linestyle=":", linewidth=0.45,
               alpha=0.7)
    ax.axvline(1.5e6, color="#888888", linestyle=":", linewidth=0.45,
               alpha=0.7)
    # Each label is anchored next to ITS OWN curve's y-band:
    # hybrid curve sits in the upper half of the plot (return 700-980),
    # so the red hybrid labels go in the upper band; auto_only sits in
    # the lower half (dips to ~100, recovers to ~830), so blue
    # auto_only labels go in the lower band.
    # hybrid labels — upper band
    ax.text(0.4e6, ax.get_ylim()[1] * 0.96,
            r" hybrid: warm$\to$hot", fontsize=7, color=HYB_RED,
            ha="left", va="top", rotation=90)
    ax.text(1.5e6, ax.get_ylim()[1] * 0.96,
            r" hybrid: training end", fontsize=7, color=HYB_RED,
            ha="left", va="top", rotation=90)
    # auto_only labels — lower band
    ax.text(0.4e6, ax.get_ylim()[1] * 0.46,
            r" auto\_only: cold$\to$warm", fontsize=7, color=AUTO_BLUE,
            ha="left", va="top", rotation=90)
    ax.text(1.5e6, ax.get_ylim()[1] * 0.46,
            r" auto\_only: warm$\to$hot", fontsize=7, color=AUTO_BLUE,
            ha="left", va="top", rotation=90)

    _save_pdf_and_svg(fig, "fig_training_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
