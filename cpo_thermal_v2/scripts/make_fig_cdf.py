"""
make_fig_cdf.py — Peak-T + makespan CDF comparison across schedulers.

Two side-by-side CDF panels (Decima SIGCOMM 2019 Fig 15 style):
  (a) Per-episode peak temperature CDF — shows how often each
      scheduler keeps the chip below T_pen=80°C.
  (b) Per-episode total makespan CDF — shows the full distribution
      of completion times, not just the mean.

Data source: eval_results/_phaseG/master.csv (grand_matrix +
grand_matrix_extras) filtered to hot ambient, N=17, n=500 per
scheduler.

Output: draft/figures/section5/fig_cdf.{pdf,svg}

Style: Nature structure + MDPI serif (inherited).

Usage:
  python -m cpo_thermal_v2.scripts.make_fig_cdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cpo_thermal_v2.scripts.make_paper_figures_final import set_paper_style, PAL


OUT_DIR = Path("draft/figures/section5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# CDF-specific palette + linestyle.
#
# For this 9-line figure we OVERRIDE the standard family-grouped
# palette because three near-same hues in the cool-blue / brown / red
# families muddied together at print scale. We pick 9 distinct hues
# (each scheduler has a unique colour) PLUS varied linestyles within
# each family so the curves remain identifiable in greyscale as well.
SCHEDULERS = [
    # (csv_name, display_label, color, action_mode, linestyle)
    # Baseline / classical family (cool blues + gray)
    ("HEFT",            "HEFT",            "#1f5e9c",   "auto_only", "-"),
    ("ThermalHEFT",     "Thermal-HEFT",    "#5fa3d8",   "auto_only", "--"),
    ("Decima-vanilla",  "Decima-vanilla",  "#3a3a3a",   "auto_only", ":"),
    ("Decima-thermal",  "Decima-thermal",  "#9a9a9a",   "auto_only", "-"),
    # GNN-prior family (warm browns + olive)
    ("HGATE-PPO",       "HGATE-PPO",       "#7b3f1e",   "auto_only", "-"),
    ("D2",              "D2",              "#e07b39",   "auto_only", "--"),
    # Ours ablation family (yellow -> red gradient + distinct styles)
    ("Ours-NoThermal",  "Ours-NoThermal",  "#c8932b",   "auto_only", "-"),
    ("Ours-auto_only",  "Ours-auto_only",  "#a8423a",   "auto_only", "--"),
    ("Ours-hybrid",     "Ours-hybrid",     "#5c1f1f",   "hybrid",    "-"),
]


def _save_pdf_and_svg(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.svg")
    print(f"  wrote {OUT_DIR / stem}.{{pdf,svg}}")


def _add_panel_label(ax: plt.Axes, label: str,
                      x: float = -0.12, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            ha="right", va="bottom")


def _cdf(values: np.ndarray):
    """Return (sorted_x, cdf_y) suitable for plt.step."""
    x_sorted = np.sort(np.asarray(values))
    n = len(x_sorted)
    if n == 0:
        return np.array([]), np.array([])
    y = np.arange(1, n + 1) / n
    return x_sorted, y


def main() -> None:
    set_paper_style()
    df = pd.read_csv("eval_results/_phaseG/master.csv")
    df = df[(df.source.isin(["grand_matrix", "grand_matrix_extras"])) &
            (df.num_nodes == 17) &
            (df.ambient == "hot")]

    # Slightly taller figure + bottom margin reserved for the
    # figure-level legend below both panels.
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.6),
                              gridspec_kw=dict(wspace=0.30,
                                                bottom=0.26,
                                                top=0.88,
                                                left=0.08,
                                                right=0.96))
    ax_a, ax_b = axes

    for csv_name, label, color, action_mode, ls in SCHEDULERS:
        sub = df[(df.scheduler == csv_name) &
                 (df.action_mode == action_mode)]
        if len(sub) == 0:
            print(f"  WARN: no data for {csv_name} (action_mode={action_mode})")
            continue
        # (a) Peak-T CDF
        xs, ys = _cdf(sub.peak_temp_episode.values)
        ax_a.step(xs, ys, where="post", color=color,
                  linewidth=1.5, linestyle=ls, label=label)
        # (b) Makespan CDF
        xs, ys = _cdf(sub.total_makespan_ms.values)
        ax_b.step(xs, ys, where="post", color=color,
                  linewidth=1.5, linestyle=ls, label=label)

    # ----- Panel (a): Peak T -----
    ax_a.axvline(80, color="#c0392b", linestyle="--",
                 linewidth=0.95, alpha=0.8)
    ax_a.axvspan(80, 100, color="#c0392b", alpha=0.06, zorder=0)
    # T_pen label: placed at the BOTTOM of the chart inside the
    # unsafe-shaded band — at x=81, y~0.05 the HEFT family curves are
    # still at 0 (their CDFs don't lift until x>85), so this region
    # is empty regardless of the bimodal Decima-thermal / D2 curves
    # above. White bounding box gives extra contrast.
    ax_a.text(81, 0.04, r"$T_{\mathrm{pen}}{=}80\,^\circ$C",
              fontsize=7.5, color="#c0392b",
              ha="left", va="bottom", rotation=90,
              bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                         edgecolor="none", alpha=0.92))
    ax_a.set_xlim(60, 100)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_xlabel("Peak temperature per episode ($^\\circ$C)")
    ax_a.set_ylabel("Empirical CDF")
    _add_panel_label(ax_a, "a")
    ax_a.text(0.5, 1.04, "Peak-temperature distribution",
              transform=ax_a.transAxes,
              fontsize=9.5, ha="center", va="bottom", fontweight="bold")
    ax_a.grid(True, which="major", linestyle=":",
              linewidth=0.4, alpha=0.30)

    # ----- Panel (b): Makespan -----
    ax_b.set_xlim(0, 9000)
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_xlabel("Total makespan per episode (ms)")
    _add_panel_label(ax_b, "b")
    ax_b.text(0.5, 1.04, "Makespan distribution",
              transform=ax_b.transAxes,
              fontsize=9.5, ha="center", va="bottom", fontweight="bold")
    ax_b.grid(True, which="major", linestyle=":",
              linewidth=0.4, alpha=0.30)
    # Shared legend placed BELOW both panels (3 cols x 3 rows) so it
    # never overlaps with the CDF curves inside either panel.
    handles, labels_ = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels_,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.03),
               frameon=False, fontsize=8,
               ncol=3, handlelength=2.4, columnspacing=1.8)

    _save_pdf_and_svg(fig, "fig_cdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
