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


# Display order + palette key + action_mode for filtering grand_matrix
SCHEDULERS = [
    # name (matches CSV scheduler column), display_label, palette_key,
    #  action_mode, line_style
    ("HEFT",            "HEFT",            "heft",          "auto_only", "-"),
    ("ThermalHEFT",     "Thermal-HEFT",    "thermal_heft",  "auto_only", "-"),
    ("Decima-vanilla",  "Decima-vanilla",  "decima_v",      "auto_only", "-"),
    ("Decima-thermal",  "Decima-thermal",  "decima_t",      "auto_only", "-"),
    ("HGATE-PPO",       "HGATE-PPO",       "hgate",         "auto_only", "-"),
    ("D2",              "D2",              "d2",            "auto_only", "-"),
    ("Ours-NoThermal",  "Ours-NoThermal",  "nothermal",     "auto_only", "-"),
    ("Ours-auto_only",  "Ours-auto_only",  "auto_only",     "auto_only", "-"),
    ("Ours-hybrid",     "Ours-hybrid",     "hybrid",        "hybrid",    "-"),
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

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2),
                              gridspec_kw=dict(wspace=0.30))
    ax_a, ax_b = axes

    for csv_name, label, pal_key, action_mode, ls in SCHEDULERS:
        sub = df[(df.scheduler == csv_name) &
                 (df.action_mode == action_mode)]
        if len(sub) == 0:
            print(f"  WARN: no data for {csv_name} (action_mode={action_mode})")
            continue
        # (a) Peak-T CDF
        xs, ys = _cdf(sub.peak_temp_episode.values)
        ax_a.step(xs, ys, where="post", color=PAL[pal_key],
                  linewidth=1.4, linestyle=ls, label=label)
        # (b) Makespan CDF
        xs, ys = _cdf(sub.total_makespan_ms.values)
        ax_b.step(xs, ys, where="post", color=PAL[pal_key],
                  linewidth=1.4, linestyle=ls, label=label)

    # ----- Panel (a): Peak T -----
    ax_a.axvline(80, color="#c0392b", linestyle="--",
                 linewidth=0.95, alpha=0.8)
    ax_a.axvspan(80, 100, color="#c0392b", alpha=0.06, zorder=0)
    ax_a.text(80.5, 0.06, r"$T_{\mathrm{pen}}$",
              fontsize=8, color="#c0392b", ha="left", va="bottom",
              rotation=90)
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
    # Shared legend in panel b, outside the axes on the right
    ax_b.legend(loc="lower right", frameon=False, fontsize=7,
                ncol=1, handlelength=2.0, columnspacing=0.8)

    _save_pdf_and_svg(fig, "fig_cdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
