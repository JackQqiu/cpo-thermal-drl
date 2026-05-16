"""
make_fig_pareto.py — Pareto plot: mean makespan vs viol_rate.

Visualises the safety-performance trade-off of all nine evaluated
schedulers at hot N=17 (the headline regime of Table 1 / fig:chain-viol).
A scheduler is Pareto-optimal if no other scheduler is simultaneously
lower-makespan AND lower viol_rate.

Output:
  draft/figures/section5/fig_pareto.{pdf,svg}

Style: Nature structure + MDPI serif (matches make_paper_figures_final).

Usage:
  python -m cpo_thermal_v2.scripts.make_fig_pareto
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cpo_thermal_v2.scripts.make_paper_figures_final import set_paper_style, PAL


OUT_DIR = Path("draft/figures/section5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_pdf_and_svg(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.svg")
    print(f"  wrote {OUT_DIR / stem}.{{pdf,svg}}")


# Data: hot N=17, n=500, from Table 1 (main results) of the paper.
# Each entry: (scheduler_name, mean_makespan_ms, viol_rate, completion_rate,
#              palette_key, label_dx_pt, label_dy_pt, ha)
DATA = [
    # Format: (name, mk, viol, comp, palette_key, dx_pt, dy_pt, ha)
    # Labels placed in staggered slots so no label overlaps another
    # label or the chart title. y-coords are slightly perturbed for the
    # near-coincident Ours-auto_only / Ours-hybrid pair so both markers
    # AND labels remain individually identifiable.
    # Left cluster (HEFT family near 1200-2000 ms, 0.97-0.99 viol).
    # Four labels staggered: HEFT below-left, Throttled-HEFT above,
    # Thermal-HEFT above (between Throttled and HGATE clusters),
    # HGATE-PPO below-right.
    ("HEFT",              1243, 0.986, 0.260, "heft",            -10, -18, "right"),
    ("Throttled-HEFT",    1278, 0.984, 0.261, "throttled_heft",  -8,  +16, "right"),
    ("Thermal-HEFT",      1697, 0.968, 0.337, "thermal_heft",    +12, +14, "left"),
    ("HGATE-PPO",         2023, 0.968, 0.401, "hgate",           +12, -18, "left"),
    # Middle band
    ("D2",                3664, 0.478, 0.697, "d2",              -12, +12, "right"),
    ("Decima-thermal",    3958, 0.382, 0.741, "decima_t",        +12, +6,  "left"),
    # Right cluster — Ours family near 5300 ms, viol 0.002-0.006:
    ("Ours-NoThermal",    5317, 0.006,  1.000, "nothermal",     -12, +5,  "right"),
    # Ours-auto_only and Ours-hybrid are near-coincident (5316/5317 ms,
    # both viol=0.002). Nudge them vertically apart by 5% so both
    # markers and labels remain visible.
    ("Ours-auto_only",    5317, 0.0024, 1.000, "auto_only",     +12, +5,  "left"),
    ("Ours-hybrid",       5316, 0.0017, 1.000, "hybrid",        +12, -18, "left"),
]


def compute_pareto_front(points):
    """Pareto-optimal points (minimising both makespan AND viol_rate)."""
    front = []
    for i, (name_i, mk_i, vi_i, *_) in enumerate(points):
        dominated = False
        for j, (name_j, mk_j, vi_j, *_) in enumerate(points):
            if i == j:
                continue
            # j dominates i if j is no worse in both AND strictly better in at least one
            if (mk_j <= mk_i and vi_j <= vi_i and
                (mk_j < mk_i or vi_j < vi_i)):
                dominated = True
                break
        if not dominated:
            front.append((name_i, mk_i, vi_i))
    # Sort by makespan ascending so we can connect them with a frontier line
    front.sort(key=lambda x: x[1])
    return front


def main() -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    # Identify Pareto-front members for highlighting + line connection
    pareto_front = compute_pareto_front(DATA)
    pareto_names = {n for n, _, _ in pareto_front}

    # Scatter every point. Size encodes DAG completion rate.
    for name, mk, viol, comp, pal_key, dx, dy, ha in DATA:
        on_front = name in pareto_names
        size = 60 + 280 * comp        # 60..340
        ax.scatter(mk, viol,
                   s=size,
                   c=PAL[pal_key],
                   edgecolor="black" if on_front else "#555555",
                   linewidth=1.3 if on_front else 0.5,
                   alpha=0.95 if on_front else 0.70,
                   zorder=3)

        # Label next to each point (offset by dx/dy points)
        ax.annotate(
            name, xy=(mk, viol),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8, fontweight="bold" if on_front else "normal",
            color="black" if on_front else "#444444",
            ha=ha,
        )

    # Draw Pareto frontier (red staircase, sorted by makespan ascending)
    pareto_mk   = [mk   for _, mk, _ in pareto_front]
    pareto_viol = [viol for _, _, viol in pareto_front]
    ax.plot(pareto_mk, pareto_viol,
            color="#c0392b", linewidth=1.5,
            linestyle="--", alpha=0.7,
            label="Pareto frontier",
            zorder=2)

    # Annotate the "ideal corner" (low makespan + low viol — bottom-left)
    ax.scatter([1200], [0.001], s=140, marker="*",
               c="#c0392b", edgecolor="black", linewidth=0.5,
               zorder=4, label="ideal corner")
    ax.annotate(
        "ideal",
        xy=(1200, 0.001), xytext=(-4, 4),
        textcoords="offset points",
        fontsize=8, fontweight="bold", color="#c0392b",
        ha="right",
    )

    # Axes
    ax.set_xlim(800, 6900)
    ax.set_ylim(0.0008, 1.6)   # extra headroom above 1.0 for top labels
    ax.set_yscale("log")
    ax.set_xlabel("Mean makespan (ms) $\\,\\downarrow$")
    ax.set_ylabel("Violation rate $\\,\\downarrow$ (log scale)")
    ax.set_title("Safety vs makespan trade-off at hot $N{=}17$ "
                 "($n{=}500$ paired episodes)",
                 fontsize=9.5, pad=8)

    # Subtle grid for the log y-axis
    ax.grid(True, which="major", axis="y", linestyle=":",
            linewidth=0.4, alpha=0.35)

    # Annotation: marker size = DAG completion rate
    # Anchored top-right (well above Pareto frontier where there's no data)
    ax.text(0.985, 0.97,
            "Marker size $\\propto$ DAG completion rate\n"
            "Filled border $\\Rightarrow$ on Pareto frontier",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="black", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                       edgecolor="#bbbbbb", linewidth=0.4))

    _save_pdf_and_svg(fig, "fig_pareto")
    plt.close(fig)


if __name__ == "__main__":
    main()
