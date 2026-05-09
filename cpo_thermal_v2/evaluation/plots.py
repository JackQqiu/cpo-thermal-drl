"""
evaluation/plots.py — Publication-grade plots from evaluation results
======================================================================

Three plot types, one CSV path → all figures.

1. **Scaling curves** (figure 1 of the paper): for each metric (makespan,
   peak temp, violation rate), plot mean ± 95% CI vs num_nodes for
   each scheduler.  This is the "does it generalise?" plot.

2. **Box plots** (figure 2): per-metric, per-size, distribution across
   episodes.  This is the "consistency?" plot.

3. **Per-DAG ρ analysis** (figure 3): scatter or 2D histogram of
   makespan_normalized vs ρ (communication-to-computation ratio),
   split by scheduler.  Shows whether RL helps more on
   communication-heavy DAGs.

All plots use matplotlib with conservative settings (Times font, small
axis labels) suitable for IEEE TPDS / TC submission.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# Style — conservative for IEEE journals
# =====================================================================
def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         10,
        "axes.titlesize":    10,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


# Paper-matched palette: cool blues for classical, warm reds for thermally-aware
_PALETTE = {
    # Classical heuristics — cool blues / gray
    "RoundRobin":                "#9CB4D8",     # light steel blue (was #888888 mid-gray)
    "HEFT":                      "#6789A8",     # mid steel blue (was #1f77b4 tab:blue)
    "ThermalHEFT":               "#3E5778",     # dark steel blue (was #ff7f0e tab:orange)
    "Thermal-HEFT":              "#3E5778",     # alternative label spelling alias

    # Throttled-HEFT (HK-2.1 new) — distinct steel blue family member
    "Throttled-HEFT-hybrid":     "#5A7090",
    "Throttled-HEFT-agent_only": "#7A8DA8",

    # Ours family — warm gold / red / burgundy gradient
    "Ours-NoThermal":            "#D4A24C",     # warm gold (matches paper figure)
    "Ours-auto_only":            "#C76B5A",     # coral red
    "Ours-auto":                 "#C76B5A",     # paper figure label alias
    "Ours-agent_only":           "#A04A5C",     # mid burgundy
    "Ours-hybrid":               "#7A2E3B",     # deep burgundy (matches paper)

    # Future Stage 2/3 baselines (placeholders, distinct hue families)
    "Decima-true":               "#5C8A3A",     # forest green (Stage 2 future)
    "HGATE-PPO":                 "#8A5C3A",     # warm brown (Stage 3 future)
}

_THRESHOLD_COLORS = {
    "T_pen":  "#C9A227",   # mustard yellow — warning / soft wall
    "T_crit": "#A1202F",   # dark red — critical / hard wall
}
# Reference lines, NOT in _PALETTE to avoid scheduler-color collision.


def _color(scheduler: str) -> str:
    return _PALETTE.get(scheduler, "#000000")


# =====================================================================
# 1. Scaling curves
# =====================================================================
def plot_scaling_curves(
    df: pd.DataFrame,
    out_dir: str,
    metrics: Optional[List[str]] = None,
    legend_loc: Optional[str] = None,
) -> None:
    """For each metric, plot one curve per scheduler vs num_nodes.

    Output: ``<out_dir>/scaling_<metric>.pdf`` and ``.png``.
    """
    _set_paper_style()
    os.makedirs(out_dir, exist_ok=True)
    if metrics is None:
        metrics = [
            "total_makespan_ms",
            "peak_temp_episode",
            "violations_total",
            "cooling_total_ms",
            "mean_makespan_normalized",
        ]

    schedulers = sorted(df["scheduler"].unique())

    for metric in metrics:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        for sched in schedulers:
            sub = df[df["scheduler"] == sched].dropna(subset=[metric])
            if sub.empty:
                continue
            agg = sub.groupby("num_nodes")[metric].agg(
                ["mean", "std", "count"]
            ).reset_index()
            # 95% CI: 1.96 σ / √n
            ci = 1.96 * agg["std"] / np.sqrt(np.maximum(1, agg["count"]))

            ax.errorbar(
                agg["num_nodes"], agg["mean"], yerr=ci,
                marker="o", linestyle="-",
                markersize=5, linewidth=1.5,
                color=_color(sched),
                label=sched,
                capsize=4, elinewidth=1.0,
            )

        ax.set_xlabel("Number of processors (N)")
        ax.set_ylabel(_pretty_metric(metric))
        ax.grid(True, alpha=0.3)
        if legend_loc is None:
            # Default: legend outside the plot (paper-figure friendly)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      borderaxespad=0., frameon=True, fontsize=8)
        else:
            ax.legend(frameon=True, loc=legend_loc, ncol=1, fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"scaling_{metric}.{ext}"))
        plt.close(fig)


# =====================================================================
# 2. Box plots
# =====================================================================
def plot_box_per_metric(
    df: pd.DataFrame,
    out_dir: str,
    metrics: Optional[List[str]] = None,
    num_nodes_focus: Optional[int] = None,
    group_annotation: Optional[Dict[str, List[str]]] = None,
) -> None:
    """One box plot per (metric, num_nodes) cell, comparing all schedulers.

    If ``num_nodes_focus`` is set, only that size is plotted (single
    figure per metric).  Otherwise one figure per (metric, num_nodes)
    pair — the paper might show only N=17, with the rest in appendix.
    """
    _set_paper_style()
    os.makedirs(out_dir, exist_ok=True)
    if metrics is None:
        metrics = ["total_makespan_ms", "peak_temp_episode",
                   "violations_total", "cooling_total_ms"]

    schedulers = sorted(df["scheduler"].unique())
    sizes = ([num_nodes_focus] if num_nodes_focus is not None
             else sorted(df["num_nodes"].unique()))

    for metric in metrics:
        if metric not in df.columns:
            continue
        for N in sizes:
            sub = df[df["num_nodes"] == N]
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            data    = []
            labels  = []
            colors  = []
            for sched in schedulers:
                vals = sub[sub["scheduler"] == sched][metric].dropna().values
                if len(vals) == 0:
                    continue
                data.append(vals)
                labels.append(sched)
                colors.append(_color(sched))

            bp = ax.boxplot(
                data, labels=labels, patch_artist=True,
                showfliers=False, widths=0.55,
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.85)
                patch.set_edgecolor("black")
            for med in bp["medians"]:
                med.set_color("black")
                med.set_linewidth(1.2)

            # Threshold lines for peak_temp_episode (paper §3 T_pen=80, T_crit=85)
            if metric == "peak_temp_episode":
                ax.axhline(80.0, ls="--", color=_THRESHOLD_COLORS["T_pen"],
                           alpha=0.7, linewidth=1.0)
                ax.axhline(85.0, ls="--", color=_THRESHOLD_COLORS["T_crit"],
                           alpha=0.7, linewidth=1.0)
                xmax = ax.get_xlim()[1]
                ax.text(xmax * 0.99, 80.4, r"$T_{\mathrm{pen}}=80\,°C$",
                        ha="right", va="bottom", fontsize=8,
                        color=_THRESHOLD_COLORS["T_pen"])
                ax.text(xmax * 0.99, 85.4, r"$T_{\mathrm{crit}}=85\,°C$",
                        ha="right", va="bottom", fontsize=8,
                        color=_THRESHOLD_COLORS["T_crit"])

            ax.set_ylabel(_pretty_metric(metric))
            ax.set_title(f"N = {N}")

            # Optional group annotations along the top axis
            if group_annotation:
                label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
                for grp_name, grp_members in group_annotation.items():
                    idxs = [label_to_idx[m] for m in grp_members
                            if m in label_to_idx]
                    if not idxs:
                        continue
                    center_x = (min(idxs) + max(idxs)) / 2 + 1
                    ax.text(center_x, 1.02, grp_name,
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom",
                            fontstyle="italic", fontsize=9, color="#444444")

            ax.grid(True, alpha=0.3, axis="y")
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out_dir, f"box_{metric}_N{N}.{ext}"))
            plt.close(fig)


# =====================================================================
# 3. ρ-analysis: makespan normalized vs DAG ρ
# =====================================================================
def plot_rho_analysis(
    df_dags: pd.DataFrame,
    out_dir: str,
    schedulers_to_plot: Optional[List[str]] = None,
    num_nodes_focus:    int = 17,
) -> None:
    """⚠ DEPRECATED CANDIDATE: not currently referenced in draft__6_.tex.
    Review before paper revision; remove if confirmed unused.

    Scatter of normalised makespan vs DAG ρ (communication-bound proxy).

    Goal: visualise whether the RL agent's advantage is uniform across
    the DAG distribution or concentrated on certain ρ ranges.
    """
    _set_paper_style()
    os.makedirs(out_dir, exist_ok=True)
    if "rho" not in df_dags.columns:
        return

    schedulers = (schedulers_to_plot
                  if schedulers_to_plot
                  else sorted(df_dags.get("scheduler_label",
                                           df_dags.get("scheduler",
                                            pd.Series())).dropna().unique()))
    if not schedulers:
        return

    sub = df_dags[df_dags.get("num_nodes_cell",
                              df_dags.get("num_nodes",
                               pd.Series([0]))) == num_nodes_focus]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for sched in schedulers:
        s = sub[sub.get("scheduler_label",
                        sub.get("scheduler", pd.Series())) == sched]
        if s.empty:
            continue
        # Bin by ρ and plot per-bin median + IQR
        rho_bins = np.linspace(0, max(2.5, s["rho"].max()), 8)
        s = s.copy()
        s["rho_bin"] = pd.cut(s["rho"], rho_bins, include_lowest=True)
        agg = s.groupby("rho_bin", observed=True)["makespan_normalized"].agg(
            ["median", "count"]).reset_index()
        agg = agg[agg["count"] >= 5]    # drop sparse bins
        if agg.empty:
            continue
        rho_centers = [(b.left + b.right) / 2 for b in agg["rho_bin"]]
        ax.plot(rho_centers, agg["median"], marker="o", linestyle="-",
                color=_color(sched), label=sched)

    ax.set_xlabel(r"DAG $\rho$ (comm-to-compute ratio)")
    ax.set_ylabel("Normalised makespan (median)")
    ax.set_title(f"DAG-level analysis at N={num_nodes_focus}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"rho_analysis_N{num_nodes_focus}.{ext}"))
    plt.close(fig)


# =====================================================================
# 4. Per-proc utilisation heatmap
# =====================================================================
def plot_proc_utilisation(
    df: pd.DataFrame,
    out_dir: str,
    num_nodes_focus: int = 17,
) -> None:
    """⚠ DEPRECATED CANDIDATE: not currently referenced in draft__6_.tex.
    Review before paper revision; remove if confirmed unused.

    Heatmap of mean per-proc utilisation across schedulers.

    Reveals e.g. that HEFT pushes everything to ASIC (proc 0) while
    Ours-hybrid spreads load across OEs.
    """
    _set_paper_style()
    os.makedirs(out_dir, exist_ok=True)
    sub = df[df["num_nodes"] == num_nodes_focus]
    if sub.empty or "proc_utilisation" not in sub.columns:
        return

    schedulers = sorted(sub["scheduler"].unique())
    util_matrix = []
    for sched in schedulers:
        s = sub[sub["scheduler"] == sched]
        utils = s["proc_utilisation"].dropna()
        if utils.empty:
            util_matrix.append([0.0] * num_nodes_focus)
            continue
        # Average per-proc utilisation across episodes
        stacked = np.array([np.asarray(u) for u in utils
                             if len(u) == num_nodes_focus])
        if stacked.size == 0:
            util_matrix.append([0.0] * num_nodes_focus)
        else:
            util_matrix.append(stacked.mean(axis=0).tolist())

    util_matrix = np.array(util_matrix)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    im = ax.imshow(util_matrix, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=util_matrix.max())
    ax.set_yticks(range(len(schedulers)))
    ax.set_yticklabels(schedulers)
    ax.set_xticks(range(num_nodes_focus))
    ax.set_xticklabels(["ASIC"] + [f"OE{i}" for i in range(num_nodes_focus - 1)],
                        rotation=90, fontsize=7)
    ax.set_xlabel("Processor")
    ax.set_title(f"Mean proc utilisation (N={num_nodes_focus})")
    fig.colorbar(im, ax=ax, label="Fraction busy")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"proc_util_N{num_nodes_focus}.{ext}"))
    plt.close(fig)


# =====================================================================
# Helpers
# =====================================================================
def _pretty_metric(name: str) -> str:
    return {
        "total_makespan_ms":         "Total makespan (ms)",
        "peak_temp_episode":         "Peak temperature (\u00b0C)",
        "mean_temp_episode":         "Mean temperature (\u00b0C)",
        "violations_total":          "Violations (count)",
        "cooling_total_ms":          "Cooling overhead (ms)",
        "agent_delay_total_ms":      "Agent-controlled delay (ms)",
        "dag_completion_rate":       "DAG completion rate",
        "mean_makespan_normalized":  "Normalised makespan (vs lower bound)",
        "episode_return":            "Episode return",
    }.get(name, name)
