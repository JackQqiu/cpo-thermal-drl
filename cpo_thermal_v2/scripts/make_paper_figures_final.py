"""
make_paper_figures_final.py — §5 paper figures, Nature-structure + MDPI-fonts hybrid.

Produces publication-grade PDFs (and editable SVG counterparts) in
draft/figures/section5/:
  fig_chain_viol.pdf / .svg        — §5.3 six-component chain bar
  fig_envelope_heatmap.pdf / .svg  — §5.4 5N × 4 ambient viol heatmap
  fig_autocool_peak.pdf / .svg     — §5.6 paired peak-T violins

Style policy:
  - serif typography (Palatino / Times New Roman fallback) to match
    the MDPI Electronics body font;
  - Nature-style structural conventions: panel labels (a, b) on
    multi-panel figures, tight y-limits, semantic palette grouped by
    method family, frameon=False legends, top/right spines hidden,
    minimal gridlines, svg.fonttype='none' so the text in SVG stays
    selectable.

Usage:
  python -m cpo_thermal_v2.scripts.make_paper_figures_final
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker


OUT_DIR = Path("draft/figures/section5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_paper_style() -> None:
    """Nature-style structural rcParams with MDPI serif typography."""
    plt.rcParams.update({
        "font.family":         "serif",
        "font.serif":          ["Palatino", "Times New Roman", "Times",
                                 "DejaVu Serif"],
        "mathtext.fontset":    "stix",
        "font.size":           9,
        "axes.titlesize":      9,
        "axes.labelsize":      9,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     8,
        "figure.dpi":          120,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      0.7,
        "grid.linewidth":      0.4,
        "grid.alpha":          0.30,
        "lines.linewidth":     1.3,
        "svg.fonttype":        "none",   # editable text in SVG
        "pdf.fonttype":         42,       # embed TrueType (editable in
                                          # Illustrator / Inkscape)
    })


# ---------------------------------------------------------------------------
# Semantic palette grouped by method family.
#
#   Baseline (classical heuristics + Decima):  cool slate-blue + neutral gray
#   GNN-prior  (HGATE-PPO, D2):                warm brown
#   Ours ablation:                              hot-red gradient with the
#                                               deepest red reserved for the
#                                               headline / deployed policy
# ---------------------------------------------------------------------------
PAL = {
    # baseline (cool blue-gray)
    "heft":            "#5b7a99",
    "thermal_heft":    "#6e8aa8",
    "throttled_heft":  "#809ab6",
    "round_robin":     "#8aa1b8",
    "decima_v":        "#7a7a85",
    "decima_t":        "#9a9aa3",
    # GNN-prior (warm brown)
    "hgate":           "#a47148",
    "d2":              "#c87f47",
    # Ours ablation (red gradient)
    "nothermal":       "#c8932b",   # ochre — secondary RC-edge ablation
    "norcedge":        "#d65644",   # contrast red
    "auto_only":       "#a8423a",   # mid red
    "hybrid":          "#7a1f1f",   # deepest red, hero
}


def _save_pdf_and_svg(fig: plt.Figure, stem: str) -> None:
    """Save figure as both PDF (for paper inclusion) and SVG (editable)."""
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.svg")
    print(f"  wrote {OUT_DIR / stem}.{{pdf,svg}}")


def _add_panel_label(ax: plt.Axes, label: str,
                      x: float = -0.08, y: float = 1.02) -> None:
    """Nature-style panel label (bold lowercase) at top-left in axes coords."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            ha="right", va="bottom")


# =============================================================================
# Fig 3 — §5.3 6-component chain bar
# =============================================================================
def fig_chain_viol() -> None:
    """Bar chart of viol_rate across the eight chain configurations."""
    schedulers = [
        ("HEFT",            0.986,  "ref",  "heft"),
        ("Decima-vanilla",  1.000,  "*",    "decima_v"),
        ("Decima-thermal",  0.382,  "***",  "decima_t"),
        ("HGATE-PPO",       0.968,  "***",  "hgate"),
        ("D2",              0.478,  "*",    "d2"),
        ("Ours-NoThermal",  0.006,  "***",  "nothermal"),
        ("Ours-auto_only",  0.002,  "ns",   "auto_only"),
        ("Ours-hybrid",     0.002,  "—",    "hybrid"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    labels = [s[0] for s in schedulers]
    rates  = [s[1] for s in schedulers]
    sigs   = [s[2] for s in schedulers]
    colors = [PAL[s[3]] for s in schedulers]
    x = np.arange(len(schedulers))
    bars = ax.bar(x, rates, color=colors, edgecolor="black",
                  linewidth=0.45, width=0.70)

    # Significance star + numeric value above each bar
    for i, (b, r, sig) in enumerate(zip(bars, rates, sigs)):
        # value label
        if r >= 0.05:
            ax.text(i, b.get_height() + 0.018, f"{r:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
        else:
            ax.text(i, b.get_height() + 0.025, f"{r:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
        # significance star ABOVE the numeric value
        if sig in ("*", "**", "***"):
            ax.text(i, b.get_height() + 0.07, sig, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="black")
        elif sig == "ns":
            ax.text(i, b.get_height() + 0.06, "n.s.", ha="center", va="bottom",
                    fontsize=7, color="#888888", style="italic")

    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Violation rate (hot $N{=}17$, $n{=}500$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}"))

    # Subtle horizontal guide at 1.0 for the upper bound
    ax.axhline(1.0, color="#cccccc", linewidth=0.4, linestyle=":")

    # Caption-like annotation inside the chart area
    ax.text(0.99, 0.97,
            "Paired McNemar vs previous chain link\n"
            "Holm-Bonferroni, six tests, $\\alpha{=}0.05$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="black", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                       edgecolor="#bbbbbb", linewidth=0.45))

    _save_pdf_and_svg(fig, "fig_chain_viol")
    plt.close(fig)


# =============================================================================
# Fig 4 — §5.4 envelope heatmap (5N × 4 ambient), 2-panel HEFT vs Ours-hybrid
# =============================================================================
def fig_envelope_heatmap() -> None:
    df = pd.read_csv("eval_results/_phaseG/master.csv")
    df = df[df["source"].isin(["grand_matrix", "grand_matrix_extras"])]

    sizes  = [9, 13, 17, 24, 33]
    ambs   = ["cold", "warm", "hot", "extreme"]
    amb_labels = [r"cold $[25,40]$", r"warm $[40,65]$",
                  r"hot $[60,75]$", r"extreme $[70,80]$"]

    def grid_for(sched_label: str, action_mode: str = "auto_only") -> np.ndarray:
        g = np.full((len(sizes), len(ambs)), np.nan)
        sub = df[(df["scheduler"] == sched_label) &
                 (df["action_mode"] == action_mode)]
        for i, n in enumerate(sizes):
            for j, a in enumerate(ambs):
                cell = sub[(sub["num_nodes"] == n) &
                           (sub["ambient"]   == a)]
                if len(cell) > 0:
                    g[i, j] = cell["is_unsafe"].mean()
        return g

    grid_heft = grid_for("HEFT", action_mode="auto_only")
    grid_ours = grid_for("Ours-hybrid", action_mode="hybrid")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.95),
                              gridspec_kw=dict(wspace=0.32, left=0.10,
                                                right=0.92, top=0.86,
                                                bottom=0.20))

    for ax, grid, panel_lbl, title in (
        (axes[0], grid_heft, "a", "HEFT"),
        (axes[1], grid_ours, "b", "Ours-hybrid"),
    ):
        im = ax.imshow(grid, aspect="auto", cmap="Reds",
                       vmin=0, vmax=1, origin="lower")
        ax.set_xticks(np.arange(len(ambs)))
        ax.set_xticklabels(amb_labels, rotation=18, ha="right")
        ax.set_yticks(np.arange(len(sizes)))
        ax.set_yticklabels([f"$N{{=}}{n}$" for n in sizes])
        # Panel label (Nature style) top-left in axes coords
        ax.text(-0.18, 1.05, panel_lbl, transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="right", va="bottom")
        # Inline title (right of the panel label)
        ax.text(0.5, 1.05, title, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="center", va="bottom")
        # Cell values
        for i in range(len(sizes)):
            for j in range(len(ambs)):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                txt_color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=txt_color)

    cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.04, shrink=0.85)
    cbar.set_label("Violation rate")
    cbar.ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Processor count $N$")
    _save_pdf_and_svg(fig, "fig_envelope_heatmap")
    plt.close(fig)


# =============================================================================
# Fig 6 — §5.6 auto-cool ablation paired peak-T violins
# =============================================================================
def fig_autocool_peak() -> None:
    """Violin plot of per-episode peak-T for 5 schedulers x 2 cool settings."""
    order = ["HEFT", "Decima-thermal", "Ours-NoThermal",
             "Ours-NoRCEdge", "Ours-auto_only"]

    # cool=100 peak T per scheduler
    cool100 = {}
    df_paired = pd.read_csv(
        "eval_results/ours_no_rc_edge_paired_hot_n17/paired.csv")
    cool100["Ours-auto_only"] = df_paired[
        df_paired.scheduler == "Ours-auto_only"]["peak_temp_episode"].values
    cool100["Ours-NoRCEdge"] = df_paired[
        df_paired.scheduler == "Ours-NoRCEdge"]["peak_temp_episode"].values

    df_gm = pd.read_csv("eval_results/_phaseG/master.csv")
    df_gm = df_gm[(df_gm.source.isin(["grand_matrix", "grand_matrix_extras"])) &
                  (df_gm.num_nodes == 17) &
                  (df_gm.ambient == "hot") &
                  (df_gm.action_mode == "auto_only")]
    cool100["HEFT"] = df_gm[
        df_gm.scheduler == "HEFT"]["peak_temp_episode"].values
    cool100["Decima-thermal"] = df_gm[
        df_gm.scheduler == "Decima-thermal"]["peak_temp_episode"].values
    cool100["Ours-NoThermal"] = df_gm[
        df_gm.scheduler == "Ours-NoThermal"]["peak_temp_episode"].values

    # cool=0
    cool0 = {}
    df_ablation = pd.read_csv(
        "eval_results/autocool_ablation_hot_n17/paired.csv")
    for s in ["HEFT", "Decima-thermal", "Ours-NoRCEdge", "Ours-auto_only"]:
        cool0[s] = df_ablation[
            df_ablation.scheduler == s]["peak_temp_episode"].values
    df_nt0 = pd.read_csv("eval_results/nothermal_nocool/paired.csv")
    cool0["Ours-NoThermal"] = df_nt0["peak_temp_episode"].values

    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    palette = {
        "HEFT":           PAL["heft"],
        "Decima-thermal": PAL["decima_t"],
        "Ours-NoThermal": PAL["nothermal"],
        "Ours-NoRCEdge":  PAL["norcedge"],
        "Ours-auto_only": PAL["auto_only"],
    }

    width = 0.36
    pos_left  = np.arange(len(order)) - width / 2
    pos_right = np.arange(len(order)) + width / 2

    for i, s in enumerate(order):
        c100 = cool100.get(s, np.array([]))
        c0   = cool0.get(s,   np.array([]))
        if len(c100) > 0:
            parts = ax.violinplot([c100], positions=[pos_left[i]],
                                   widths=width * 0.9, showmedians=True,
                                   showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(palette[s])
                pc.set_alpha(0.40)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.45)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(0.9)
        if len(c0) > 0:
            parts = ax.violinplot([c0], positions=[pos_right[i]],
                                   widths=width * 0.9, showmedians=True,
                                   showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(palette[s])
                pc.set_alpha(0.95)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.45)
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(0.9)

    # T_pen reference line — thicker, with shaded "unsafe" region
    ax.axhline(80, color="#c0392b", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhspan(80, 100, color="#c0392b", alpha=0.07, zorder=0)
    ax.text(len(order) - 0.4, 80.7,
            r"$T_{\mathrm{pen}}{=}80\,^\circ$C",
            fontsize=7.5, color="#c0392b", ha="right", va="bottom")

    ax.set_xticks(np.arange(len(order)))
    short_labels = ["HEFT", "Dec-t", "NoTherm", "NoRCEdge", "auto_only"]
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Peak temperature per episode ($^\\circ$C)")
    ax.set_ylim(58, 96)

    # Legend: light = cool=100, dark = cool=0
    handles = [
        mpatches.Patch(facecolor="gray", alpha=0.40, edgecolor="black",
                        linewidth=0.45,
                        label=r"$\mathtt{max\_cooling\_steps}{=}100$"
                              r" (standard env)"),
        mpatches.Patch(facecolor="gray", alpha=0.95, edgecolor="black",
                        linewidth=0.45,
                        label=r"$\mathtt{max\_cooling\_steps}{=}0$"
                              r" (env safety floor off)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.5)

    _save_pdf_and_svg(fig, "fig_autocool_peak")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    set_paper_style()
    print("Generating §5 paper figures (Nature structure + MDPI serif)...")
    fig_chain_viol()
    fig_envelope_heatmap()
    fig_autocool_peak()
    print("Done.")


if __name__ == "__main__":
    main()
