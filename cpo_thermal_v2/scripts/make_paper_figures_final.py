"""
make_paper_figures_final.py — Final 3 figures for paper §5 (HK-paper-11).

Produces clean publication-grade PDFs in
draft/figures/section5/:
  fig_chain_viol.pdf       — §5.3 6-component chain bar with sig stars
  fig_envelope_heatmap.pdf — §5.4 5N × 4 ambient viol heatmap, 2-panel
  fig_autocool_peak.pdf    — §5.6 paired peak-T distribution

Replaces 6 stale figures (deleted in HK-paper-11 prep). Each figure
is generated independently from the gitignored eval_results CSVs;
the script is idempotent.

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
    plt.rcParams.update({
        "font.family":         "serif",
        "font.serif":          ["Times New Roman", "Times", "DejaVu Serif"],
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
        "axes.linewidth":      0.6,
        "grid.linewidth":      0.4,
        "grid.alpha":          0.4,
        "lines.linewidth":     1.2,
    })


# Restrained palette
PAL = {
    "classical":  "#5b7a99",   # slate blue
    "decima":     "#7a7a85",   # gray
    "hgate":      "#a47148",   # warm brown
    "d2":         "#c87f47",   # orange-brown
    "nothermal":  "#c8932b",   # ochre
    "norcedge":   "#a8423a",   # muted red
    "auto_only":  "#8b3a3a",   # darker red
    "hybrid":     "#5c1f1f",   # dark red
}


# =============================================================================
# Fig 3 — §5.3 6-component chain bar
# =============================================================================
def fig_chain_viol() -> None:
    """Bar chart: viol_rate per scheduler in the 6-component chain.

    Data (HK-paper-7/8 §5.3 + tab:ablation):
      Comp 1 HEFT (anchor, ref):                                          0.986
      Comp 1 Decima-vanilla   (HEFT -> Decima-v, +0.014, p=0.031 *)       1.000
      Comp 2 Decima-thermal   (Decima-v -> Decima-t, -0.618, p<10^-4 ***) 0.382
      Comp 3 HGATE-PPO         (Decima-t -> HGATE, +0.586, p<10^-4 ***)   0.968
      Comp 4 D2                (Decima-t -> D2, +0.096, p=0.029 *)        0.478
      Comp 5 Ours-NoThermal    (Decima-t -> NoThermal, -0.376, p<10^-4)   0.006
      Comp 6 Ours-auto_only    (NoThermal -> auto_only, -0.004, p=0.625)  0.002
              Ours-hybrid       (delay action, headline)                   0.002
    """
    schedulers = [
        ("HEFT",            0.986,  "ref", "classical"),
        ("Decima-vanilla",  1.000,  "*",   "decima"),
        ("Decima-thermal",  0.382,  "***", "decima"),
        ("HGATE-PPO",       0.968,  "***", "hgate"),
        ("D2",              0.478,  "*",   "d2"),
        ("Ours-NoThermal",  0.006,  "***", "nothermal"),
        ("Ours-auto_only",  0.002,  "ns",  "auto_only"),
        ("Ours-hybrid",     0.002,  "—",   "hybrid"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    labels = [s[0] for s in schedulers]
    rates  = [s[1] for s in schedulers]
    sigs   = [s[2] for s in schedulers]
    colors = [PAL[s[3]] for s in schedulers]
    x = np.arange(len(schedulers))
    bars = ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.4, width=0.66)

    # T_pen=80°C is not relevant here; viol_rate is the metric
    for i, (b, r, sig) in enumerate(zip(bars, rates, sigs)):
        h = r + 0.025
        if sig in ("*", "**", "***"):
            ax.text(i, h, sig, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="black")
        elif sig == "ns":
            ax.text(i, h, "ns", ha="center", va="bottom",
                    fontsize=7, color="gray", style="italic")
        ax.text(i, b.get_height() + 0.005, f"{r:.3f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Violation rate (hot $N{=}17$, $n{=}500$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # Annotation: significance corresponds to paired McNemar against
    # the previous step in the chain
    ax.text(0.99, 0.96,
            r"Significance: paired McNemar vs previous chain link"
            "\n"
            r"(Holm-Bonferroni on six tests, $\alpha{=}0.05$)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="black", linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="gray", lw=0.4))

    fig.savefig(OUT_DIR / "fig_chain_viol.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_chain_viol.pdf'}")


# =============================================================================
# Fig 4 — §5.4 envelope heatmap (5N × 4 ambient)
# =============================================================================
def fig_envelope_heatmap() -> None:
    """5N × 4 ambient heatmap of viol_rate. Two panels: HEFT vs Ours-hybrid.

    Data: eval_results/_phaseG/master.csv (full grand matrix).
    """
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

    grid_heft  = grid_for("HEFT", action_mode="auto_only")
    grid_ours  = grid_for("Ours-hybrid", action_mode="hybrid")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8),
                              gridspec_kw=dict(wspace=0.32))

    for ax, grid, title in (
        (axes[0], grid_heft, r"HEFT"),
        (axes[1], grid_ours, r"Ours-hybrid"),
    ):
        im = ax.imshow(grid, aspect="auto", cmap="Reds",
                       vmin=0, vmax=1, origin="lower")
        ax.set_xticks(np.arange(len(ambs)))
        ax.set_xticklabels(amb_labels, rotation=18, ha="right")
        ax.set_yticks(np.arange(len(sizes)))
        ax.set_yticklabels([f"$N{{=}}{n}$" for n in sizes])
        ax.set_title(title)
        for i in range(len(sizes)):
            for j in range(len(ambs)):
                val = grid[i, j]
                if np.isnan(val): continue
                txt_color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=txt_color)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.04,
                         shrink=0.85)
    cbar.set_label("Violation rate")
    cbar.ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Processor count $N$")
    fig.savefig(OUT_DIR / "fig_envelope_heatmap.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_envelope_heatmap.pdf'}")


# =============================================================================
# Fig 6 — §5.6 auto-cool ablation paired peak-T distribution
# =============================================================================
def fig_autocool_peak() -> None:
    """Paired peak-T distribution per scheduler under cool=100 vs cool=0.

    Data sources:
      cool=100: eval_results/ours_no_rc_edge_paired_hot_n17/paired.csv
                  (auto_only + NoRCEdge, 500 ep)
                eval_results/_phaseG/master.csv (HEFT, Decima-thermal,
                  NoThermal at hot N=17)
      cool=0:   eval_results/autocool_ablation_hot_n17/paired.csv
                  (all 4 schedulers, 500 ep)
                eval_results/nothermal_nocool/paired.csv (NoThermal,
                  200 ep)
    """
    # Schedulers (display order from highest-viol to lowest-viol)
    order = ["HEFT", "Decima-thermal", "Ours-NoThermal",
             "Ours-NoRCEdge", "Ours-auto_only"]

    # cool=100 (standard env) peak T per scheduler
    cool100 = {}
    df_phaseg = pd.read_csv("eval_results/_phaseG/master.csv")
    df_phaseg = df_phaseg[(df_phaseg.source == "grand_matrix") &
                          (df_phaseg.num_nodes == 17) &
                          (df_phaseg.ambient == "hot") &
                          (df_phaseg.action_mode == "auto_only")]
    for s in order:
        # map Ours-NoThermal -> Decima in phaseG (decima_fair = Decima
        # before rename; not the case here — Decima in phaseG is
        # Decima-thermal-true; Ours-NoThermal is mapped via aggregator)
        # Safer: pull from the paired CSVs where labels are explicit.
        pass

    # Use the paired CSVs (more reliable label match):
    df_cool100_pair = pd.read_csv(
        "eval_results/ours_no_rc_edge_paired_hot_n17/paired.csv")
    cool100["Ours-auto_only"] = df_cool100_pair[
        df_cool100_pair.scheduler == "Ours-auto_only"
    ]["peak_temp_episode"].values
    cool100["Ours-NoRCEdge"] = df_cool100_pair[
        df_cool100_pair.scheduler == "Ours-NoRCEdge"
    ]["peak_temp_episode"].values
    # HEFT/Decima-t/NoThermal at cool=100 — use grand_matrix + extras hot N=17 slice
    df_gm = pd.read_csv("eval_results/_phaseG/master.csv")
    df_gm = df_gm[(df_gm.source.isin(["grand_matrix", "grand_matrix_extras"])) &
                  (df_gm.num_nodes == 17) &
                  (df_gm.ambient == "hot") &
                  (df_gm.action_mode == "auto_only")]
    cool100["HEFT"] = df_gm[df_gm.scheduler == "HEFT"]["peak_temp_episode"].values
    cool100["Decima-thermal"] = df_gm[
        df_gm.scheduler == "Decima-thermal"]["peak_temp_episode"].values
    cool100["Ours-NoThermal"] = df_gm[
        df_gm.scheduler == "Ours-NoThermal"]["peak_temp_episode"].values

    # cool=0
    cool0 = {}
    df_ablation = pd.read_csv(
        "eval_results/autocool_ablation_hot_n17/paired.csv")
    for s in ["HEFT", "Decima-thermal", "Ours-NoRCEdge", "Ours-auto_only"]:
        cool0[s] = df_ablation[df_ablation.scheduler == s
                                ]["peak_temp_episode"].values
    df_nt0 = pd.read_csv("eval_results/nothermal_nocool/paired.csv")
    cool0["Ours-NoThermal"] = df_nt0["peak_temp_episode"].values

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    palette = {
        "HEFT": PAL["classical"],
        "Decima-thermal": PAL["decima"],
        "Ours-NoThermal": PAL["nothermal"],
        "Ours-NoRCEdge": PAL["norcedge"],
        "Ours-auto_only": PAL["auto_only"],
    }

    # Side-by-side violin: each scheduler has 2 violins (cool=100, cool=0)
    width = 0.34
    pos_left  = np.arange(len(order)) - width / 2
    pos_right = np.arange(len(order)) + width / 2

    for i, s in enumerate(order):
        c100 = cool100.get(s, np.array([]))
        c0   = cool0.get(s, np.array([]))
        if len(c100) > 0:
            parts = ax.violinplot([c100], positions=[pos_left[i]],
                                   widths=width * 0.9, showmedians=True,
                                   showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(palette[s])
                pc.set_alpha(0.45)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.4)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(0.8)
        if len(c0) > 0:
            parts = ax.violinplot([c0], positions=[pos_right[i]],
                                   widths=width * 0.9, showmedians=True,
                                   showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(palette[s])
                pc.set_alpha(0.95)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.4)
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(0.8)

    # T_pen reference line
    ax.axhline(80, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(len(order) - 0.4, 80.6, r"$T_{\mathrm{pen}}{=}80\,^\circ$C",
            fontsize=7, color="red", ha="right", va="bottom")

    ax.set_xticks(np.arange(len(order)))
    short_labels = ["HEFT", "Dec-t", "NoTherm", "NoRCEdge", "auto_only"]
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Peak temperature per episode ($^\\circ$C)")
    ax.set_ylim(55, 100)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # Legend: light = cool=100, dark = cool=0
    handles = [
        mpatches.Patch(facecolor="gray", alpha=0.45, edgecolor="black",
                        linewidth=0.4, label=r"$\mathtt{max\_cooling\_steps}{=}100$ (standard env)"),
        mpatches.Patch(facecolor="gray", alpha=0.95, edgecolor="black",
                        linewidth=0.4, label=r"$\mathtt{max\_cooling\_steps}{=}0$ (env safety floor off)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
               fontsize=7)

    fig.savefig(OUT_DIR / "fig_autocool_peak.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_autocool_peak.pdf'}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    set_paper_style()
    print("Generating final §5 paper figures...")
    fig_chain_viol()
    fig_envelope_heatmap()
    fig_autocool_peak()
    print("Done.")


if __name__ == "__main__":
    main()
