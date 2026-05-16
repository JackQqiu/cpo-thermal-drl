"""
make_fig_concept.py — 3-panel concept figure for §3 (paper).

Reads eval_results/concept_figure_trace.json (produced by
trace_concept_figure.py) and emits draft/figures/section3/fig_concept.{pdf,svg}.

Panels (vertical stack):
  (a) DAG: a sample microservice DAG drawn as a node-link diagram.
      Node size encodes workload; the critical path is highlighted.
  (b) Gantt: 2-row stacked schedule (HEFT top, Ours-hybrid bottom)
      showing each task's execution block on its placed processor,
      colour-coded by DAG, with cooling-pause hashes.
  (c) Thermal trace: peak temperature vs time for both schedulers,
      with the T_pen=80°C line and the violation region above it
      shaded in light red.

The styling matches the §5 figures (Nature structure + MDPI serif).

Usage:
  python -m cpo_thermal_v2.scripts.make_fig_concept
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Reuse paper style from the other figure script
from cpo_thermal_v2.scripts.make_paper_figures_final import set_paper_style, PAL


OUT_DIR = Path("draft/figures/section3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_pdf_and_svg(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.svg")
    print(f"  wrote {OUT_DIR / stem}.{{pdf,svg}}")


def _add_panel_label(ax: plt.Axes, label: str,
                      x: float = -0.06, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            ha="right", va="bottom")


def _draw_dag(ax: plt.Axes, dag: dict) -> None:
    """Render a single DAG using a simple layered (depth-based) layout."""
    nodes = dag["nodes"]
    edges = [tuple(e) for e in dag["edges"]]
    workloads = dag["workloads"]

    # Compute depth (longest path from source) for each node
    depth = {n: 0 for n in nodes}
    succ = {n: [] for n in nodes}
    pred = {n: [] for n in nodes}
    for u, v in edges:
        succ[u].append(v)
        pred[v].append(u)
    # Topological order: nodes with no predecessor first
    visited = set()
    order = []
    sources = [n for n in nodes if not pred[n]]
    stack = list(sources)
    while stack:
        n = stack.pop(0)
        if n in visited:
            continue
        if not all(p in visited for p in pred[n]):
            stack.append(n)
            continue
        visited.add(n)
        order.append(n)
        for s in succ[n]:
            depth[s] = max(depth[s], depth[n] + 1)
            if s not in visited and s not in stack:
                stack.append(s)
    # Some DAGs may have cycles in the data? Just take order so far.
    for n in nodes:
        if n not in visited:
            order.append(n)
    max_depth = max(depth.values()) if depth else 1

    # Layered layout: x = depth (stretched), y = within-depth slot (tighter)
    # X-stretch pushes depth layers further apart so the DAG fills the
    # full panel width; the tighter Y allows the figure to stay compact
    # vertically while preventing node overlap (combined with the smaller
    # node radius cap below).
    depth_groups = {d: [] for d in range(max_depth + 1)}
    for n in order:
        depth_groups[depth[n]].append(n)
    pos = {}
    X_STRETCH = 2.4
    Y_TIGHTEN = 0.95
    for d, ns in depth_groups.items():
        for i, n in enumerate(ns):
            pos[n] = (d * X_STRETCH,
                      (i - (len(ns) - 1) / 2) * Y_TIGHTEN)

    # Find a simple longest path (critical path approximation)
    # Use weighted edge by successor workload
    max_w = max(workloads.values()) if workloads else 1.0
    cp_nodes = set()
    if order:
        # Greedy: from each source, follow highest-workload successor
        for src in sources:
            curr = src
            cp_nodes.add(curr)
            while succ[curr]:
                nxt = max(succ[curr], key=lambda s: workloads.get(s, 0))
                cp_nodes.add(nxt)
                curr = nxt

    # Draw edges
    for u, v in edges:
        xu, yu = pos[u]
        xv, yv = pos[v]
        is_cp = (u in cp_nodes) and (v in cp_nodes)
        ax.annotate("",
                    xy=(xv, yv), xytext=(xu, yu),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#a8423a" if is_cp else "#888888",
                        lw=1.6 if is_cp else 0.7,
                        alpha=0.95 if is_cp else 0.55,
                    ))

    # Draw nodes — size by workload, colored by critical-path membership.
    # Radius cap reduced (was 0.20-0.50) so adjacent same-depth nodes do
    # not overlap when the workload-scaled radius hits its upper bound.
    for n in nodes:
        x, y = pos[n]
        w = workloads.get(n, 1.0) / max_w
        radius = 0.17 + 0.22 * w   # 0.17..0.39
        if n in cp_nodes:
            face = "#d65644"
            edge = "#7a1f1f"
        else:
            face = "#8aa1b8"
            edge = "#5b7a99"
        circle = plt.Circle((x, y), radius,
                              facecolor=face, edgecolor=edge,
                              linewidth=0.8, zorder=3)
        ax.add_patch(circle)

    ax.set_xlim(-0.7, max_depth * X_STRETCH + 0.7)
    ymax = max(len(g) for g in depth_groups.values()) if depth_groups else 1
    ax.set_ylim(-(ymax * Y_TIGHTEN) / 2 - 0.6,
                 (ymax * Y_TIGHTEN) / 2 + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend for DAG panel
    cp_patch = mpatches.Patch(color="#d65644",
                                label="task on critical path")
    nor_patch = mpatches.Patch(color="#8aa1b8",
                                label="parallel-branch task")
    ax.legend(handles=[cp_patch, nor_patch],
              loc="lower right", frameon=False, fontsize=7)


def _draw_gantt(ax: plt.Axes, schedule_h, schedule_o,
                num_nodes: int, t_max: float) -> None:
    """Two-row Gantt: HEFT on top, Ours-hybrid on bottom."""
    cmap = cm.get_cmap("tab20")

    def draw_one(sched, y0):
        for entry in sched:
            t0 = entry["t_start"]
            t_cool = entry["t_cool"]
            t_exec = entry["t_exec"]
            proc = entry["proc"]
            dag_id = entry["dag_id"]
            # Cooling block (hashed)
            if t_cool > 0:
                ax.barh(y0 + proc, t_cool, left=t0,
                        height=0.7, color="white",
                        edgecolor="#888888", linewidth=0.3,
                        hatch="///", alpha=0.7)
            # Execution block
            ax.barh(y0 + proc, t_exec, left=t0 + t_cool,
                    height=0.7,
                    color=cmap(dag_id % 20),
                    edgecolor="black", linewidth=0.25)

    # HEFT in upper half, Ours-hybrid in lower half
    y_heft   = num_nodes + 2     # 0..N-1 are processor rows; offset by +N+2
    y_ours   = 0
    draw_one(schedule_h, y_heft)
    draw_one(schedule_o, y_ours)

    # Y-axis tick labels (processor IDs, each row)
    # Use compact: every 4th processor labeled
    ytick_positions = []
    ytick_labels    = []
    for p in [0, 4, 8, 12, 16]:
        if p < num_nodes:
            ytick_positions.append(y_heft + p)
            ytick_labels.append(f"P{p}")
            ytick_positions.append(y_ours + p)
            ytick_labels.append(f"P{p}")
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=7)

    # Row group labels (HEFT / Ours-hybrid) — pushed left of y-tick labels
    ax.text(-0.085, (y_heft + num_nodes/2) / (y_heft + num_nodes + 1),
            "HEFT", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold",
            color="#5b7a99",
            ha="right", va="center", rotation=90)
    ax.text(-0.085, (y_ours + num_nodes/2) / (y_heft + num_nodes + 1),
            "Ours-hybrid", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold",
            color="#7a1f1f",
            ha="right", va="center", rotation=90)

    # Horizontal separator between HEFT and Ours-hybrid bands
    sep_y = num_nodes + 0.5
    ax.axhline(sep_y, color="#bbbbbb", linewidth=0.5, linestyle=":")

    ax.set_xlim(0, t_max)
    ax.set_ylim(-1, y_heft + num_nodes + 1)
    ax.set_xlabel("Time (ms)")

    # Cooling-hatch legend element
    cool_patch = mpatches.Patch(facecolor="white", edgecolor="#888888",
                                  hatch="///", linewidth=0.3,
                                  label="env auto-cool insert")
    task_patch = mpatches.Patch(facecolor="#7a7a85", edgecolor="black",
                                  linewidth=0.3,
                                  label="task execution (color $=$ DAG)")
    ax.legend(handles=[task_patch, cool_patch],
              loc="upper right", frameon=False, fontsize=7,
              bbox_to_anchor=(1.0, 1.12), ncol=2)


def _draw_thermal(ax: plt.Axes, thermal_h, thermal_o, t_max: float) -> None:
    """Peak-T trace for both schedulers."""
    t_h    = [r["t"]      for r in thermal_h]
    pk_h   = [r["peak_T"] for r in thermal_h]
    t_o    = [r["t"]      for r in thermal_o]
    pk_o   = [r["peak_T"] for r in thermal_o]

    ax.plot(t_h, pk_h, color="#5b7a99", linewidth=1.4,
            label="HEFT")
    ax.plot(t_o, pk_o, color="#7a1f1f", linewidth=1.4,
            label="Ours-hybrid")

    # T_pen reference
    ax.axhline(80, color="#c0392b", linestyle="--", linewidth=0.95, alpha=0.85)
    ax.axhspan(80, 100, color="#c0392b", alpha=0.08, zorder=0)
    ax.text(t_max * 0.98, 81.2, r"$T_{\mathrm{pen}}{=}80\,^\circ$C",
            fontsize=8, color="#c0392b", ha="right", va="bottom")

    ax.set_xlim(0, t_max)
    ax.set_ylim(60, 100)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Peak temperature ($^\\circ$C)")
    ax.legend(loc="upper left", frameon=False, fontsize=8,
              bbox_to_anchor=(0.0, 1.05), ncol=2)


def main() -> None:
    set_paper_style()
    with open("eval_results/concept_figure_trace.json") as f:
        data = json.load(f)

    num_nodes = data["num_nodes"]
    trace_h = data["HEFT"]
    trace_o = data["Ours-hybrid"]

    # Choose a representative DAG: pick the first DAG with 8-15 tasks
    candidate_dags = [d for d in trace_h["dags"] if 7 <= d["n_tasks"] <= 16]
    if candidate_dags:
        sample_dag = candidate_dags[0]
    else:
        sample_dag = trace_h["dags"][0]
    print(f"Selected DAG {sample_dag['dag_id']} with {sample_dag['n_tasks']} tasks")

    # Restrict the Gantt + thermal trace to the same time window so both panels
    # share x-axis. Use a window long enough for HEFT to visibly cross T_pen.
    t_cross_80 = None
    for r in trace_h["thermal"]:
        if r["peak_T"] >= 80.0:
            t_cross_80 = r["t"]
            break
    if t_cross_80 is None:
        t_cross_80 = trace_h["thermal"][-1]["t"]
    # Extend window beyond the crossing so the violation is visible
    t_max = min(t_cross_80 + 400.0, trace_h["thermal"][-1]["t"])
    print(f"HEFT first crosses T_pen=80°C at t={t_cross_80:.1f} ms")
    print(f"Gantt + thermal x-axis: [0, {t_max:.1f} ms]")

    # Filter schedule + thermal to t < t_max
    sched_h = [s for s in trace_h["schedule"] if s["t_start"] < t_max]
    sched_o = [s for s in trace_o["schedule"] if s["t_start"] < t_max]
    therm_h = [r for r in trace_h["thermal"]  if r["t"]       <= t_max]
    therm_o = [r for r in trace_o["thermal"]  if r["t"]       <= t_max]

    # ---- Build figure ----
    fig = plt.figure(figsize=(6.8, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.4, 1.5], hspace=0.45)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    _draw_dag(ax_a, sample_dag)
    _add_panel_label(ax_a, "a", x=-0.06, y=0.96)
    ax_a.text(0.5, 1.02,
              f"Sample DAG ({sample_dag['n_tasks']} tasks, "
              f"{len(sample_dag['edges'])} edges)",
              transform=ax_a.transAxes,
              fontsize=9, ha="center", va="bottom")

    _draw_gantt(ax_b, sched_h, sched_o, num_nodes, t_max)
    _add_panel_label(ax_b, "b", x=-0.08, y=1.06)
    ax_b.text(0.5, 1.06,
              f"Schedule timeline (HEFT vs Ours-hybrid, "
              f"$N{{=}}{num_nodes}$ processors)",
              transform=ax_b.transAxes,
              fontsize=9, ha="center", va="bottom")

    _draw_thermal(ax_c, therm_h, therm_o, t_max)
    _add_panel_label(ax_c, "c", x=-0.08, y=1.05)
    ax_c.text(0.5, 1.05,
              "Peak processor temperature over the same time window",
              transform=ax_c.transAxes,
              fontsize=9, ha="center", va="bottom")

    _save_pdf_and_svg(fig, "fig_concept")
    plt.close(fig)


if __name__ == "__main__":
    main()
