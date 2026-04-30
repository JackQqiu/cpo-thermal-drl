"""
evaluation/tables.py — LaTeX table generation
==============================================

Outputs ready-to-paste LaTeX tables for the paper:

  Table 1 — main results table (one row per scheduler, columns are
            metrics, values are mean ± stderr at N=17)
  Table 2 — scaling table (rows: scheduler, cols: N values, value: a
            single key metric like normalised makespan)
  Table 3 — ablation table (Ours-auto_only vs Ours-agent_only vs
            Ours-hybrid: shows what each piece contributes)

Generated tables use ``booktabs``-style rules and ``siunitx``-friendly
formatting.  The required preamble for the user's main.tex::

    \\usepackage{booktabs}
    \\usepackage{siunitx}
    \\sisetup{
        round-mode = places, round-precision = 2,
        separate-uncertainty = true,
    }
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =====================================================================
# Table 1 — main results
# =====================================================================
def main_results_table(
    df: pd.DataFrame,
    num_nodes:    int = 17,
    metrics:      Optional[List[str]] = None,
    out_path:     Optional[str] = None,
    caption:      str = "Comparison of schedulers at $N=17$.  "
                        "Mean over 500 evaluation episodes; smaller is better "
                        "for makespan and violation count, larger for "
                        "completion rate.",
    label:        str = "tab:main",
) -> str:
    """Generate Table 1: scheduler × metric, mean ± stderr."""
    if metrics is None:
        metrics = [
            "total_makespan_ms",
            "peak_temp_episode",
            "violations_total",
            "cooling_total_ms",
            "dag_completion_rate",
        ]
    sub = df[df["num_nodes"] == num_nodes]
    if sub.empty:
        return f"% [main_results_table] no data for N={num_nodes}\n"

    schedulers = _scheduler_order(sub["scheduler"].unique())

    # Build cell strings
    rows: List[str] = []
    for sched in schedulers:
        s = sub[sub["scheduler"] == sched]
        cells = [_escape_latex(sched)]
        for m in metrics:
            if m not in s.columns or s[m].dropna().empty:
                cells.append("--")
                continue
            vals = s[m].dropna().values
            mu, se = vals.mean(), vals.std() / np.sqrt(max(1, len(vals)))
            cells.append(f"\\num{{{mu:.2f} \\pm {se:.2f}}}")
        rows.append(" & ".join(cells) + r" \\")

    n_metrics = len(metrics)
    col_spec  = "l" + "S" * n_metrics
    header_metrics = " & ".join(_metric_header(m) for m in metrics)
    body = "\n".join(rows)
    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"Scheduler & {header_metrics} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    if out_path:
        with open(out_path, "w") as f:
            f.write(latex)
    return latex


# =====================================================================
# Table 2 — scaling
# =====================================================================
def scaling_table(
    df: pd.DataFrame,
    metric:   str = "mean_makespan_normalized",
    out_path: Optional[str] = None,
    caption:  Optional[str] = None,
    label:    str = "tab:scaling",
) -> str:
    """Generate Table 2: scheduler × N, single metric (default normalised
    makespan).  Demonstrates generalisation across topology sizes."""
    if metric not in df.columns:
        return f"% [scaling_table] metric {metric!r} not found\n"
    sizes = sorted(df["num_nodes"].unique())
    schedulers = _scheduler_order(df["scheduler"].unique())

    if caption is None:
        caption = (
            f"Scheduler performance across topology sizes "
            f"({_metric_text(metric)}; mean $\\pm$ stderr).  "
            "Schedulers were trained at $N=17$ and evaluated zero-shot."
        )

    rows: List[str] = []
    for sched in schedulers:
        s = df[df["scheduler"] == sched]
        cells = [_escape_latex(sched)]
        for N in sizes:
            ss = s[s["num_nodes"] == N][metric].dropna()
            if ss.empty:
                cells.append("--")
                continue
            mu, se = ss.mean(), ss.std() / np.sqrt(max(1, len(ss)))
            cells.append(f"\\num{{{mu:.2f} \\pm {se:.2f}}}")
        rows.append(" & ".join(cells) + r" \\")

    col_spec  = "l" + "S" * len(sizes)
    header_N  = " & ".join(f"$N={N}$" for N in sizes)
    body      = "\n".join(rows)
    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"Scheduler & {header_N} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    if out_path:
        with open(out_path, "w") as f:
            f.write(latex)
    return latex


# =====================================================================
# Table 3 — ablation
# =====================================================================
def ablation_table(
    df: pd.DataFrame,
    num_nodes: int = 17,
    out_path:  Optional[str] = None,
) -> str:
    """Ours-auto_only vs Ours-agent_only vs Ours-hybrid at fixed N.

    Shows the contribution of each design element:
        auto_only  : env auto-cool only (no agent delay)
        agent_only : agent controls delay, env doesn't auto-cool
        hybrid     : agent + env both
    """
    ours_labels = ["Ours-auto_only", "Ours-agent_only", "Ours-hybrid"]
    sub = df[(df["num_nodes"] == num_nodes)
             & df["scheduler"].isin(ours_labels)]
    if sub.empty:
        return f"% [ablation_table] no Ours-* rows at N={num_nodes}\n"

    metrics = [
        "total_makespan_ms",
        "peak_temp_episode",
        "violations_total",
        "agent_delay_total_ms",
        "cooling_total_ms",
    ]
    rows: List[str] = []
    for sched in ours_labels:
        s = sub[sub["scheduler"] == sched]
        cells = [_escape_latex(sched.replace("Ours-", ""))]
        for m in metrics:
            ss = s[m].dropna()
            if ss.empty:
                cells.append("--")
                continue
            mu = ss.mean()
            cells.append(f"\\num{{{mu:.2f}}}")
        rows.append(" & ".join(cells) + r" \\")

    col_spec  = "l" + "S" * len(metrics)
    header_metrics = " & ".join(_metric_header(m) for m in metrics)
    body = "\n".join(rows)
    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{Ablation study at $N={num_nodes}$: "
        "contribution of each design element.}}\n"
        f"\\label{{tab:ablation_N{num_nodes}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"Variant & {header_metrics} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    if out_path:
        with open(out_path, "w") as f:
            f.write(latex)
    return latex


# =====================================================================
# Helpers
# =====================================================================
def _scheduler_order(unique: list) -> List[str]:
    """Standard ordering: baselines first, then Ours-* variants."""
    canonical = ["RoundRobin", "HEFT", "ThermalHEFT", "Decima",
                 "Ours-auto_only", "Ours-agent_only", "Ours-hybrid"]
    return [s for s in canonical if s in unique] + \
           [s for s in unique if s not in canonical]


def _metric_header(m: str) -> str:
    return {
        "total_makespan_ms":         "{Makespan (ms)}",
        "peak_temp_episode":         "{Peak T (\\si{\\celsius})}",
        "violations_total":          "{Violations}",
        "cooling_total_ms":          "{Cool. ovhd (ms)}",
        "agent_delay_total_ms":      "{Agent delay (ms)}",
        "dag_completion_rate":       "{DAG comp.}",
        "mean_makespan_normalized":  "{Norm. makespan}",
        "episode_return":            "{Return}",
    }.get(m, "{" + m.replace("_", r"\_") + "}")


def _metric_text(m: str) -> str:
    return {
        "total_makespan_ms":         "total makespan (ms)",
        "peak_temp_episode":         "peak temperature (°C)",
        "violations_total":          "violation count",
        "cooling_total_ms":          "cooling overhead (ms)",
        "mean_makespan_normalized":  "normalised makespan",
    }.get(m, m.replace("_", " "))


def _escape_latex(s: str) -> str:
    return str(s).replace("_", r"\_")


# =====================================================================
# All-tables convenience entrypoint
# =====================================================================
def emit_all_tables(
    df_episodes: pd.DataFrame,
    out_dir:     str,
    num_nodes_main: int = 17,
) -> None:
    """Write tab1.tex, tab2.tex, tab3.tex to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    main_results_table(df_episodes, num_nodes=num_nodes_main,
                       out_path=os.path.join(out_dir, "tab1_main.tex"))
    scaling_table(df_episodes, metric="mean_makespan_normalized",
                  out_path=os.path.join(out_dir, "tab2_scaling_makespan.tex"))
    scaling_table(df_episodes, metric="violations_total",
                  out_path=os.path.join(out_dir, "tab2_scaling_violations.tex"),
                  label="tab:scaling_viol",
                  caption="Violation count across topology sizes.")
    ablation_table(df_episodes, num_nodes=num_nodes_main,
                   out_path=os.path.join(out_dir, "tab3_ablation.tex"))
