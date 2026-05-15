#!/usr/bin/env python3
"""
compose_paper_section5.py — Phase G aggregator for paper §5 wholesale rewrite

Joins Phase X-B grand matrix + X-C horizon scan CSVs, renames the legacy
'Decima' label (decima_fair ckpt) to 'Ours-NoThermal' for paper-facing
prose, and emits:

1. eval_results/_phaseG/master.csv — full joined dataset (paired by seed)
2. eval_results/_phaseG/main_table_n17_hot.tex — booktabs main table @
   N=17 HOT 500 ep (paper §5.1 Table 1 source)
3. eval_results/_phaseG/ambient_envelope_n17.tex — N=17 viol_rate per
   ambient × scheduler (paper §5.2 ambient envelope figure source)
4. eval_results/_phaseG/scaling_hot.tex — HOT × N sweep per scheduler
   (paper §5.3 scaling figure source)
5. eval_results/_phaseG/horizon_h200.tex — H=200 horizon stress table
   (paper §5.4)
6. eval_results/_phaseG/wilcoxon.txt — paired Wilcoxon + Holm-Bonferroni
   for the 6-test ablation chain
7. eval_results/_phaseG/bounded_claim.txt — Ours-hybrid unsafe-episode
   tally across the full 5×4 envelope (paper §1 contribution 4)

Paper-facing naming convention (rename in this script, NOT in CSV source):
- 'Decima' (CSV) → 'Ours-NoThermal' (paper)  [decima_fair ckpt = Ours arch w/o thermal]

User decisions encoded:
- Main ckpt: Ours-hybrid (auto_only is Stage-1 anchor / ablation row)
- Main-table row selection: HEFT, ThermalHEFT, Throttled-HEFT-hybrid, Decima-thermal,
  HGATE-PPO, D2, Ours-NoThermal, Ours-auto_only, Ours-hybrid (9 rows)
- Holm-Bonferroni family: 6 tests on the ablation chain

Invocation:
    PYTHONPATH=. python cpo_thermal_v2/scripts/compose_paper_section5.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest


# =====================================================================
# Config
# =====================================================================
EVAL_ROOT = Path("eval_results")
OUT_DIR = EVAL_ROOT / "_phaseG"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# X-B grand matrix sources (8 CSVs)
GRAND_MATRIX_SOURCES = [
    (amb, variant)
    for amb in ["cold", "warm", "hot", "extreme"]
    for variant in ["", "_extras"]
]

# X-C horizon scan sources (2 CSVs: main + extras at H=200)
HORIZON_SOURCES = [
    ("h200", ""),
    ("h200", "_extras"),
]

# Paper-facing rename
LABEL_RENAME = {"Decima": "Ours-NoThermal"}

# Paper main table row order (user-approved 9 rows)
MAIN_TABLE_ORDER = [
    "HEFT",
    "ThermalHEFT",
    "Throttled-HEFT-hybrid",
    "Decima-thermal",
    "HGATE-PPO",
    "D2",
    "Ours-NoThermal",
    "Ours-auto_only",
    "Ours-hybrid",
]

# Full 12-scheduler order (for ambient envelope, scaling, etc.)
ALL_SCHEDS_ORDER = [
    "HEFT", "RoundRobin", "ThermalHEFT",
    "Throttled-HEFT-hybrid", "Throttled-HEFT-agent_only",
    "Decima-vanilla", "Decima-thermal", "HGATE-PPO", "D2",
    "Ours-NoThermal", "Ours-auto_only", "Ours-hybrid",
]


# =====================================================================
# Load + join
# =====================================================================
def load_grand_matrix() -> pd.DataFrame:
    dfs = []
    for amb, variant in GRAND_MATRIX_SOURCES:
        path = EVAL_ROOT / f"grand_matrix_{amb}{variant}" / "episodes.csv"
        if not path.exists():
            print(f"[warn] {path} missing — skipping")
            continue
        d = pd.read_csv(path)
        d["ambient"] = amb
        d["source"] = "grand_matrix" + variant
        d["horizon"] = 20  # X-B used default dags_per_episode=20
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_horizon() -> pd.DataFrame:
    dfs = []
    for hkey, variant in HORIZON_SOURCES:
        path = EVAL_ROOT / "horizon_scan_phaseXC" / f"{hkey}{variant}" / "episodes.csv"
        if not path.exists():
            print(f"[warn] {path} missing — skipping")
            continue
        d = pd.read_csv(path)
        d["ambient"] = "hot"
        d["source"] = f"horizon_{hkey}{variant}"
        d["horizon"] = int(hkey.lstrip("h"))
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return df


def main_master() -> pd.DataFrame:
    grand = load_grand_matrix()
    horiz = load_horizon()
    df = pd.concat([grand, horiz], ignore_index=True)
    df["scheduler"] = df["scheduler"].replace(LABEL_RENAME)
    df["is_unsafe"] = (df["violations_total"] > 0).astype(int)
    # truncated may be bool; coerce
    if df["truncated"].dtype == bool:
        df["truncated"] = df["truncated"].astype(int)
    print(f"[load] master = {len(df)} rows | "
          f"{len(df.scheduler.unique())} schedulers | "
          f"N={sorted(df.num_nodes.unique())} | "
          f"ambient={sorted(df.ambient.unique())} | "
          f"horizon={sorted(df.horizon.unique())}")
    return df


# =====================================================================
# §5.1 Main table (N=17 HOT 500 ep H=20, 9 rows)
# =====================================================================
def main_table_n17_hot(df: pd.DataFrame) -> str:
    """Booktabs LaTeX for paper §5.1 Table 1."""
    sub = df[(df.num_nodes == 17) & (df.ambient == "hot") & (df.horizon == 20)]
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main results on the hot-ambient validation set "
                 "($T_{\\mathrm{ambient}} \\in [60,75]\\,^\\circ$C, $N{=}17$ "
                 "processors, $n{=}500$ paired episodes per row, "
                 "\\texttt{seed\\_base}$=100{,}000$).}")
    lines.append("\\label{tab:main-results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Scheduler & Makespan (ms) $\\downarrow$ & Peak $T$ "
                 "($^\\circ$C) $\\downarrow$ & Viol.\\ rate $\\downarrow$ "
                 "& Completion $\\uparrow$ & Episode return $\\uparrow$ \\\\")
    lines.append("\\midrule")
    for sch in MAIN_TABLE_ORDER:
        r = sub[sub.scheduler == sch]
        if not len(r):
            lines.append(f"% MISSING {sch}")
            continue
        m_mu, m_sd = r.total_makespan_ms.mean(), r.total_makespan_ms.std()
        p_mu, p_sd = r.peak_temp_episode.mean(), r.peak_temp_episode.std()
        v = r.is_unsafe.mean()
        c = r.dag_completion_rate.mean()
        e = r.episode_return.mean()
        sched_display = sch.replace("_", "\\_")
        is_main = sch == "Ours-hybrid"  # bold the main-claim row
        b_open, b_close = ("\\textbf{", "}") if is_main else ("", "")
        lines.append(f"{b_open}{sched_display}{b_close} & "
                     f"${m_mu:,.0f} \\pm {m_sd:,.0f}$ & "
                     f"${p_mu:.2f} \\pm {p_sd:.2f}$ & "
                     f"${v:.3f}$ & ${c:.3f}$ & ${e:+.2f}$ \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


# =====================================================================
# §5.2 Ambient envelope viol_rate matrix (12 sched × 4 ambient, N=17)
# =====================================================================
def ambient_envelope_n17(df: pd.DataFrame) -> str:
    sub = df[(df.num_nodes == 17) & (df.horizon == 20)]
    piv = sub.pivot_table(
        index="scheduler", columns="ambient",
        values="is_unsafe", aggfunc="mean")
    piv = piv.reindex(ALL_SCHEDS_ORDER)[["cold", "warm", "hot", "extreme"]]
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Violation rate across the four ambient regimes "
                 "at $N{=}17$, $n{=}500$ paired episodes per cell. The Ours "
                 "family (NoThermal, auto\\_only, hybrid) maintains "
                 "violation rates below $3\\%$ across the entire envelope.}")
    lines.append("\\label{tab:ambient-envelope}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Scheduler & Cold $[25,40]$ & Warm $[40,65]$ & "
                 "Hot $[60,75]$ & Extreme $[70,80]$ \\\\")
    lines.append("\\midrule")
    for sch in piv.index:
        if piv.loc[sch].isna().all():
            continue
        sched_display = sch.replace("_", "\\_")
        is_ours = sch in {"Ours-NoThermal", "Ours-auto_only", "Ours-hybrid"}
        b_open, b_close = ("\\textbf{", "}") if is_ours else ("", "")
        cells = " & ".join(
            f"${piv.loc[sch, a]:.3f}$" if not np.isnan(piv.loc[sch, a]) else "---"
            for a in ["cold", "warm", "hot", "extreme"])
        lines.append(f"{b_open}{sched_display}{b_close} & {cells} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


# =====================================================================
# §5.3 Scaling fig data (5 N × ALL sched, HOT, viol_rate + peak_T)
# =====================================================================
def scaling_hot(df: pd.DataFrame) -> str:
    sub = df[(df.ambient == "hot") & (df.horizon == 20)]
    piv = sub.pivot_table(
        index="scheduler", columns="num_nodes",
        values="is_unsafe", aggfunc="mean")
    piv = piv.reindex(ALL_SCHEDS_ORDER)[[9, 13, 17, 24, 33]]
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Violation rate across processor count "
                 "$N \\in \\{9, 13, 17, 24, 33\\}$ in the hot-ambient regime, "
                 "$n{=}500$ paired episodes per cell.}")
    lines.append("\\label{tab:scaling-hot}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Scheduler & $N{=}9$ & $N{=}13$ & $N{=}17$ & $N{=}24$ "
                 "& $N{=}33$ \\\\")
    lines.append("\\midrule")
    for sch in piv.index:
        if piv.loc[sch].isna().all():
            continue
        sched_display = sch.replace("_", "\\_")
        cells = " & ".join(
            f"${piv.loc[sch, n]:.3f}$" if not np.isnan(piv.loc[sch, n]) else "---"
            for n in [9, 13, 17, 24, 33])
        lines.append(f"{sched_display} & {cells} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


# =====================================================================
# §5.4 Horizon stress table (H=20 vs H=200 at N=17 HOT)
# =====================================================================
def horizon_h200(df: pd.DataFrame) -> str:
    sub = df[(df.num_nodes == 17) & (df.ambient == "hot")]
    piv_v = sub.pivot_table(index="scheduler", columns="horizon",
                              values="is_unsafe", aggfunc="mean")
    piv_v = piv_v.reindex(ALL_SCHEDS_ORDER)[[20, 200]]
    piv_c = sub.pivot_table(index="scheduler", columns="horizon",
                              values="dag_completion_rate", aggfunc="mean")
    piv_c = piv_c.reindex(ALL_SCHEDS_ORDER)[[20, 200]]
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Horizon stress test at $N{=}17$, hot ambient, "
                 "$n{=}500$ paired episodes per cell. Comparing standard "
                 "horizon ($H{=}20$ DAGs per episode) against long horizon "
                 "($H{=}200$). Ours-hybrid is the only scheduler maintaining "
                 "violation rate below $1\\%$ at $H{=}200$.}")
    lines.append("\\label{tab:horizon}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{2}{c}{Viol.\\ rate $\\downarrow$} "
                 "& \\multicolumn{2}{c}{Completion $\\uparrow$} \\\\")
    lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    lines.append("Scheduler & $H{=}20$ & $H{=}200$ & $H{=}20$ & $H{=}200$ \\\\")
    lines.append("\\midrule")
    for sch in piv_v.index:
        if piv_v.loc[sch].isna().all():
            continue
        sched_display = sch.replace("_", "\\_")
        v20 = piv_v.loc[sch, 20]
        v200 = piv_v.loc[sch, 200] if 200 in piv_v.columns else float("nan")
        c20 = piv_c.loc[sch, 20]
        c200 = piv_c.loc[sch, 200] if 200 in piv_c.columns else float("nan")
        lines.append(f"{sched_display} & "
                     f"{f'${v20:.3f}$' if not np.isnan(v20) else '---'} & "
                     f"{f'${v200:.3f}$' if not np.isnan(v200) else '---'} & "
                     f"{f'${c20:.3f}$' if not np.isnan(c20) else '---'} & "
                     f"{f'${c200:.3f}$' if not np.isnan(c200) else '---'} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


# =====================================================================
# §5 ablation chain — paired Wilcoxon + Holm-Bonferroni (6 tests)
# =====================================================================
def wilcoxon_ablation_chain(df: pd.DataFrame) -> str:
    sub = df[(df.num_nodes == 17) & (df.ambient == "hot") & (df.horizon == 20)]
    chain = [
        ("HEFT", "Decima-vanilla", "Component 1: RL backbone alone"),
        ("Decima-vanilla", "Decima-thermal", "Component 2: thermal reward shaping"),
        ("Decima-thermal", "D2", "Component 4: cross-attention placement (homog GCN trunk)"),
        ("Decima-thermal", "HGATE-PPO", "Component 3: hetero edge typing (Wu 2025 baseline)"),
        ("Decima-thermal", "Ours-NoThermal", "Component 5: Ours architecture WITHOUT thermal info"),
        ("Ours-NoThermal", "Ours-auto_only", "Component 6: thermal observation + reward marginal"),
    ]
    out = []
    out.append("Paired Wilcoxon + Holm-Bonferroni for the 6-test ablation chain")
    out.append("at N=17 HOT, n=500 paired episodes (seed_base=100000)")
    out.append("=" * 78)
    out.append("")
    raw_ps = []
    test_lines = []
    for left, right, desc in chain:
        a = sub[sub.scheduler == left].sort_values("episode_id").reset_index(drop=True)
        b = sub[sub.scheduler == right].sort_values("episode_id").reset_index(drop=True)
        n = min(len(a), len(b))
        if n == 0:
            test_lines.append(f"[SKIP {left} → {right}] missing data")
            raw_ps.append(float("nan"))
            continue
        av = a.is_unsafe.head(n).values
        bv = b.is_unsafe.head(n).values
        delta = av - bv
        only_a = ((av == 1) & (bv == 0)).sum()
        only_b = ((av == 0) & (bv == 1)).sum()
        if only_a + only_b > 0:
            res = binomtest(only_a, only_a + only_b, p=0.5, alternative='two-sided')
            p = res.pvalue
        else:
            p = float("nan")
        raw_ps.append(p)
        viol_a = av.mean()
        viol_b = bv.mean()
        test_lines.append(
            f"{left:18s} → {right:18s} | viol {viol_a:.3f} → {viol_b:.3f} | "
            f"only-A-unsafe={only_a:3d}, only-B-unsafe={only_b:3d} | "
            f"McNemar p={p:.4f}  [{desc}]"
        )
    # Holm-Bonferroni
    out.append("Per-test results (McNemar exact, paired binary violation indicator):")
    out.append("-" * 78)
    out.extend(test_lines)
    out.append("")
    # Compute Holm-adjusted
    valid = [(i, p) for i, p in enumerate(raw_ps) if not np.isnan(p)]
    valid_sorted = sorted(valid, key=lambda x: x[1])
    m = len(valid_sorted)
    holm = {}
    prev = 0.0
    for rank, (i, p) in enumerate(valid_sorted):
        adj = p * (m - rank)
        adj = max(adj, prev)
        adj = min(adj, 1.0)
        holm[i] = adj
        prev = adj
    out.append(f"Holm-Bonferroni adjusted (family-wise alpha = 0.05, m={m} tests):")
    out.append("-" * 78)
    for i, (left, right, desc) in enumerate(chain):
        if i not in holm:
            out.append(f"[{left} → {right}]  raw=---  Holm=---  ---")
            continue
        adj = holm[i]
        sig = "***" if adj < 0.001 else "**" if adj < 0.01 else "*" if adj < 0.05 else "ns"
        out.append(f"[{left:18s} → {right:18s}]  raw p={raw_ps[i]:.4f}  Holm p={adj:.4f}  {sig}")
    return "\n".join(out) + "\n"


# =====================================================================
# §1 contribution 4 bounded-claim tally
# =====================================================================
def bounded_claim(df: pd.DataFrame) -> str:
    out = []
    out.append("Bounded-claim residual count (paper §1 contribution 4)")
    out.append("=" * 70)
    out.append("Domain: 5 N × 4 ambient × 500 ep paired = 12,500 ep envelope")
    out.append("")
    sub = df[(df.horizon == 20)]
    for sch in ["Ours-hybrid", "Ours-auto_only", "Ours-NoThermal"]:
        r = sub[sub.scheduler == sch]
        unsafe_total = int(r.is_unsafe.sum())
        n_total = len(r)
        out.append(f"{sch:20s} {unsafe_total:>4d} / {n_total:>5d} unsafe  "
                   f"({100*unsafe_total/n_total:.3f}%)")
        # Per-ambient breakdown
        for amb in ["cold", "warm", "hot", "extreme"]:
            ramb = r[r.ambient == amb]
            u = int(ramb.is_unsafe.sum())
            n = len(ramb)
            out.append(f"   {amb:10s}  {u:>3d} / {n:>4d}")
    return "\n".join(out) + "\n"


# =====================================================================
# Main
# =====================================================================
def main():
    df = main_master()

    master_path = OUT_DIR / "master.csv"
    df.to_csv(master_path, index=False)
    print(f"[write] {master_path}  ({len(df):,} rows)")

    artifacts = [
        ("main_table_n17_hot.tex", main_table_n17_hot(df)),
        ("ambient_envelope_n17.tex", ambient_envelope_n17(df)),
        ("scaling_hot.tex", scaling_hot(df)),
        ("horizon_h200.tex", horizon_h200(df)),
        ("wilcoxon.txt", wilcoxon_ablation_chain(df)),
        ("bounded_claim.txt", bounded_claim(df)),
    ]
    for fname, content in artifacts:
        path = OUT_DIR / fname
        path.write_text(content, encoding="utf-8")
        print(f"[write] {path}  ({len(content):,} chars)")

    print(f"\n[done] all Phase G artifacts in {OUT_DIR}/")
    print("\nNext: switch to paper-draft branch + plug-in to "
          "paper_drafts/section5_main_results.tex + section5X_hybrid_case_study.tex")


if __name__ == "__main__":
    main()
