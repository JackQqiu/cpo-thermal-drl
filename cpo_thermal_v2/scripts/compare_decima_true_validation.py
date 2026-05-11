#!/usr/bin/env python3
"""
compare_decima_true_validation.py
4-way paired compare for paper §5 Decima rows.

Loads two CSVs (HOT regime, N=17, paired seed_base=100000):
  DECIMA_MAIN: eval_results/decima_true_validation_hot/episodes.csv
               HEFT + Decima-vanilla + Decima-thermal
  V2_COMP    : eval_results/stage2_v2_validation_stage1ckpt_hot/episodes.csv
               Ours-auto_only (Stage 1 ckpt — Pareto anchor)

Final 4-scheduler table:
  HEFT
  Ours-auto_only-s1
  Decima-vanilla
  Decima-thermal

Paired Wilcoxon for each Decima variant vs HEFT and vs s1-auto.
"""
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=str, default="hot",
                    choices=["warm", "hot"])
    return ap.parse_args()


def load(d: Path) -> pd.DataFrame:
    p = d / "episodes.csv"
    if not p.exists():
        for alt in ["episode_records.csv", "ep_records.csv", "results.csv"]:
            if (d / alt).exists():
                p = d / alt; break
        else:
            print(f"FATAL: no csv in {d}", file=sys.stderr); sys.exit(1)
    return pd.read_csv(p)


_UP_GOOD = {"dag_completion_rate", "episode_return"}


def _paired_print(name_a, vals_a, name_b, vals_b, comment, metrics):
    n = min(len(vals_a), len(vals_b))
    print("-"*82)
    print(f"COMPARISON: {name_a} vs {name_b}  [n={n}]")
    print(comment)
    print("-"*82)
    for m in metrics:
        av = vals_a[m].head(n).values
        bv = vals_b[m].head(n).values
        delta = av - bv
        direction = "↑" if m in _UP_GOOD else "↓"
        try:
            if np.allclose(delta, 0):
                p = float("nan"); sig = "n/a"
            else:
                _, p = stats.wilcoxon(av, bv)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        except Exception as e:
            p = float("nan"); sig = f"err:{e}"
        print(f"  {m:25s} ({direction})  {name_a}={av.mean():8.2f}  "
              f"{name_b}={bv.mean():8.2f}  Δ={delta.mean():+7.2f}  "
              f"p={p:.4f} {sig}")


def main():
    args = parse_args()
    regime = args.regime

    DECIMA_MAIN = Path(f"eval_results/decima_true_validation_{regime}")
    V2_COMP     = Path(f"eval_results/stage2_v2_validation_stage1ckpt_{regime}")

    print(f"\nLoading from regime={regime!r}:")
    print(f"  DECIMA_MAIN: {DECIMA_MAIN}")
    print(f"  V2_COMP:     {V2_COMP}")

    df_d = load(DECIMA_MAIN)
    df_s = load(V2_COMP)

    # Relabel
    df_s = df_s[df_s["scheduler"] == "Ours-auto_only"].copy()
    df_s.loc[:, "scheduler"] = "Ours-auto_only-s1"

    df = pd.concat([df_d, df_s], ignore_index=True)

    if "truncated" in df.columns and df["truncated"].dtype == bool:
        df["truncated"] = df["truncated"].astype(int)

    candidates = ["total_makespan_ms", "peak_temp_episode", "cooling_total_ms",
                  "truncated", "dag_completion_rate", "episode_return",
                  "violations_total"]
    metrics = [c for c in candidates if c in df.columns]

    print("\n" + "="*82)
    print(f"DECIMA TRUE 4-WAY ({regime.upper()}) — N=17, 50 paired episodes")
    print("="*82)
    print(f"\nMetrics: {metrics}\n")

    print("Per-scheduler summary (mean ± std):\n")
    print(df.groupby("scheduler")[metrics].agg(["mean", "std"]).round(3).to_string())

    ep_col = next((c for c in ["episode_id", "episode_idx", "ep_idx", "seed"]
                    if c in df.columns), None)
    print(f"\nPairing on column: '{ep_col}'\n")

    heft = df[df["scheduler"] == "HEFT"].sort_values(ep_col).reset_index(drop=True)
    s1   = df[df["scheduler"] == "Ours-auto_only-s1"].sort_values(ep_col).reset_index(drop=True)
    dv   = df[df["scheduler"] == "Decima-vanilla"].sort_values(ep_col).reset_index(drop=True)
    dt   = df[df["scheduler"] == "Decima-thermal"].sort_values(ep_col).reset_index(drop=True)

    # Paired comparisons that matter for §5
    _paired_print("Decima-vanilla", dv, "HEFT", heft,
                  "Decima-vanilla vs HEFT — does makespan-only RL beat the heuristic baseline?",
                  metrics)
    print()
    _paired_print("Decima-thermal", dt, "HEFT", heft,
                  "Decima-thermal vs HEFT — does thermal-aware reward signal help over heuristic?",
                  metrics)
    print()
    _paired_print("Decima-thermal", dt, "Decima-vanilla", dv,
                  "Decima-thermal vs Decima-vanilla — pure isolation of reward signal effect",
                  metrics)
    print()
    _paired_print("Decima-thermal", dt, "Ours-auto_only-s1", s1,
                  "Decima-thermal vs Ours-s1 — does hetero-GAT + dual reward shaping outperform homog GCN + thermal reward?",
                  metrics)
    print()
    _paired_print("Decima-vanilla", dv, "Ours-auto_only-s1", s1,
                  "Decima-vanilla vs Ours-s1 — baseline-to-Ours gap with vanilla algorithm",
                  metrics)

    # 4-row Pareto positioning
    print("\n" + "="*82)
    print(f"PARETO POSITIONING — {regime.upper()} REGIME")
    print("="*82)
    print(f"{'Scheduler':25s}  {'makespan':>14s}  {'peak_T':>14s}  "
          f"{'viol_rate':>10s}  {'completion':>10s}  {'ep_ret':>10s}")
    for sch in ["HEFT", "Ours-auto_only-s1", "Decima-vanilla", "Decima-thermal"]:
        r = df[df["scheduler"] == sch]
        if not len(r):
            continue
        # violations rate = fraction of episodes with violations_total > 0
        viol_rate = (r["violations_total"] > 0).mean() if "violations_total" in r else float("nan")
        print(f"{sch:25s}  "
              f"{r['total_makespan_ms'].mean():>7.1f}±{r['total_makespan_ms'].std():>5.1f}  "
              f"{r['peak_temp_episode'].mean():>7.2f}±{r['peak_temp_episode'].std():>5.2f}  "
              f"{viol_rate:>9.3f}  "
              f"{r['dag_completion_rate'].mean():>7.3f}   "
              f"{r['episode_return'].mean():>7.2f}")

    # Violation rate breakdown
    print("\n" + "="*82)
    print(f"VIOLATION RATES — fraction of {len(heft)} episodes with violations_total > 0")
    print("="*82)
    for sch in ["HEFT", "Ours-auto_only-s1", "Decima-vanilla", "Decima-thermal"]:
        r = df[df["scheduler"] == sch]
        if not len(r):
            continue
        viol_eps = int((r["violations_total"] > 0).sum())
        mean_viol_when_present = (r[r["violations_total"] > 0]["violations_total"].mean()
                                   if viol_eps > 0 else 0.0)
        print(f"  {sch:25s}  {viol_eps:>3d}/{len(r):<3d} eps  "
              f"(mean viol/ep when present: {mean_viol_when_present:.2f})")


if __name__ == "__main__":
    main()
