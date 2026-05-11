#!/usr/bin/env python3
"""
compare_hgate_ppo_validation.py
3-way paired compare for paper §5 HGATE-PPO row.

Loads two CSVs (HOT regime, N=17, paired seed_base=100000):
  HGATE_MAIN: eval_results/hgate_ppo_validation_hot/episodes.csv
              HEFT + HGATE-PPO
  V2_COMP   : eval_results/stage2_v2_validation_stage1ckpt_hot/episodes.csv
              Ours-auto_only (Stage 1 ckpt — Pareto anchor)

If the Decima true 4-way CSV is also present, it's joined too so the
§5 chain HEFT -> Decima-thermal -> HGATE-PPO -> Ours-s1 reads end-to-end
in a single compare run.

Paired Wilcoxon:
  HGATE-PPO vs HEFT             — does Wu 2025 architecture beat heuristic?
  HGATE-PPO vs Decima-thermal   — does +hetero-GAT help over homog-GCN?
                                  (the explicit §5 isolation point)
  HGATE-PPO vs Ours-auto_only-s1 — what does removing RC edges + cross-
                                   attention placement actor cost?
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


def load_optional(d: Path):
    p = d / "episodes.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


_UP_GOOD = {"dag_completion_rate", "episode_return"}


def _paired_print(name_a, vals_a, name_b, vals_b, comment, metrics):
    if vals_a is None or vals_b is None or len(vals_a) == 0 or len(vals_b) == 0:
        print(f"\n[SKIP COMPARISON: {name_a} vs {name_b}] "
              f"missing input CSV")
        return
    n = min(len(vals_a), len(vals_b))
    print("-"*82)
    print(f"COMPARISON: {name_a} vs {name_b}  [n={n}]")
    print(comment)
    print("-"*82)
    for m in metrics:
        if m not in vals_a.columns or m not in vals_b.columns:
            continue
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

    HGATE_MAIN  = Path(f"eval_results/hgate_ppo_validation_{regime}")
    V2_COMP     = Path(f"eval_results/stage2_v2_validation_stage1ckpt_{regime}")
    DECIMA_MAIN = Path(f"eval_results/decima_true_validation_{regime}")

    print(f"\nLoading from regime={regime!r}:")
    print(f"  HGATE_MAIN: {HGATE_MAIN}")
    print(f"  V2_COMP:    {V2_COMP}")
    print(f"  DECIMA_MAIN (optional): {DECIMA_MAIN}")

    df_h = load_optional(HGATE_MAIN)
    df_s = load_optional(V2_COMP)
    df_d = load_optional(DECIMA_MAIN)

    if df_h is None:
        print(f"FATAL: HGATE eval missing at {HGATE_MAIN}/episodes.csv. "
              f"Run the eval first: \n"
              f"  PYTHONPATH=. python -m cpo_thermal_v2.evaluation.evaluate \\\n"
              f"      --config cpo_thermal_v2/configs/eval_hgate_ppo_{regime}.yaml")
        sys.exit(1)

    # Slice s1-auto out of V2_COMP if present
    if df_s is not None:
        df_s = df_s[df_s["scheduler"] == "Ours-auto_only"].copy()
        df_s.loc[:, "scheduler"] = "Ours-auto_only-s1"

    frames = [df_h]
    if df_s is not None:
        frames.append(df_s)
    if df_d is not None:
        frames.append(df_d)
    df = pd.concat(frames, ignore_index=True)

    if "truncated" in df.columns and df["truncated"].dtype == bool:
        df["truncated"] = df["truncated"].astype(int)

    candidates = ["total_makespan_ms", "peak_temp_episode", "cooling_total_ms",
                  "truncated", "dag_completion_rate", "episode_return",
                  "violations_total"]
    metrics = [c for c in candidates if c in df.columns]

    print("\n" + "="*82)
    print(f"HGATE-PPO VALIDATION ({regime.upper()}) — N=17, paired episodes")
    print("="*82)
    print(f"\nMetrics: {metrics}\n")
    print("Per-scheduler summary (mean ± std):\n")
    print(df.groupby("scheduler")[metrics].agg(["mean", "std"]).round(3).to_string())

    ep_col = next((c for c in ["episode_id", "episode_idx", "ep_idx", "seed"]
                    if c in df.columns), None)
    print(f"\nPairing on column: '{ep_col}'\n")

    heft = df[df["scheduler"] == "HEFT"].sort_values(ep_col).reset_index(drop=True)
    hg   = df[df["scheduler"] == "HGATE-PPO"].sort_values(ep_col).reset_index(drop=True)
    s1   = df[df["scheduler"] == "Ours-auto_only-s1"].sort_values(ep_col).reset_index(drop=True)
    dt   = df[df["scheduler"] == "Decima-thermal"].sort_values(ep_col).reset_index(drop=True) \
            if "Decima-thermal" in df["scheduler"].values else None

    _paired_print("HGATE-PPO", hg, "HEFT", heft,
                  "HGATE-PPO vs HEFT — does Wu 2025 hetero-GAT + PPO beat the heuristic?",
                  metrics)
    if dt is not None and len(dt) > 0:
        print()
        _paired_print("HGATE-PPO", hg, "Decima-thermal", dt,
                      "HGATE-PPO vs Decima-thermal — §5 isolation point: +hetero edge typing on top of homog GCN + thermal reward",
                      metrics)
    if s1 is not None and len(s1) > 0:
        print()
        _paired_print("HGATE-PPO", hg, "Ours-auto_only-s1", s1,
                      "HGATE-PPO vs Ours-s1 — cost of removing RC edges + cross-attention placement actor",
                      metrics)

    # Pareto positioning
    print("\n" + "="*82)
    print(f"PARETO POSITIONING — {regime.upper()} REGIME")
    print("="*82)
    print(f"{'Scheduler':25s}  {'makespan':>14s}  {'peak_T':>14s}  "
          f"{'viol_rate':>10s}  {'completion':>10s}  {'ep_ret':>10s}")
    row_order = ["HEFT", "Decima-thermal", "HGATE-PPO", "Ours-auto_only-s1"]
    for sch in row_order:
        r = df[df["scheduler"] == sch]
        if not len(r):
            continue
        viol_rate = (r["violations_total"] > 0).mean() if "violations_total" in r else float("nan")
        print(f"{sch:25s}  "
              f"{r['total_makespan_ms'].mean():>7.1f}±{r['total_makespan_ms'].std():>5.1f}  "
              f"{r['peak_temp_episode'].mean():>7.2f}±{r['peak_temp_episode'].std():>5.2f}  "
              f"{viol_rate:>9.3f}  "
              f"{r['dag_completion_rate'].mean():>7.3f}   "
              f"{r['episode_return'].mean():>7.2f}")


if __name__ == "__main__":
    main()
