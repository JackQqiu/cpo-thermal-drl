"""
scripts/sweep_thermal_sensitivity.py — R2.1 thermal calibration-error sweep
===========================================================================

Paper revision R2.1.  Measures how a TRAINED policy's thermal safety
(violation rate, peak temperature) degrades as the RC thermal model is
mis-calibrated — the *true* calibration-error test:

    The agent uses the NOMINAL (calibrated) thermal values it trained on
    for its OBSERVATION, while the real PLANT differs.

This is implemented through the env's decoupled-mismatch hooks (added for
R2.1, see cpo_thermal_env.py constructor):

    rc_A / rc_B / rc_D        → PERTURBED plant   (used for stepping)
    rc_A_obs                  → NOMINAL A         (GNN RC-coupling edge obs)
    leakage_base_power        → PERTURBED plant leakage (stepping)
    leakage_obs_base_power    → NOMINAL leakage   (proc-feature obs)

So the policy *believes* it is on the calibrated system (it sees the
nominal RC-coupling edges and nominal leakage in its observation) while
the physics it is actually controlling has drifted.  This isolates the
robustness question reviewers asked: "what happens when your RC model is
wrong?"

Perturbation families (swept independently, one scalar family per run):

    G_cross   — scale G_CROSS_ASIC_OE & G_CROSS_OE_OE (chip-to-chip
                conduction).  Rebuilds A,B,D from generate_matrices.
    G_env     — scale G_ENV_ASIC & G_ENV_OE (chip-to-ambient conduction).
                Rebuilds A,B,D.
    leakage   — scale leakage_base_power only.  Matrices unchanged
                (nominal A used for both plant and observation).

Output:  a tidy CSV + small JSON summary with columns
    [param, delta_pct, scheduler, n_ep,
     violation_rate, peak_T_mean, peak_T_p95, makespan_mean]

Usage
-----
    PYTHONPATH=. conda run -n cpo_rl python -m \
        cpo_thermal_v2.scripts.sweep_thermal_sensitivity --episodes 200

    # quick end-to-end smoke
    PYTHONPATH=. conda run -n cpo_rl python -m \
        cpo_thermal_v2.scripts.sweep_thermal_sensitivity --smoke
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cpo_thermal_v2.data_pipeline import generate_matrices as gm
from cpo_thermal_v2.evaluation.runner import run_grid


# =====================================================================
# Constants for the N17 main topology
# =====================================================================
NUM_NODES = 17
NUM_OE = NUM_NODES - 1          # 16 OE + 1 ASIC
DT = 0.001                      # 1 ms RC step (matches default.yaml)

NOMINAL_LEAKAGE_BASE = 30.0     # default.yaml env.leakage_base_power
NOMINAL_LEAKAGE_BETA = 0.015    # default.yaml env.leakage_beta

# Default perturbation grid (includes 0.0 nominal control)
DEFAULT_DELTAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
DEFAULT_PARAMS = ["G_cross", "G_env", "leakage"]

OURS_HYBRID_CKPT = "checkpoints/stage2_hybrid_N17/best.pt"
NO_RC_EDGE_CKPT = "checkpoints/ours_no_rc_edge_N17/best.pt"

# Disk matrices for the nominal-vs-generated cross-check
DISK_MATRIX_DIR = "cpo_thermal_v2/data/thermal_matrics/N17"


# =====================================================================
# Nominal matrices (the calibration the policy trained on)
# =====================================================================
def build_nominal_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return (A0, B0, D0, used_disk).

    Build the nominal matrices from generate_matrices with default
    constants, then VERIFY against the on-disk N17 matrices the env would
    autodiscover.  If they mismatch, fall back to the disk matrices (so
    the "nominal observation" really equals what an unperturbed run would
    see) and flag it via ``used_disk``.
    """
    A0, B0, D0 = gm.build_state_space_matrices(NUM_OE, dt=DT)
    used_disk = False

    try:
        Ad = np.load(os.path.join(DISK_MATRIX_DIR, "matrix_A.npy")).astype(np.float32)
        Bd = np.load(os.path.join(DISK_MATRIX_DIR, "matrix_B.npy")).astype(np.float32)
        Dd = np.load(os.path.join(DISK_MATRIX_DIR, "matrix_D.npy")).astype(np.float32)
        if np.allclose(A0, Ad, atol=1e-6):
            print(f"[sweep] nominal A matches disk {DISK_MATRIX_DIR} "
                  f"(maxdiff={float(np.max(np.abs(A0 - Ad))):.2e}) ✓")
        else:
            print(f"[sweep] WARNING: generated nominal A != disk A "
                  f"(maxdiff={float(np.max(np.abs(A0 - Ad))):.2e}); "
                  f"using DISK matrices as nominal.")
            A0, B0, D0, used_disk = Ad, Bd, Dd, True
    except FileNotFoundError:
        print(f"[sweep] note: disk matrices not found under {DISK_MATRIX_DIR}; "
              f"using generated nominal (cannot cross-check).")

    return A0, B0, D0, used_disk


# =====================================================================
# Perturbed-plant matrix builder (G_cross / G_env families)
# =====================================================================
def build_perturbed_matrices(param: str, delta: float
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild (A, B, D) with the named conductance family scaled by
    (1 + delta), restoring the module constants afterwards.

    ``param`` ∈ {"G_cross", "G_env"}.
    """
    scale = 1.0 + delta

    if param == "G_cross":
        names = ["G_CROSS_ASIC_OE", "G_CROSS_OE_OE"]
    elif param == "G_env":
        names = ["G_ENV_ASIC", "G_ENV_OE"]
    else:
        raise ValueError(f"build_perturbed_matrices only handles G_cross / "
                         f"G_env, got {param!r}")

    # Snapshot, scale, rebuild, restore — keep the module pristine for the
    # next call (and so the nominal build above is never contaminated).
    saved = {n: getattr(gm, n) for n in names}
    try:
        for n in names:
            setattr(gm, n, saved[n] * scale)
        # build_state_space_matrices reads the module-level constants we
        # just mutated (build_conductance_matrix uses them directly).
        A, B, D = gm.build_state_space_matrices(NUM_OE, dt=DT)
    finally:
        for n in names:
            setattr(gm, n, saved[n])
    return A, B, D


# =====================================================================
# base env kwargs (mirror eval_scaling.yaml / default.yaml)
# =====================================================================
def make_base_env_kwargs(temp_lo: float, temp_hi: float,
                         dags_per_episode: int) -> Dict:
    """Construct the env kwargs shared by every cell.

    Mirrors default.yaml env defaults + eval_scaling.yaml per-cell
    overrides (hard truncation, fixed dags_per_episode, moderate-hot
    initial temperature).  The dataset_path is the repo-relative default
    the env loader resolves against the package root.
    """
    return dict(
        # topology / step
        num_nodes=NUM_NODES,
        dt=DT,
        # action mode is overlaid per-cell by run_grid (we pass hybrid)
        K_delay=5,
        delay_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        max_agent_delay_ms=50.0,
        # thermal thresholds (default.yaml)
        thermal_target=75.0,
        thermal_guardband=80.0,
        thermal_critical=85.0,
        mask_temp=82.0,
        # auto-cool
        max_cooling_steps=100,
        precool_target_temp=None,
        # power model (default.yaml) — PLANT leakage; overridden per-cell
        # for the leakage family.
        asic_active_power=150.0,
        oe_active_power=40.0,
        oe_serialization_power=20.0,
        cpo_bandwidth_gbps=800.0,
        oe_conversion_delay_ms=0.005,
        leakage_base_power=NOMINAL_LEAKAGE_BASE,
        leakage_beta=NOMINAL_LEAKAGE_BETA,
        # temp-rise heuristics (default.yaml)
        temp_rise_per_ms_asic=0.08,
        temp_rise_per_ms_oe=0.18,
        # evaluation regime (eval_scaling.yaml)
        initial_temp_range=(temp_lo, temp_hi),
        max_dag_size=None,
        dags_per_episode=dags_per_episode,
        truncate_mode="hard",
        soft_truncate_recovery_temp=75.0,
        max_soft_recovery_steps=200,
        # data
        dataset_path="./data_pipeline/process/alibaba_dags_v2.json",
        rc_matrix_dir=None,
    )


# =====================================================================
# Scheduler factories
# =====================================================================
def build_scheduler_factories(device: str, deterministic: bool,
                              ckpt: str = OURS_HYBRID_CKPT,
                              label: str = "Ours-hybrid"
                              ) -> List[Tuple[str, object]]:
    """Build the scheduler factory list.

    Primary: Ours-hybrid (the RC-edge model, action_mode='hybrid').

    The no-RC-edge ablation (checkpoints/ours_no_rc_edge_N17) is the
    natural architectural contrast, BUT it is an ``auto_only`` checkpoint
    trained with a now-removed ``disable_rc_edge`` env flag (it appears in
    that ckpt's resolved_config.yaml but no longer exists in the env API,
    so it would be silently dropped — the model would see RC-coupling
    edges it was never trained on).  Running it in this *hybrid* paired
    sweep would therefore be an out-of-distribution / mode-mismatched
    comparison, not the clean contrast it is meant to be.  We deliberately
    run Ours-hybrid only and print this note; see the script docstring.
    """
    from cpo_thermal_v2.baselines import TrainedPPOScheduler

    factories: List[Tuple[str, object]] = []

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"[sweep] primary checkpoint missing: {ckpt}")

    def ours_factory(num_nodes: int, action_mode: str):
        if action_mode != "hybrid":
            raise ValueError(f"{label} only runs at action_mode=hybrid")
        return TrainedPPOScheduler(
            ckpt_path=ckpt,
            action_mode="hybrid",
            deterministic=deterministic,
            device=device,
            scheduler_label=label,
        )
    factories.append((label, ours_factory))

    # Explicit note about the no-RC-edge ablation decision.
    if os.path.exists(NO_RC_EDGE_CKPT):
        print(f"[sweep] NOTE: {NO_RC_EDGE_CKPT} exists but is NOT run. It is "
              f"an auto_only ckpt trained with the removed `disable_rc_edge` "
              f"env flag; running it in this hybrid paired sweep would be a "
              f"mode-mismatched / out-of-distribution comparison. Running "
              f"Ours-hybrid only (see docstring).")
    else:
        print(f"[sweep] NOTE: no-RC-edge ckpt not found at {NO_RC_EDGE_CKPT}; "
              f"running Ours-hybrid only.")

    return factories


# =====================================================================
# Aggregation
# =====================================================================
def aggregate_cell(df_eps: pd.DataFrame, param: str, delta: float) -> List[Dict]:
    """Collapse a single-cell df_episodes into per-scheduler summary rows.

    Uses the EpisodeRecord column names from evaluation/metrics.py:
        scheduler, violations_total, peak_temp_episode, total_makespan_ms.
    A "violation" episode = violations_total > 0 (any thermal violation).
    """
    rows: List[Dict] = []
    if df_eps.empty:
        return rows
    for sched, sub in df_eps.groupby("scheduler"):
        viol = sub["violations_total"].to_numpy(dtype=float)
        peak = sub["peak_temp_episode"].to_numpy(dtype=float)
        mk = sub["total_makespan_ms"].to_numpy(dtype=float)
        rows.append(dict(
            param=param,
            delta_pct=round(delta * 100.0, 4),
            scheduler=str(sched),
            n_ep=int(len(sub)),
            violation_rate=float(np.mean(viol > 0)),
            peak_T_mean=float(np.mean(peak)),
            peak_T_p95=float(np.percentile(peak, 95)),
            makespan_mean=float(np.mean(mk)),
        ))
    return rows


# =====================================================================
# Main sweep
# =====================================================================
def run_sweep(params: List[str], deltas: List[float], num_episodes: int,
              seed_base: int, device: str, deterministic: bool,
              temp_lo: float, temp_hi: float, dags_per_episode: int,
              out_csv: str, ckpt: str = OURS_HYBRID_CKPT,
              label: str = "Ours-hybrid") -> pd.DataFrame:
    A0, B0, D0, used_disk = build_nominal_matrices()
    base_kwargs = make_base_env_kwargs(temp_lo, temp_hi, dags_per_episode)
    factories = build_scheduler_factories(device, deterministic, ckpt, label)

    work_dir = os.path.dirname(os.path.abspath(out_csv)) or "."
    os.makedirs(work_dir, exist_ok=True)
    grid_scratch = os.path.join(work_dir, "_grid_scratch")

    all_rows: List[Dict] = []
    n_cells = len(params) * len(deltas)
    cell_i = 0

    for param in params:
        for delta in deltas:
            cell_i += 1
            print(f"\n========== cell {cell_i}/{n_cells}: "
                  f"param={param} delta={delta:+.2f} "
                  f"(n_ep={num_episodes}) ==========")

            env_kwargs = dict(base_kwargs)

            if param == "leakage":
                # Plant leakage scaled; observation leakage stays nominal.
                # Matrices stay nominal (both plant and obs).
                env_kwargs["rc_A"] = A0
                env_kwargs["rc_B"] = B0
                env_kwargs["rc_D"] = D0
                env_kwargs["rc_A_obs"] = A0          # obs == nominal (same here)
                env_kwargs["leakage_base_power"] = NOMINAL_LEAKAGE_BASE * (1.0 + delta)
                env_kwargs["leakage_obs_base_power"] = NOMINAL_LEAKAGE_BASE
                env_kwargs["leakage_obs_beta"] = NOMINAL_LEAKAGE_BETA
            else:
                # G_cross / G_env: perturb the plant matrices, keep the
                # observation edges nominal (rc_A_obs = A0). Leakage nominal.
                A_p, B_p, D_p = build_perturbed_matrices(param, delta)
                env_kwargs["rc_A"] = A_p
                env_kwargs["rc_B"] = B_p
                env_kwargs["rc_D"] = D_p
                env_kwargs["rc_A_obs"] = A0          # NOMINAL observation
                env_kwargs["leakage_base_power"] = NOMINAL_LEAKAGE_BASE
                env_kwargs["leakage_obs_base_power"] = NOMINAL_LEAKAGE_BASE
                env_kwargs["leakage_obs_beta"] = NOMINAL_LEAKAGE_BETA

            df_eps, _df_dags = run_grid(
                base_env_kwargs=env_kwargs,
                scheduler_factories=factories,
                num_nodes_list=[NUM_NODES],
                action_mode_list=["hybrid"],
                num_episodes=num_episodes,
                output_dir=grid_scratch,
                seed_base=seed_base,           # SAME across all cells → paired
                verbose=True,
            )

            cell_rows = aggregate_cell(df_eps, param, delta)
            all_rows.extend(cell_rows)
            for r in cell_rows:
                print(f"   -> {r['scheduler']}: viol_rate={r['violation_rate']:.3f} "
                      f"peak_T_mean={r['peak_T_mean']:.2f} "
                      f"peak_T_p95={r['peak_T_p95']:.2f} "
                      f"makespan_mean={r['makespan_mean']:.1f}")

            # incremental save
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_csv, index=False)

    # JSON summary
    summary = dict(
        params=params,
        deltas=deltas,
        num_episodes=num_episodes,
        seed_base=seed_base,
        deterministic=deterministic,
        initial_temp_range=[temp_lo, temp_hi],
        dags_per_episode=dags_per_episode,
        nominal_matrices_from="disk" if used_disk else "generate_matrices",
        schedulers=[f[0] for f in factories],
        rows=all_rows,
    )
    out_json = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[sweep] wrote {out_csv}")
    print(f"[sweep] wrote {out_json}")
    return df_out


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="R2.1 thermal calibration-error sensitivity sweep "
                    "(decoupled nominal-obs / perturbed-plant).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes", type=int, default=200,
                   help="Episodes per (param, delta) cell.")
    p.add_argument("--params", nargs="+", default=DEFAULT_PARAMS,
                   choices=["G_cross", "G_env", "leakage"],
                   help="Perturbation families to sweep.")
    p.add_argument("--deltas", nargs="+", type=float, default=DEFAULT_DELTAS,
                   help="Perturbation magnitudes (fractional, e.g. 0.3 = +30%%).")
    p.add_argument("--seed-base", type=int, default=100000,
                   help="Paired seed base; SAME across all cells.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device for the trained policy.")
    p.add_argument("--no-deterministic", action="store_true",
                   help="Sample actions instead of argmax (default argmax).")
    p.add_argument("--temp-lo", type=float, default=60.0,
                   help="Initial-temp range low (hot regime exposes the "
                        "policy's thermal margin so mis-calibration is "
                        "visible).")
    p.add_argument("--temp-hi", type=float, default=75.0,
                   help="Initial-temp range high.")
    p.add_argument("--dags-per-episode", type=int, default=20,
                   help="DAGs per episode (eval_scaling.yaml uses 20).")
    p.add_argument("--out", type=str,
                   default="repro_outputs/sweep_thermal_sensitivity.csv",
                   help="Output CSV path (JSON summary written alongside).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny end-to-end run: param=G_cross, deltas={0.0,+0.3}, "
                        "5 episodes. Overrides --params/--deltas/--episodes "
                        "and writes to repro_outputs/sweep_smoke.csv.")
    p.add_argument("--ckpt", type=str, default=OURS_HYBRID_CKPT,
                   help="Policy checkpoint to sweep (default Ours-hybrid; pass "
                        "the Lagrangian ckpt to compare its calibration "
                        "tolerance against Ours).")
    p.add_argument("--label", type=str, default="Ours-hybrid",
                   help="Scheduler label written into the output rows.")
    args = p.parse_args()

    if args.smoke:
        params = ["G_cross"]
        deltas = [0.0, 0.3]
        episodes = 5
        out_csv = "repro_outputs/sweep_smoke.csv"
        print("[sweep] SMOKE mode: param=G_cross, deltas=[0.0, +0.3], "
              "episodes=5")
    else:
        params = args.params
        deltas = args.deltas
        episodes = args.episodes
        out_csv = args.out

    df = run_sweep(
        params=params,
        deltas=deltas,
        num_episodes=episodes,
        seed_base=args.seed_base,
        device=args.device,
        deterministic=not args.no_deterministic,
        temp_lo=args.temp_lo,
        temp_hi=args.temp_hi,
        dags_per_episode=args.dags_per_episode,
        out_csv=out_csv,
        ckpt=args.ckpt,
        label=args.label,
    )

    # Pretty final table
    print("\n================ RESULT TABLE ================")
    if not df.empty:
        cols = ["param", "delta_pct", "scheduler", "n_ep",
                "violation_rate", "peak_T_mean", "peak_T_p95", "makespan_mean"]
        with pd.option_context("display.width", 160,
                               "display.max_columns", None):
            print(df[cols].to_string(index=False))
    else:
        print("(empty)")


if __name__ == "__main__":
    main()
