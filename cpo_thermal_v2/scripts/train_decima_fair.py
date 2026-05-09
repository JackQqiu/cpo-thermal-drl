"""
scripts/train_decima_fair.py — Train Decima from scratch, no thermal info
==========================================================================

Trains a fair Decima reimplementation:
  - Same GIN encoder architecture as Ours (HeteroEncoder)
  - proc_in_dim = 3      (drops 4 thermal proc features)
  - edge_dim_t2p = 1     (drops est_temp_rise from task→proc edges)
  - env.thermal_blind = true triggers _install_thermal_blind_patch in
    env_factory, which monkey-patches env._build_graph_obs to strip
    thermal info from info['graph_obs'] at every step (proc_x 7→3,
    edges_t2p_attr 2→1)

Reward is unchanged (still uses thermal violation penalty).  The
fairness requirement is about INPUT FEATURES only — Decima would learn
from the same makespan + safety reward signal that any deployment
provides; the question is whether thermal features in the OBSERVATION
are needed.

Output:
  checkpoints/decima_fair_N17/best.pt
  tb_logs/decima_fair_N17/

Usage on V100::

    sbatch cpo_thermal_v2/scripts/train_decima_fair.sbatch

Or local::

    PYTHONPATH=. python -m cpo_thermal_v2.scripts.train_decima_fair

This is a thin wrapper.  The actual fairness logic is in:
  - configs/train_decima_fair.yaml   (proc_in_dim=3, edge_dim_t2p=1, thermal_blind=true)
  - training/env_factory.py          (_install_thermal_blind_patch)
  - training/train.py                (reads proc_in_dim/edge_dim_t2p from config)
  - models/ppo_actor_critic.py       (accepts edge_dim_t2p as parameter)
"""
from __future__ import annotations

import sys


def main():
    if "--config" not in sys.argv:
        sys.argv += ["--config",
                     "cpo_thermal_v2/configs/train_decima_fair.yaml"]
    from cpo_thermal_v2.training.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
