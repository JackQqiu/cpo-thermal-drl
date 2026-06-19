"""
smoke_lagrangian.py — local CPU smoke test for the Lagrangian/RCPO variant
=========================================================================

Verifies the constrained-PPO additions WITHOUT a full training run:

  Check 1 (no data): byte-identical Ours.  An enable_cost=False model and an
          enable_cost=True model built from the SAME seed share IDENTICAL
          weights on every common parameter (no RNG-stream shift); only the
          extra `critic.value_cost.*` head exists in the cost model.

  Check 2 (real model+env, FORCED cost): λ dynamics.  We drive the real
          model/buffer/trainer with a forced per-step cost of 1.0 (>> limit):
          the dual variable λ must rise from 0, and V_cost must train.
          Deterministic — isolates the controller from env stochasticity.

  Check 2b (FORCED cost = 0): λ must stay clipped at 0 (constraint slack).

  Check 3 (real env): Ours path inert.  lagrangian disabled ⇒ λ≡0,
          loss_value_cost≡0, runs end-to-end (no cost-path leakage).

  Check 4 (real env, hot regime): the env genuinely emits the cost signal.
          In a hot/loaded regime a fresh policy DOES breach the guardband,
          so info['would_violate_without_delay'] is present and fires, and λ
          rises from the real signal (full-integration check).

Run:  conda run -n cpo_rl python -m cpo_thermal_v2.scripts.smoke_lagrangian
Exit code 0 = all pass.
"""
from __future__ import annotations

import numpy as np
import torch

from cpo_thermal_v2.models import PPOActorCritic, build_batch
from cpo_thermal_v2.training.config_loader import load_config
from cpo_thermal_v2.training.env_factory import make_vector_env, _extract_per_env
from cpo_thermal_v2.training.ppo_trainer import PPOTrainer, make_optimizer
from cpo_thermal_v2.training.rollout_buffer import RolloutBuffer
from cpo_thermal_v2.training.train import (
    _extract_reward_channels, _extract_graph_obs_list, _extract_action_masks,
)

DEVICE = "cpu"
N_NODES = 17
NUM_ENVS = 2
T = 16


def check1_byte_identical_ours() -> None:
    print("\n[Check 1] byte-identical Ours (no RNG-stream shift) ...")
    torch.manual_seed(0)
    ours = PPOActorCritic(action_mode="hybrid", K_delay=5, enable_cost=False)
    torch.manual_seed(0)
    lag = PPOActorCritic(action_mode="hybrid", K_delay=5, enable_cost=True)
    sd_ours, sd_lag = ours.state_dict(), lag.state_dict()
    for k, v in sd_ours.items():
        assert k in sd_lag, f"key {k} missing from cost model"
        assert torch.equal(v, sd_lag[k]), f"param {k} differs (RNG shift!)"
    extra = set(sd_lag) - set(sd_ours)
    assert extra == {"critic.value_cost.weight", "critic.value_cost.bias"}, \
        f"unexpected extra keys: {extra}"
    assert not hasattr(ours.critic, "value_cost"), "Ours must NOT have value_cost"
    print(f"  ✅ {len(sd_ours)} shared params identical; cost model adds only {sorted(extra)}")


def _make_cfg(lagrangian: bool, hot: bool = False):
    cfg = load_config("cpo_thermal_v2/configs/default.yaml")
    cfg["env"]["num_nodes"] = N_NODES
    cfg["env"]["action_mode"] = "hybrid"
    if hot:
        cfg["env"]["initial_temp_range"] = [76.0, 80.0]   # at the guardband
        cfg["env"]["max_dag_size"] = 15
        cfg["env"]["dags_per_episode"] = 10
    else:
        cfg["env"]["initial_temp_range"] = [70.0, 75.0]
        cfg["env"]["max_dag_size"] = 5
        cfg["env"]["dags_per_episode"] = 4
    cfg["curriculum"]["enabled"] = False
    cfg["training"]["device"] = DEVICE
    cfg["training"]["num_envs"] = NUM_ENVS
    cfg["training"]["rollout_length"] = T
    if lagrangian:
        cfg["training"]["lagrangian"] = {
            "enabled": True, "cost_signal": "would_violate",
            "cost_limit": 0.1, "lam_init": 0.0, "lam_lr": 0.5, "lam_max": 10.0,
        }
    return cfg


def _run_loop(cfg, enable_cost, lagrangian, n_phases, force_cost=None):
    """Returns dict with per-phase lam/cost lists + real-signal diagnostics."""
    seed = 42
    np.random.seed(seed); torch.manual_seed(seed)
    vec = make_vector_env(cfg, num_envs=NUM_ENVS, seed_base=seed, mode="sync")
    model = PPOActorCritic(action_mode="hybrid", K_delay=cfg["env"]["K_delay"],
                           enable_cost=enable_cost).to(DEVICE)
    optim = make_optimizer(model, cfg["training"])
    lag = cfg["training"].get("lagrangian", {}) if lagrangian else {}
    trainer = PPOTrainer(
        model=model, optimizer=optim,
        clip_epsilon=cfg["training"]["clip_epsilon"], gamma=cfg["training"]["gamma"],
        gae_lambda=cfg["training"]["gae_lambda"], ppo_epochs=2, num_minibatches=2,
        max_grad_norm=cfg["training"]["max_grad_norm"], vf_coef=cfg["training"]["vf_coef"],
        ent_coef=cfg["training"]["ent_coef"], delay_loss_coef=1.0, device=DEVICE,
        action_mode="hybrid",
        lagrangian=lagrangian, cost_limit=float(lag.get("cost_limit", 0.0)),
        lam_init=float(lag.get("lam_init", 0.0)), lam_lr=float(lag.get("lam_lr", 0.0)),
        lam_max=float(lag.get("lam_max", 10.0)),
    )
    buf = RolloutBuffer(T, NUM_ENVS, action_mode="hybrid", device=DEVICE)
    _, info = vec.reset(seed=seed)
    g_list = _extract_graph_obs_list(info, NUM_ENVS)
    masks = _extract_action_masks(info, NUM_ENVS)

    lams, costs, vcosts = [], [], []
    real_viol_sum, n_cost_none = 0.0, 0
    for phase in range(n_phases):
        buf.reset()
        for _ in range(T):
            batch = build_batch(g_list, masks, device=DEVICE)
            with torch.no_grad():
                act = model.act(batch)
            nobs, rew, term, trunc, ninfo = vec.step(act["action"].cpu().numpy())
            done = np.logical_or(term, trunc).astype(np.float32)
            rp, rd = _extract_reward_channels(ninfo, NUM_ENVS)
            rc = None
            if lagrangian:
                if force_cost is not None:
                    rc = np.full(NUM_ENVS, float(force_cost), dtype=np.float32)
                else:
                    wv = _extract_per_env(ninfo, "would_violate_without_delay", NUM_ENVS)
                    rc = np.zeros(NUM_ENVS, dtype=np.float32)
                    for n in range(NUM_ENVS):
                        if wv[n] is None:
                            n_cost_none += 1
                        else:
                            rc[n] = float(wv[n])
                    real_viol_sum += float(rc.sum())
            buf.add(graph_obs=g_list, action_masks=masks, actions=act["action"],
                    log_probs=act["log_prob"], log_probs_p=act["log_prob_p"],
                    log_probs_d=act["log_prob_d"], values_p=act["v_placement"],
                    values_d=act["v_delay"],
                    values_cost=act["v_cost"] if lagrangian else None,
                    rewards_cost=rc, rewards_p=rp, rewards_d=rd, dones=done)
            g_list = _extract_graph_obs_list(ninfo, NUM_ENVS)
            masks = _extract_action_masks(ninfo, NUM_ENVS)
        with torch.no_grad():
            tb = build_batch(g_list, masks, device=DEVICE)
            vp, vd, vc = model.get_value(tb, include_cost=True)
        m = trainer.update(buf, vp, vd, last_values_cost=vc)
        lams.append(m.lam); costs.append(m.cost_per_episode); vcosts.append(m.loss_value_cost)
        print(f"    phase {phase}: lam={m.lam:.4f}  cost/ep={m.cost_per_episode:.3f}  "
              f"loss_value_cost={m.loss_value_cost:.4f}")
    vec.close()
    return dict(lams=lams, costs=costs, vcosts=vcosts,
                real_viol_sum=real_viol_sum, n_cost_none=n_cost_none)


def check2_lambda_rises_forced() -> None:
    print("\n[Check 2] λ dynamics — FORCED cost=1.0 (λ must rise) ...")
    r = _run_loop(_make_cfg(True), enable_cost=True, lagrangian=True,
                  n_phases=4, force_cost=1.0)
    assert r["lams"][-1] > 0.0, f"λ never rose: {r['lams']}"
    assert r["lams"][-1] >= r["lams"][0], f"λ not non-decreasing: {r['lams']}"
    assert max(r["vcosts"]) > 0.0, f"cost critic never trained: {r['vcosts']}"
    print(f"  ✅ λ: {r['lams'][0]:.3f} → {r['lams'][-1]:.3f}; cost critic active")


def check2b_lambda_stays_zero_forced() -> None:
    print("\n[Check 2b] λ dynamics — FORCED cost=0.0 (λ must stay 0) ...")
    r = _run_loop(_make_cfg(True), enable_cost=True, lagrangian=True,
                  n_phases=3, force_cost=0.0)
    assert all(l == 0.0 for l in r["lams"]), f"λ must stay 0 when cost<limit: {r['lams']}"
    print(f"  ✅ λ stayed clipped at 0 (constraint slack): {r['lams']}")


def check3_ours_path_inert() -> None:
    print("\n[Check 3] Ours path inert (lagrangian disabled) ...")
    r = _run_loop(_make_cfg(False), enable_cost=False, lagrangian=False, n_phases=2)
    assert all(l == 0.0 for l in r["lams"]), f"λ must stay 0 when disabled: {r['lams']}"
    assert all(v == 0.0 for v in r["vcosts"]), f"loss_value_cost must be 0: {r['vcosts']}"
    print(f"  ✅ ran end-to-end; λ≡0, loss_value_cost≡0 (no cost-path leakage)")


def check4_env_emits_real_cost() -> None:
    print("\n[Check 4] real env emits genuine violations (hot regime) ...")
    r = _run_loop(_make_cfg(True, hot=True), enable_cost=True, lagrangian=True, n_phases=4)
    print(f"    real would_violate sum over rollout = {r['real_viol_sum']:.0f}  "
          f"(None-extractions: {r['n_cost_none']})")
    assert r["n_cost_none"] == 0, \
        f"would_violate_without_delay key missing on {r['n_cost_none']} extractions"
    assert r["real_viol_sum"] > 0.0, \
        "env produced ZERO violations even in hot regime — cost signal would be dead in training"
    assert r["lams"][-1] > 0.0, f"λ never rose from real signal: {r['lams']}"
    print(f"  ✅ env emits real cost; λ rose from genuine signal to {r['lams'][-1]:.3f}")


if __name__ == "__main__":
    try:
        check1_byte_identical_ours()
        check2_lambda_rises_forced()
        check2b_lambda_stays_zero_forced()
        check3_ours_path_inert()
        check4_env_emits_real_cost()
    except Exception as e:
        print(f"\n❌ SMOKE FAILED: {type(e).__name__}: {e}")
        raise
    print("\n✅✅✅ ALL LAGRANGIAN SMOKE CHECKS PASSED")
