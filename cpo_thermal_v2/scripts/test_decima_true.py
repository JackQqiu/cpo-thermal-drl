#!/usr/bin/env python3
"""
test_decima_true.py — per-step acceptance tests for Decima true.

Maps 1:1 to paper_drafts/decima_true_implementation_checklist.md.
Each test prints "[PASS]" or "[FAIL: <reason>]" and contributes to an
overall exit code (0 = all pass).

Run:
    PYTHONPATH=. python cpo_thermal_v2/scripts/test_decima_true.py
"""
from __future__ import annotations

import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

# pytest is optional — when present, decorators on the cuda test become
# native skipif marks.  When absent, the cuda test raises Skipped at
# runtime which the custom driver translates to a [SKIP] result.
try:
    import pytest                                              # noqa: F401
    _PYTEST = True
except ImportError:                                           # pragma: no cover
    _PYTEST = False


class _Skip(Exception):
    """Raised inside a test to indicate it should be skipped rather than
    pass/fail.  Used in lieu of pytest.skip() when pytest is unavailable.
    """

from cpo_thermal_v2.baselines.decima_true import (
    DecimaTruePolicy, DecimaTrueAgent,
)
from cpo_thermal_v2.envs.reward_shaping import (
    RewardConfig, compute_reward,
)


# =====================================================================
# Test harness
# =====================================================================
_RESULTS: List[Tuple[str, bool, str]] = []


def _make_dummy_obs(N_task: int = 4, N_proc: int = 5,
                    task_in: int = 8, proc_in: int = 7,
                    n_edges: int = 5) -> Dict[str, Any]:
    """Synth a graph_obs dict matching cpo_thermal_env's schema."""
    rng = np.random.default_rng(0)
    edges = []
    for _ in range(n_edges):
        u, v = rng.integers(0, N_task, size=2)
        if u != v:
            edges.append([int(u), int(v)])
    return {
        "proc_x":           rng.standard_normal((N_proc, proc_in)).tolist(),
        "task_x":           rng.standard_normal((N_task, task_in)).tolist(),
        "edges_t2t":        edges,
        "edges_t2t_attr":   [[0.5] for _ in edges],
        "edges_p2p":        [],
        "edges_p2p_attr":   [],
        "edges_t2p":        [],
        "edges_t2p_attr":   [],
        "current_task_idx": 0,
        "task_id_order":    [str(i) for i in range(N_task)],
        "num_uncompleted":  N_task,
    }


def _run(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(f"[PASS] {name}")
        _RESULTS.append((name, True, ""))
    except _Skip as e:
        msg = str(e)
        print(f"[SKIP] {name}: {msg}")
        _RESULTS.append((name, True, f"SKIP: {msg}"))     # skip counts as pass
    except AssertionError as e:
        msg = str(e) or "assertion failed"
        print(f"[FAIL] {name}: {msg}")
        _RESULTS.append((name, False, msg))
    except Exception as e:                              # pragma: no cover
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


# =====================================================================
# Step 1 — Homogeneous GCN encoder
# =====================================================================
def test_step1_gcn_encoder() -> None:
    """forward returns finite logits of shape (N_proc,)."""
    policy = DecimaTruePolicy(hidden_dim=64, num_gcn_layers=3)
    obs = _make_dummy_obs(N_task=6, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    logits = policy.forward(obs, mask)
    assert isinstance(logits, torch.Tensor), \
        f"forward must return Tensor, got {type(logits)}"
    assert logits.shape == (5,), \
        f"expected logits shape (5,), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), \
        f"logits contain non-finite values: {logits}"


# =====================================================================
# Step 2 — Mao-style score function (returns per-proc scores via
#         task-conditioned head)
# =====================================================================
def test_step2_score_function() -> None:
    """Different current_task_idx changes logits (proves conditioning)."""
    torch.manual_seed(0)
    policy = DecimaTruePolicy(hidden_dim=64, num_gcn_layers=3)
    obs1 = _make_dummy_obs(N_task=6, N_proc=5)
    obs2 = dict(obs1)
    obs2["current_task_idx"] = 3   # different ready task
    mask = np.array([True] * 5, dtype=bool)
    l1 = policy.forward(obs1, mask)
    l2 = policy.forward(obs2, mask)
    # Logits should differ because the Stage-B head sees a different
    # cur_task embedding.  Allowing for the (vanishing) chance that two
    # task embeddings are bit-identical, require at least one position
    # to differ by > 1e-5.
    delta = (l1 - l2).abs().max().item()
    assert delta > 1e-5, (
        f"logits did not change when current_task_idx changed "
        f"(max |Δ| = {delta:.2e}) — score head not conditioning on task")


# =====================================================================
# Step 3 — Action sampling + masking
# =====================================================================
def test_step3_action_masking() -> None:
    """Sampled actions always satisfy the action mask; argmax is deterministic."""
    torch.manual_seed(42)
    policy = DecimaTruePolicy(hidden_dim=64, num_gcn_layers=3)
    obs = _make_dummy_obs(N_task=6, N_proc=7)

    # mask out procs 0, 2, 5 — only 1, 3, 4, 6 are valid
    mask = np.array([False, True, False, True, True, False, True], dtype=bool)
    valid = set(np.where(mask)[0].tolist())

    # 200 stochastic samples — every action must be in `valid`
    for _ in range(200):
        a, lp, ent = policy.select_action(obs, mask, deterministic=False)
        assert a in valid, f"sampled invalid action {a} (valid={valid})"
        assert torch.isfinite(lp), f"log_prob is non-finite: {lp}"
        assert torch.isfinite(ent) and ent.item() >= -1e-6, \
            f"entropy invalid: {ent}"

    # determinism: argmax must repeat
    a1, _, _ = policy.select_action(obs, mask, deterministic=True)
    a2, _, _ = policy.select_action(obs, mask, deterministic=True)
    assert a1 == a2, f"deterministic action differs across calls: {a1} vs {a2}"


# =====================================================================
# Step 4 — REINFORCE agent (loss is finite + non-zero; baseline updates)
# =====================================================================
def test_step4_reinforce_update() -> None:
    """One update() returns finite loss + baseline starts updating."""
    torch.manual_seed(0)
    policy = DecimaTruePolicy(hidden_dim=32, num_gcn_layers=2)
    agent = DecimaTrueAgent(policy, lr=1e-3, baseline_window=10)

    # Synthesise one rollout
    obs = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.ones(5, dtype=bool)
    log_probs, entropies, rewards = [], [], []
    for t in range(10):
        a, lp, ent = policy.select_action(obs, mask, deterministic=False)
        log_probs.append(lp)
        entropies.append(ent)
        rewards.append(1.0 if t == 9 else 0.0)   # sparse: +1 at the end

    assert agent.baseline == 0.0, "baseline should start at 0"
    metrics = agent.update(log_probs, rewards, entropies=entropies)
    assert np.isfinite(metrics["loss"]), \
        f"loss not finite: {metrics['loss']}"
    assert metrics["n_steps"] == 10, \
        f"expected n_steps=10, got {metrics['n_steps']}"
    # baseline should reflect this episode's mean discounted return
    assert agent.baseline != 0.0, \
        "baseline did not update after one episode"


def test_step4_reinforce_baseline_window() -> None:
    """Baseline approximates moving average over the configured window."""
    policy = DecimaTruePolicy(hidden_dim=16, num_gcn_layers=2)
    agent = DecimaTrueAgent(policy, lr=1e-3, baseline_window=5)
    # Drive 7 episodes with known per-episode returns
    for ep_ret in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]:
        agent.record_episode_return(ep_ret)
    # baseline should be mean of last 5 = (30+40+50+60+70)/5 = 50.0
    assert abs(agent.baseline - 50.0) < 1e-6, \
        f"baseline window broken: got {agent.baseline}, want 50.0"


# =====================================================================
# Step 5 — train.py dispatch picks up algorithm: reinforce
# =====================================================================
def test_step5_train_dispatch_imports() -> None:
    """train_decima_true.train_reinforce is importable; train.py routes it."""
    from cpo_thermal_v2.training.train_decima_true import train_reinforce
    assert callable(train_reinforce), \
        f"train_reinforce not callable: {train_reinforce}"
    # Inspect main() for the dispatch line
    import inspect
    from cpo_thermal_v2.training import train as _train_mod
    src = inspect.getsource(_train_mod.main)
    assert "algorithm" in src and "reinforce" in src, (
        f"train.main() does not appear to dispatch on training.algorithm "
        f"== 'reinforce' — source excerpt:\n{src}")
    assert "train_decima_true" in src, (
        f"train.main() does not import train_decima_true under the "
        f"reinforce branch")


# =====================================================================
# Step 6 — env reward_mode + observation_keys filter
# =====================================================================
def test_step6_reward_mode_makespan_only_drops_thermal() -> None:
    """makespan_only is thermally insensitive; thermal_aware is sensitive.

    Stronger than a single-Δ check: vary ``max_temp_during`` and
    ``cooling_used`` across two regimes; makespan_only must return
    identical reward at both (proving every thermal term is dropped),
    while thermal_aware must differ.
    """
    cfg_ma = RewardConfig(reward_mode="makespan_only")
    cfg_th = RewardConfig(reward_mode="thermal_aware")

    # base kwargs (no termination, mid-episode step)
    base: Dict[str, Any] = dict(
        exec_time     = 30.0, base_workload = 30.0,
        final_temps   = np.array([70.0]*5),
        target_node   = 0,
        dag_done      = False, episode_done = False, truncated = False,
    )
    cold = dict(base, max_temp_during=60.0, cooling_used=0.0,
                would_violate_without_delay=False)
    hot  = dict(base, max_temp_during=84.0, cooling_used=50.0,
                would_violate_without_delay=False)

    r_ma_cold = compute_reward(cfg=cfg_ma, **cold)
    r_ma_hot  = compute_reward(cfg=cfg_ma, **hot)
    r_th_cold = compute_reward(cfg=cfg_th, **cold)
    r_th_hot  = compute_reward(cfg=cfg_th, **hot)

    # makespan_only insensitive to temperature/cooling
    assert abs(r_ma_cold - r_ma_hot) < 1e-9, (
        f"makespan_only reward shifted across thermal regimes: "
        f"cold={r_ma_cold} hot={r_ma_hot} (delta={r_ma_cold - r_ma_hot})")
    # And it equals exactly base_step_reward in a no-termination step
    assert abs(r_ma_cold - cfg_ma.base_step_reward) < 1e-9, (
        f"makespan_only mid-episode reward = {r_ma_cold}, "
        f"expected base_step_reward = {cfg_ma.base_step_reward}")
    # thermal_aware DOES differ across regimes (soft wall + cooling penalty)
    assert abs(r_th_cold - r_th_hot) > 1.0, (
        f"thermal_aware reward should differ between cold/hot regimes; "
        f"cold={r_th_cold} hot={r_th_hot} (delta={r_th_cold - r_th_hot})")


def test_step6_reward_mode_truncate_still_fires() -> None:
    """Truncation penalty fires under both modes."""
    cfg_ma = RewardConfig(reward_mode="makespan_only")
    cfg_th = RewardConfig(reward_mode="thermal_aware")
    kw = dict(
        exec_time=0.0, base_workload=0.0, max_temp_during=85.0,
        final_temps=np.array([85.0]*5), target_node=0,
        cooling_used=0.0, would_violate_without_delay=False,
        dag_done=False, episode_done=False, truncated=True,
    )
    r_ma = compute_reward(cfg=cfg_ma, **kw)
    r_th = compute_reward(cfg=cfg_th, **kw)
    assert r_ma == -float(cfg_ma.truncate_pen), \
        f"makespan_only truncate broken: {r_ma}"
    assert r_th == -float(cfg_th.truncate_pen), \
        f"thermal_aware truncate broken: {r_th}"


def test_step6_observation_keys_filter() -> None:
    """env.observation_keys filters _build_graph_obs output."""
    try:
        from cpo_thermal_v2.envs.cpo_thermal_env import CPOThermalDAGEnvV2
    except Exception as e:
        # Skip if env can't even be imported (dataset missing, etc.)
        raise AssertionError(f"env import failed: {e}")

    import os
    candidates = [
        "./data_pipeline/process/alibaba_dags_v2.json",
        "./cpo_thermal_v2/data_pipeline/process/alibaba_dags_v2.json",
    ]
    ds_path = next((p for p in candidates if os.path.exists(p)), None)
    if ds_path is None:
        raise AssertionError(
            f"dataset not found in any of: {candidates}; "
            f"observation_keys test requires it")

    env = CPOThermalDAGEnvV2(
        dataset_path=ds_path,
        num_nodes=17,
        observation_keys=["task_x", "edges_t2t"],
    )
    obs, info = env.reset(seed=0)
    g = info["graph_obs"][0] if isinstance(info["graph_obs"], np.ndarray) \
        else info["graph_obs"]
    # mandatory structural keys preserved
    for k in ["current_task_idx", "num_uncompleted", "task_id_order"]:
        assert k in g, f"structural key {k!r} was stripped"
    # requested keys present
    assert "task_x" in g, "task_x missing despite being in observation_keys"
    assert "edges_t2t" in g, "edges_t2t missing"
    # thermal-leaking keys stripped
    for k in ["proc_x", "edges_p2p", "edges_p2p_attr",
              "edges_t2p", "edges_t2p_attr", "edges_t2t_attr"]:
        assert k not in g, (
            f"observation_keys filter did not strip {k!r}; "
            f"present keys = {sorted(g.keys())}")


# =====================================================================
# HK-3.1.1 — Device discipline regression tests
# =====================================================================
# Catches the bug where DecimaTruePolicy + forward + agent build CPU
# tensors regardless of where the model parameters live (the V100 0%
# GPU-Util pilot symptom).  After HK-3.1.1's fix:
#   - meta test: forward must NOT raise "Tensor on device * is not on
#     the expected device cpu!" — that exact message is the bug
#     signature.  If forward fully completes, output device must match
#     the model's parameter device.
#   - cuda test (skipif no cuda): hard regression check on the actual
#     V100 deployment device.

# Exact substring of the pre-fix error (PyTorch wording stable across 1.x/2.x)
_PRE_FIX_ERROR = "not on the expected device cpu"


def _device_of(p: DecimaTruePolicy) -> torch.device:
    return next(p.parameters()).device


def test_policy_respects_meta_device() -> None:
    """Bug regression: after moving policy to 'meta', forward must not
    create CPU tensors that fight the meta-resident parameters.

    Acceptance:
      - either forward returns without RuntimeError AND output.device == meta
      - or forward raises an error whose message does NOT contain
        ``"not on the expected device cpu"`` (the pre-fix signature).
    """
    policy = DecimaTruePolicy(hidden_dim=16, num_gcn_layers=2).to("meta")
    assert _device_of(policy).type == "meta", \
        "test setup failed: policy did not move to meta"

    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.ones(5, dtype=bool)
    try:
        logits = policy.forward(obs, mask)
    except RuntimeError as e:
        msg = str(e)
        assert _PRE_FIX_ERROR not in msg, (
            f"forward still has CPU-tensor leak (HK-3.1.1 bug "
            f"signature present):\n  {msg}")
        # Some other RuntimeError (e.g. torch_geometric scatter not
        # supported on meta) is acceptable — the device-discipline
        # regression we care about is the cpu-leak path.
        return
    # If forward completed cleanly, output must respect model device.
    assert logits.device.type == "meta", (
        f"forward succeeded but output device is {logits.device}, "
        f"expected meta — model device was not propagated to output")


def test_policy_respects_cuda_device() -> None:
    """End-to-end CUDA regression for the V100 deployment path.

    Skipped on systems without CUDA (e.g. Mac dev box).  Hard regression
    check on the actual V100 deployment device.

    Acceptance: model on cuda, forward returns logits on cuda, sampled
    action is a valid masked int, log_prob and entropy are finite cuda
    tensors.
    """
    if not torch.cuda.is_available():
        raise _Skip("CUDA not available on this host (runs on V100 only)")

    policy = DecimaTruePolicy(hidden_dim=16, num_gcn_layers=2).to("cuda")
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True, True, False, True, True], dtype=bool)

    logits = policy.forward(obs, mask)
    assert logits.device.type == "cuda", \
        f"forward output device {logits.device}, expected cuda"
    assert torch.isfinite(logits).all(), f"non-finite cuda logits: {logits}"

    # Stochastic + deterministic both honour mask + return cuda tensors
    a_sto, lp_sto, ent_sto = policy.select_action(obs, mask,
                                                  deterministic=False)
    a_det, lp_det, ent_det = policy.select_action(obs, mask,
                                                  deterministic=True)
    assert mask[a_sto] and mask[a_det], \
        f"sampled invalid actions: sto={a_sto} det={a_det} mask={mask}"
    for tname, t in [("log_prob_sto", lp_sto), ("entropy_sto", ent_sto),
                     ("log_prob_det", lp_det), ("entropy_det", ent_det)]:
        assert t.device.type == "cuda", \
            f"{tname}.device = {t.device}, expected cuda"
        assert torch.isfinite(t), f"{tname} non-finite: {t}"


# =====================================================================
# Driver
# =====================================================================
def main() -> int:
    print("=" * 72)
    print("Decima true — per-step acceptance tests")
    print("=" * 72)

    print("\n-- Step 1: Homogeneous GCN encoder --")
    _run("forward shape + finiteness", test_step1_gcn_encoder)

    print("\n-- Step 2: Mao-style score function --")
    _run("score head conditions on current task", test_step2_score_function)

    print("\n-- Step 3: Action sampling + masking --")
    _run("mask honoured + argmax deterministic", test_step3_action_masking)

    print("\n-- Step 4: REINFORCE agent --")
    _run("update() finite + baseline updates", test_step4_reinforce_update)
    _run("baseline window arithmetic",         test_step4_reinforce_baseline_window)

    print("\n-- Step 5: train.py dispatch --")
    _run("algorithm: reinforce dispatch wired", test_step5_train_dispatch_imports)

    print("\n-- Step 6: env reward_mode + observation_keys --")
    _run("makespan_only drops thermal terms",        test_step6_reward_mode_makespan_only_drops_thermal)
    _run("truncate penalty fires under both modes",  test_step6_reward_mode_truncate_still_fires)
    _run("observation_keys filter strips graph_obs", test_step6_observation_keys_filter)

    print("\n-- HK-3.1.1: device discipline regression --")
    _run("policy respects meta device (no cpu leak)", test_policy_respects_meta_device)
    _run("policy respects cuda device (V100 path)",   test_policy_respects_cuda_device)

    print()
    print("=" * 72)
    n_skip = sum(1 for _, ok, msg in _RESULTS if ok and msg.startswith("SKIP"))
    n_pass = sum(1 for _, ok, msg in _RESULTS if ok and not msg.startswith("SKIP"))
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"SUMMARY:  {n_pass} passed / {n_skip} skipped / {n_fail} failed  "
          f"({len(_RESULTS)} total)")
    if n_fail:
        print("\nFailures:")
        for name, ok, msg in _RESULTS:
            if not ok:
                print(f"  - {name}: {msg}")
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
