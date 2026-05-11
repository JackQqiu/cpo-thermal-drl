#!/usr/bin/env python3
"""
test_hgate_ppo.py — per-step acceptance tests for HGATE-PPO baseline.

Maps to paper_drafts/hgate_ppo_checklist.md Step 1 (Hetero GATv2
encoder) initially; later steps add tests at the bottom of this file.

Run:
    PYTHONPATH=. python cpo_thermal_v2/scripts/test_hgate_ppo.py
"""
from __future__ import annotations

import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cpo_thermal_v2.baselines.hgate_ppo import (
    HGATEEncoder, HGATEActorCritic, HGATEPPOScheduler,
)


# =====================================================================
# Skip mechanism (pytest-optional, same shape as test_decima_true.py)
# =====================================================================
try:
    import pytest                                              # noqa: F401
    _PYTEST = True
except ImportError:                                           # pragma: no cover
    _PYTEST = False


class _Skip(Exception):
    """Raised inside a test to indicate it should be skipped."""


# =====================================================================
# Fixture: synthetic graph_obs matching cpo_thermal_env schema
# =====================================================================
def _make_dummy_obs(
    N_task: int = 6,
    N_proc: int = 5,
    task_in:    int = 8,
    proc_in:    int = 7,
    n_t2t:      int = 5,
    fully_ready: bool = True,
) -> Dict[str, Any]:
    """Synthesise a graph_obs dict whose shape matches what
    cpo_thermal_env._build_graph_obs produces.

    ``fully_ready=True`` populates edges_t2p covering all (ready_task,
    proc) pairs so message passing has signal in tests.  Sets
    edges_t2p_attr to 2 columns so the encoder's RC-stripping path is
    actually exercised.
    """
    rng = np.random.default_rng(0)
    edges_t2t: List[List[int]] = []
    for _ in range(n_t2t):
        u, v = rng.integers(0, N_task, size=2)
        if u != v:
            edges_t2t.append([int(u), int(v)])

    edges_t2p: List[List[int]] = []
    edges_t2p_attr: List[List[float]] = []
    if fully_ready:
        for t in range(N_task):
            for p in range(N_proc):
                edges_t2p.append([t, p])
                # 2 cols: [est_time_norm, est_temp_rise_norm]
                # The encoder MUST keep est_time and drop est_temp_rise
                edges_t2p_attr.append([float(rng.uniform(0.1, 1.0)),
                                       float(rng.uniform(0.05, 0.5))])

    return {
        "proc_x":           rng.standard_normal((N_proc, proc_in)).tolist(),
        "task_x":           rng.standard_normal((N_task, task_in)).tolist(),
        "edges_t2t":        edges_t2t,
        "edges_t2t_attr":   [[0.5] for _ in edges_t2t],
        "edges_p2p":        [],         # HGATE doesn't use these
        "edges_p2p_attr":   [],
        "edges_t2p":        edges_t2p,
        "edges_t2p_attr":   edges_t2p_attr,
        "current_task_idx": 0,
        "task_id_order":    [str(i) for i in range(N_task)],
        "num_uncompleted":  N_task,
    }


# =====================================================================
# Driver harness
# =====================================================================
_RESULTS: List[Tuple[str, bool, str]] = []


def _run(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(f"[PASS] {name}")
        _RESULTS.append((name, True, ""))
    except _Skip as e:
        msg = str(e)
        print(f"[SKIP] {name}: {msg}")
        _RESULTS.append((name, True, f"SKIP: {msg}"))
    except AssertionError as e:
        msg = str(e) or "assertion failed"
        print(f"[FAIL] {name}: {msg}")
        _RESULTS.append((name, False, msg))
    except Exception as e:                              # pragma: no cover
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


# =====================================================================
# Step 1 — HGATEEncoder.__init__
# =====================================================================
def test_step1_init_module_shapes() -> None:
    """__init__ creates task_proj / proc_proj with correct dims."""
    enc = HGATEEncoder(task_in_dim=8, proc_in_dim=7,
                        hidden_dim=64, num_layers=3, num_heads=2)
    assert isinstance(enc.task_proj, torch.nn.Linear), \
        f"task_proj missing or wrong type: {type(getattr(enc, 'task_proj', None))}"
    assert enc.task_proj.in_features == 8
    assert enc.task_proj.out_features == 64
    assert isinstance(enc.proc_proj, torch.nn.Linear)
    assert enc.proc_proj.in_features == 7
    assert enc.proc_proj.out_features == 64


def test_step1_layer_count_and_norms() -> None:
    """GAT stack + LayerNorms have length = num_layers, per node type."""
    enc = HGATEEncoder(hidden_dim=64, num_layers=3, num_heads=4)
    # Layer count contract: there are num_layers separate GAT stages
    # spanning the two edge types. The module exposes them via
    # parallel ModuleLists for explicit-bipartite handling.
    assert hasattr(enc, "t2t_convs"), \
        "encoder must expose t2t_convs (task-task GATv2 stack)"
    assert hasattr(enc, "t2p_convs"), \
        "encoder must expose t2p_convs (task-proc GATv2 stack)"
    assert len(enc.t2t_convs) == 3, \
        f"t2t_convs has len {len(enc.t2t_convs)}, expected 3"
    assert len(enc.t2p_convs) == 3, \
        f"t2p_convs has len {len(enc.t2p_convs)}, expected 3"
    assert hasattr(enc, "task_norms") and len(enc.task_norms) == 3
    assert hasattr(enc, "proc_norms") and len(enc.proc_norms) == 3


def test_step1_forward_output_shape() -> None:
    """enc(obs) returns finite tensors of shape (T, H), (P, H)."""
    enc = HGATEEncoder(hidden_dim=32, num_layers=2)
    obs = _make_dummy_obs(N_task=6, N_proc=5)
    task_embs, proc_embs = enc(obs)
    assert isinstance(task_embs, torch.Tensor) and isinstance(proc_embs, torch.Tensor), \
        f"forward must return two tensors; got {type(task_embs)}, {type(proc_embs)}"
    assert task_embs.shape == (6, 32), f"task_embs.shape = {tuple(task_embs.shape)}, expected (6, 32)"
    assert proc_embs.shape == (5, 32), f"proc_embs.shape = {tuple(proc_embs.shape)}, expected (5, 32)"
    assert torch.isfinite(task_embs).all(), "task_embs has non-finite values"
    assert torch.isfinite(proc_embs).all(), "proc_embs has non-finite values"


def test_step1_gradient_flow() -> None:
    """Every learnable parameter receives a gradient after forward + backward."""
    enc = HGATEEncoder(hidden_dim=32, num_layers=2)
    obs = _make_dummy_obs(N_task=4, N_proc=5)
    task_embs, proc_embs = enc(obs)
    loss = task_embs.sum() + proc_embs.sum()
    loss.backward()
    missing = [n for n, p in enc.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad: {missing}"


def test_step1_drops_rc_edge_attribute() -> None:
    """HGATE controlled-comparison contract: encoder must NOT see
    est_temp_rise (the 2nd column of edges_t2p_attr) and must NOT
    consume edges_p2p (the RC thermal coupling)."""
    enc = HGATEEncoder(hidden_dim=32, num_layers=1)
    obs = _make_dummy_obs(N_task=4, N_proc=5)

    # Capture the actual edge_attr the adapter exposes
    device = next(enc.parameters()).device
    out = enc._graph_obs_to_hetero(obs, device)
    # Adapter contract: must yield (task_x, proc_x, edge_t2t,
    # edge_t2p, edge_t2p_attr) in that order
    assert isinstance(out, tuple) and len(out) == 5, (
        f"_graph_obs_to_hetero must return a 5-tuple; got {type(out)} "
        f"of len {len(out) if hasattr(out, '__len__') else '?'}")
    _, _, edge_t2t, edge_t2p, edge_t2p_attr = out
    assert edge_t2p_attr.dim() == 2, \
        f"edge_t2p_attr must be 2-D, got {edge_t2p_attr.dim()}-D shape {tuple(edge_t2p_attr.shape)}"
    assert edge_t2p_attr.shape[-1] == 1, (
        f"edge_t2p_attr.shape[-1] = {edge_t2p_attr.shape[-1]}, expected 1 "
        f"(HGATE keeps est_time only, drops est_temp_rise — controlled-"
        f"comparison anchor with Ours)")

    # Second check: feeding an obs with proc<->proc edges must not
    # change the output (encoder ignores edges_p2p entirely).
    obs_with_p2p = dict(obs)
    obs_with_p2p["edges_p2p"]      = [[0, 1], [1, 2]]
    obs_with_p2p["edges_p2p_attr"] = [[0.5], [0.7]]
    torch.manual_seed(0); enc1 = HGATEEncoder(hidden_dim=32, num_layers=1)
    torch.manual_seed(0); enc2 = HGATEEncoder(hidden_dim=32, num_layers=1)
    t1, p1 = enc1(obs)
    t2, p2 = enc2(obs_with_p2p)
    assert torch.allclose(t1, t2), \
        "task_embs differ between obs with/without edges_p2p — encoder is reading p2p"
    assert torch.allclose(p1, p2), \
        "proc_embs differ between obs with/without edges_p2p — encoder is reading p2p"


# =====================================================================
# Step 2 — HGATEActorCritic (__init__ + forward)
# =====================================================================
def test_step2_init_attaches_encoder_actor_critic() -> None:
    """__init__ wires HGATEEncoder + actor scorer + critic head."""
    m = HGATEActorCritic(hidden_dim=64, num_procs=5,
                          num_heads=2, num_gat_layers=2,
                          task_in_dim=8, proc_in_dim=7)
    assert hasattr(m, "encoder"), "ActorCritic must expose .encoder"
    assert isinstance(m.encoder, HGATEEncoder), \
        f"encoder type wrong: {type(m.encoder).__name__}"
    assert m.encoder.hidden_dim == 64
    assert m.encoder.num_layers == 2
    assert m.encoder.num_heads == 2
    assert hasattr(m, "actor_score"), \
        "ActorCritic must expose .actor_score (Decision 1: per-pair Path B)"
    assert hasattr(m, "critic"), \
        "ActorCritic must expose .critic (single scalar value head)"


def test_step2_actor_score_input_dim_is_two_hidden() -> None:
    """Path B per-pair scorer: input is [global_ctx, proc_emb_i] -> 2*hidden_dim."""
    hidden_dim = 64
    m = HGATEActorCritic(hidden_dim=hidden_dim, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    # Find the first Linear inside actor_score; its in_features must be
    # 2 * hidden_dim per Decision 1 (per-pair scoring).
    linears = [mod for mod in m.actor_score.modules()
               if isinstance(mod, torch.nn.Linear)]
    assert len(linears) >= 1, "actor_score has no Linear layers"
    assert linears[0].in_features == 2 * hidden_dim, (
        f"actor_score first Linear in_features = {linears[0].in_features}, "
        f"expected 2*{hidden_dim} = {2*hidden_dim} (per Decision 1: per-pair "
        f"scoring takes [global_ctx, proc_emb_i])")
    assert linears[-1].out_features == 1, (
        f"actor_score final out_features = {linears[-1].out_features}, "
        f"expected 1 (one scalar per processor)")


def test_step2_critic_is_single_scalar_head() -> None:
    """Single-value critic — NOT dual placement/delay (that's Ours)."""
    hidden_dim = 64
    m = HGATEActorCritic(hidden_dim=hidden_dim, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    linears = [mod for mod in m.critic.modules()
               if isinstance(mod, torch.nn.Linear)]
    assert len(linears) >= 1, "critic has no Linear layers"
    assert linears[0].in_features == hidden_dim, (
        f"critic first Linear in_features = {linears[0].in_features}, "
        f"expected {hidden_dim} (operates on pooled global context)")
    assert linears[-1].out_features == 1, (
        f"critic final out_features = {linears[-1].out_features}, "
        f"expected 1 (single value head — Wu 2025 spec, not dual critic)")


def test_step2_forward_logits_and_value_shape() -> None:
    """forward(obs, mask) returns (logits[N_proc], value[scalar]), both finite."""
    m = HGATEActorCritic(hidden_dim=32, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    obs  = _make_dummy_obs(N_task=6, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    logits, value = m.forward(obs, mask)
    assert isinstance(logits, torch.Tensor), \
        f"logits type wrong: {type(logits).__name__}"
    assert isinstance(value, torch.Tensor), \
        f"value type wrong: {type(value).__name__}"
    assert logits.shape == (5,), \
        f"logits.shape = {tuple(logits.shape)}, expected (5,)"
    # value must collapse to scalar (shape ()) per checklist Step 2 acceptance
    assert value.dim() == 0 or value.shape == torch.Size([1]) or value.numel() == 1, (
        f"value must be scalar; got shape {tuple(value.shape)}, numel={value.numel()}")
    assert torch.isfinite(logits).all(), "logits contain non-finite values"
    assert torch.isfinite(value).all(), "value is non-finite"


def test_step2_gradient_flow_through_full_actor_critic() -> None:
    """backward through (logits.sum() + value) populates grads on every param,
    including the encoder (no broken graph through Path B scoring)."""
    m = HGATEActorCritic(hidden_dim=32, num_procs=5,
                          num_heads=2, num_gat_layers=2)
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    logits, value = m.forward(obs, mask)
    loss = logits.sum() + value.sum()    # .sum() collapses scalar or (1,)
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, (
        f"params with no grad after backward: {missing[:5]}"
        f"{' ...' if len(missing) > 5 else ''}")


def test_step2_actor_critic_respects_meta_device() -> None:
    """HK-3.1.1 carryover applied to the new heads: moving the whole
    ActorCritic to 'meta' must not surface the pre-fix cpu-leak error."""
    m = HGATEActorCritic(hidden_dim=16, num_procs=5,
                          num_heads=2, num_gat_layers=1).to("meta")
    assert next(m.parameters()).device.type == "meta"
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    try:
        logits, value = m.forward(obs, mask)
    except RuntimeError as e:
        assert _PRE_FIX_ERROR not in str(e), (
            f"actor-critic has CPU-tensor leak (HK-3.1.1 signature):\n  {e}")
        return
    assert logits.device.type == "meta", \
        f"logits.device = {logits.device}, expected meta"
    assert value.device.type == "meta", \
        f"value.device = {value.device}, expected meta"


# =====================================================================
# Step 3 — HGATEActorCritic.act + get_value
# =====================================================================
def test_step3_act_returns_expected_dict_keys() -> None:
    """act() returns dict with action / log_prob / entropy / value."""
    m = HGATEActorCritic(hidden_dim=32, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    out = m.act(obs, mask, deterministic=False)
    assert isinstance(out, dict), f"act() must return dict, got {type(out).__name__}"
    for key in ("action", "log_prob", "entropy", "value"):
        assert key in out, f"act() output missing key {key!r}; got {sorted(out.keys())}"
    # action must be a python int (or 0-d int tensor) — see contract
    a = out["action"]
    if isinstance(a, torch.Tensor):
        assert a.dim() == 0, f"action tensor must be scalar; got shape {tuple(a.shape)}"
    else:
        assert isinstance(a, int), f"action must be int or 0-d tensor; got {type(a).__name__}"
    # log_prob / entropy / value all finite tensors
    for k in ("log_prob", "entropy", "value"):
        t = out[k]
        assert isinstance(t, torch.Tensor), f"{k} not a tensor: {type(t).__name__}"
        assert torch.isfinite(t).all(), f"{k} is non-finite: {t}"


def test_step3_act_mask_honored_1000_samples() -> None:
    """1000 stochastic samples must all satisfy action_mask."""
    torch.manual_seed(42)
    m = HGATEActorCritic(hidden_dim=32, num_procs=7,
                          num_heads=2, num_gat_layers=1)
    obs = _make_dummy_obs(N_task=4, N_proc=7)
    # Mask out half the procs
    mask = np.array([False, True, False, True, True, False, True], dtype=bool)
    valid = set(np.where(mask)[0].tolist())
    for _ in range(1000):
        out = m.act(obs, mask, deterministic=False)
        a = out["action"]
        a_int = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
        assert a_int in valid, (
            f"act() sampled invalid action {a_int} not in {valid} "
            f"(mask = {mask.tolist()})")


def test_step3_act_deterministic_repeatable() -> None:
    """deterministic=True must produce the same action across repeated calls."""
    m = HGATEActorCritic(hidden_dim=32, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([False, True, True, False, True], dtype=bool)
    a1 = m.act(obs, mask, deterministic=True)["action"]
    a2 = m.act(obs, mask, deterministic=True)["action"]
    a3 = m.act(obs, mask, deterministic=True)["action"]
    aint = lambda x: int(x.item()) if isinstance(x, torch.Tensor) else int(x)
    assert aint(a1) == aint(a2) == aint(a3), (
        f"deterministic action varies across calls: {aint(a1)} / {aint(a2)} / {aint(a3)}")


def test_step3_get_value_returns_scalar_finite() -> None:
    """get_value(obs) returns finite scalar tensor; signature accepts NO mask."""
    m = HGATEActorCritic(hidden_dim=32, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    obs = _make_dummy_obs(N_task=4, N_proc=5)
    v = m.get_value(obs)
    assert isinstance(v, torch.Tensor), f"get_value must return tensor; got {type(v).__name__}"
    assert v.dim() == 0 or v.numel() == 1, \
        f"get_value must return scalar; got shape {tuple(v.shape)} numel={v.numel()}"
    assert torch.isfinite(v).all(), f"get_value returned non-finite: {v}"


def test_step3_get_value_agrees_with_forward() -> None:
    """get_value(obs) must equal forward(obs, mask)[1] for the same obs."""
    m = HGATEActorCritic(hidden_dim=32, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    m.eval()
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    with torch.no_grad():
        v_from_get   = m.get_value(obs)
        _, v_from_fwd = m.forward(obs, mask)
    # Both should be the same scalar value (encoder + critic + pool are
    # deterministic and shared between the two code paths).
    diff = (v_from_get.flatten() - v_from_fwd.flatten()).abs().max().item()
    assert diff < 1e-6, (
        f"get_value and forward()[1] disagree by {diff:.2e}; "
        f"either critic path drifted or get_value re-ran actor by mistake")


def test_step3_act_get_value_respect_meta_device() -> None:
    """HK-3.1.1 carryover for the new Step-3 methods."""
    m = HGATEActorCritic(hidden_dim=16, num_procs=5,
                          num_heads=2, num_gat_layers=1).to("meta")
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    for entry_label, fn in [
        ("act",       lambda: m.act(obs, mask, deterministic=True)),
        ("get_value", lambda: m.get_value(obs)),
    ]:
        try:
            out = fn()
        except RuntimeError as e:
            assert _PRE_FIX_ERROR not in str(e), (
                f"{entry_label}() has CPU-tensor leak (HK-3.1.1 signature):\n  {e}")
            continue
        # When meta-forward succeeds, every returned tensor must be on meta.
        if isinstance(out, dict):
            for k, t in out.items():
                if isinstance(t, torch.Tensor):
                    assert t.device.type == "meta", \
                        f"act()[{k}].device = {t.device}, expected meta"
        elif isinstance(out, torch.Tensor):
            assert out.device.type == "meta", \
                f"{entry_label}().device = {out.device}, expected meta"


# =====================================================================
# Step 8 — HGATEPPOScheduler (eval-time wrapper)
# =====================================================================
def test_step8_scheduler_loads_ckpt_and_schedules() -> None:
    """Save a fresh ActorCritic state_dict, load via Scheduler, schedule()
    returns a valid masked int that satisfies the action mask."""
    import tempfile, os
    torch.manual_seed(0)
    inner = HGATEActorCritic(hidden_dim=16, num_procs=5,
                              num_heads=2, num_gat_layers=1,
                              task_in_dim=8, proc_in_dim=7)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "tiny.pt")
        torch.save({"model": inner.state_dict(),
                    "global_step": 0,
                    "metrics_summary": {"ep_ret_mean": 0.0}}, p)

        from cpo_thermal_v2.baselines.hgate_ppo import HGATEPPOScheduler
        sch = HGATEPPOScheduler(
            ckpt_path=p,
            num_nodes=5,
            deterministic=True,
            device="cpu",
            hidden_dim=16,
            num_gat_layers=1,
            num_heads=2,
        )
        assert sch.name == "HGATE-PPO", f"name = {sch.name!r}"
        assert sch.action_mode == "auto_only"

        # Construct an info dict matching what cpo_thermal_env emits
        obs = _make_dummy_obs(N_task=4, N_proc=5)
        info = {
            "graph_obs":   [obs],
            "action_mask": np.array([True, False, True, True, False],
                                     dtype=bool),
        }
        action = sch.schedule(None, info)
        assert isinstance(action, int), \
            f"schedule must return int (auto_only); got {type(action).__name__}"
        assert 0 <= action < 5, f"action {action} out of range"
        assert info["action_mask"][action], \
            f"schedule returned masked-out action {action}; " \
            f"mask = {info['action_mask'].tolist()}"


def test_step8_scheduler_deterministic_repeatable() -> None:
    """deterministic=True in scheduler ctor -> same schedule output across calls."""
    import tempfile, os
    torch.manual_seed(1)
    inner = HGATEActorCritic(hidden_dim=16, num_procs=5,
                              num_heads=2, num_gat_layers=1)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "tiny.pt")
        torch.save({"model": inner.state_dict()}, p)

        from cpo_thermal_v2.baselines.hgate_ppo import HGATEPPOScheduler
        sch = HGATEPPOScheduler(ckpt_path=p, num_nodes=5,
                                  deterministic=True, device="cpu",
                                  hidden_dim=16, num_gat_layers=1, num_heads=2)
        obs = _make_dummy_obs(N_task=4, N_proc=5)
        info = {"graph_obs": [obs],
                "action_mask": np.array([False, True, True, True, True], dtype=bool)}
        a1 = sch.schedule(None, info)
        a2 = sch.schedule(None, info)
        a3 = sch.schedule(None, info)
        assert a1 == a2 == a3, \
            f"deterministic scheduler varies across calls: {a1} / {a2} / {a3}"


# =====================================================================
# Step 4 (HK-4.5 perf-fix) — multi-env rollout + minibatch PPO update
# =====================================================================
def _make_dummy_rollout(M: int = 8, num_procs: int = 5, device="cpu"):
    """Build a tiny RolloutBatch in-memory with M flat transitions.
    Uses the SAME synthetic graph_obs schema as Step 1-3 tests.
    """
    from cpo_thermal_v2.training.train_hgate_ppo import RolloutBatch

    rng = np.random.default_rng(7)
    graph_obs = [_make_dummy_obs(N_task=4, N_proc=num_procs) for _ in range(M)]
    masks     = [np.array([True] * num_procs, dtype=bool) for _ in range(M)]
    actions   = rng.integers(0, num_procs, size=M).astype(np.int64)
    old_log_probs = torch.full((M,), -float(np.log(num_procs)),
                                dtype=torch.float32, device=device)
    # advantages already normalised (mean 0, std 1)
    raw_adv = rng.standard_normal(M).astype(np.float32)
    raw_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
    advantages = torch.tensor(raw_adv, dtype=torch.float32, device=device)
    returns    = torch.tensor(rng.standard_normal(M).astype(np.float32),
                              dtype=torch.float32, device=device)
    return RolloutBatch(
        graph_obs     = graph_obs,
        masks         = masks,
        actions       = actions,
        old_log_probs = old_log_probs,
        advantages    = advantages,
        returns       = returns,
    )


def test_step4perf_ppo_update_returns_kl_clipfrac_entropy() -> None:
    """_ppo_update must return a metrics dict containing approx_kl,
    clip_frac, entropy (with finite values), in addition to losses.
    These are the §Fix-2 metrics that replace the loss_pg=0 logging
    artifact."""
    from cpo_thermal_v2.training.train_hgate_ppo import _ppo_update

    m = HGATEActorCritic(hidden_dim=16, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    rollout = _make_dummy_rollout(M=8, num_procs=5)
    metrics = _ppo_update(
        model=m, optimizer=opt, rollout=rollout,
        ppo_epochs=2, minibatch_size=4,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
        max_grad_norm=0.5, device=torch.device("cpu"),
    )
    for k in ("loss_pg", "loss_v", "loss_ent",
              "approx_kl", "clip_frac", "entropy"):
        assert k in metrics, f"_ppo_update metrics missing {k!r}; got {sorted(metrics.keys())}"
        v = metrics[k]
        assert isinstance(v, float), f"metrics[{k!r}] not float: {type(v).__name__}"
        assert np.isfinite(v), f"metrics[{k!r}] is non-finite: {v}"
    # approx_kl should be >= 0 (Schulman approximation, non-negative)
    assert metrics["approx_kl"] >= -1e-6, \
        f"approx_kl is unexpectedly negative: {metrics['approx_kl']}"
    # clip_frac is a probability
    assert 0.0 <= metrics["clip_frac"] <= 1.0, \
        f"clip_frac out of [0,1]: {metrics['clip_frac']}"
    # entropy should be > 0 (we sample from non-degenerate softmax)
    assert metrics["entropy"] > 0.0, \
        f"entropy is non-positive: {metrics['entropy']}"


def test_step4perf_ppo_update_one_optim_step_per_minibatch() -> None:
    """With M=8 flat transitions, mb_size=4, ppo_epochs=3:
    expect 3 epochs × 2 minibatches = 6 optimizer.step() calls
    (NOT 3 × 8 = 24, which would be the old per-transition pattern)."""
    from cpo_thermal_v2.training.train_hgate_ppo import _ppo_update

    m = HGATEActorCritic(hidden_dim=16, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    # Count optimizer.step calls
    real_step = opt.step
    step_calls = [0]
    def _counted_step(*args, **kwargs):
        step_calls[0] += 1
        return real_step(*args, **kwargs)
    opt.step = _counted_step

    rollout = _make_dummy_rollout(M=8, num_procs=5)
    _ppo_update(
        model=m, optimizer=opt, rollout=rollout,
        ppo_epochs=3, minibatch_size=4,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
        max_grad_norm=0.5, device=torch.device("cpu"),
    )
    expected = 3 * (8 // 4)  # ppo_epochs × num_minibatches
    assert step_calls[0] == expected, (
        f"optimizer.step called {step_calls[0]} times, expected {expected} "
        f"(ppo_epochs=3 × num_minibatches=2).  The §Fix-B pattern is one "
        f"backward+step per minibatch, NOT per transition.")


# =====================================================================
# Step 5 (HK-4.5.2 batched-forward) — batched API bit-identical contract
# =====================================================================
def _make_diverse_obs_list(N: int, num_procs: int, seed: int = 11):
    """Build N graph_obs with VARIABLE task counts (real env property)
    and CONSTANT proc count.  This is the worst case for batching — if
    bit-identical equivalence holds here, it holds for the env."""
    rng = np.random.default_rng(seed)
    obs_list = []
    for i in range(N):
        T_i = int(rng.integers(3, 9))     # variable task count per graph
        obs_list.append(_make_dummy_obs(N_task=T_i, N_proc=num_procs))
    return obs_list


def test_step5perf_forward_batched_matches_sequential() -> None:
    """forward_batched on N graphs must produce (logits, values)
    bit-identical to N sequential forward() calls.  Tolerance ≤ 1e-5
    (float32 reduction-order noise allowed but no semantic drift)."""
    torch.manual_seed(0)
    m = HGATEActorCritic(hidden_dim=32, num_procs=5,
                          num_heads=2, num_gat_layers=2)
    m.eval()
    obs_list = _make_diverse_obs_list(N=6, num_procs=5)
    masks = [np.array([True] * 5, dtype=bool) for _ in obs_list]

    # Sequential reference
    seq_logits: List[torch.Tensor] = []
    seq_values: List[torch.Tensor] = []
    with torch.no_grad():
        for go, mk in zip(obs_list, masks):
            lg, vl = m.forward(go, mk)
            seq_logits.append(lg)
            seq_values.append(vl.view(-1))
    seq_logits_t = torch.stack(seq_logits, dim=0)   # (N, P)
    seq_values_t = torch.cat (seq_values)           # (N,)

    # Batched candidate
    assert hasattr(m, "forward_batched"), \
        "HGATEActorCritic must expose forward_batched(graph_obs_list)"
    with torch.no_grad():
        out = m.forward_batched(obs_list)
    assert "logits" in out and "values" in out, \
        f"forward_batched must return dict with logits, values; got {sorted(out.keys())}"
    bat_logits = out["logits"]
    bat_values = out["values"]
    assert tuple(bat_logits.shape) == (6, 5), \
        f"forward_batched logits shape = {tuple(bat_logits.shape)}, expected (6, 5)"
    assert tuple(bat_values.shape) == (6,), \
        f"forward_batched values shape = {tuple(bat_values.shape)}, expected (6,)"

    diff_logits = (seq_logits_t - bat_logits).abs().max().item()
    diff_values = (seq_values_t - bat_values).abs().max().item()
    assert diff_logits < 1e-5, (
        f"forward_batched logits drift: max |seq - batched| = {diff_logits:.2e}; "
        f"expected ≤ 1e-5 (encoder semantics must be bit-identical)")
    assert diff_values < 1e-5, (
        f"forward_batched values drift: max |seq - batched| = {diff_values:.2e}; "
        f"expected ≤ 1e-5")


def test_step5perf_get_value_batched_matches_sequential() -> None:
    """get_value_batched on N graphs must match N sequential get_value calls."""
    torch.manual_seed(1)
    m = HGATEActorCritic(hidden_dim=24, num_procs=5,
                          num_heads=2, num_gat_layers=2)
    m.eval()
    obs_list = _make_diverse_obs_list(N=5, num_procs=5)

    with torch.no_grad():
        seq = torch.stack([m.get_value(go).view(-1) for go in obs_list]).view(-1)
    assert hasattr(m, "get_value_batched"), \
        "HGATEActorCritic must expose get_value_batched(graph_obs_list)"
    with torch.no_grad():
        bat = m.get_value_batched(obs_list)
    assert tuple(bat.shape) == (5,), \
        f"get_value_batched shape = {tuple(bat.shape)}, expected (5,)"
    diff = (seq - bat).abs().max().item()
    assert diff < 1e-5, f"get_value_batched drift: {diff:.2e}"


def test_step5perf_evaluate_actions_batched_matches_sequential() -> None:
    """evaluate_actions_batched is the function the PPO update uses inside
    minibatch.  For B stored actions, it must produce (new_log_prob,
    entropy, value) bit-identical to B sequential forwards."""
    torch.manual_seed(2)
    m = HGATEActorCritic(hidden_dim=32, num_procs=5,
                          num_heads=2, num_gat_layers=2)
    m.eval()
    B = 7
    obs_list = _make_diverse_obs_list(N=B, num_procs=5)
    masks = [np.array([True, True, False, True, True], dtype=bool)
              for _ in obs_list]
    actions = np.array([1, 0, 3, 4, 1, 0, 3], dtype=np.int64)

    # Sequential reference: re-run forward + masking + log_softmax + gather
    seq_lp, seq_ent, seq_v = [], [], []
    with torch.no_grad():
        for i in range(B):
            lg, vl = m.forward(obs_list[i], masks[i])
            mask_t = torch.tensor(masks[i], dtype=torch.bool)
            masked = lg.masked_fill(~mask_t, float("-inf"))
            log_probs = F.log_softmax(masked, dim=-1)
            probs     = log_probs.exp()
            seq_lp.append(log_probs[int(actions[i])])
            seq_ent.append(-(probs[mask_t] * log_probs[mask_t]).sum())
            seq_v .append(vl.view(()))
    seq_lp_t  = torch.stack(seq_lp )
    seq_ent_t = torch.stack(seq_ent)
    seq_v_t   = torch.stack(seq_v  )

    assert hasattr(m, "evaluate_actions_batched"), \
        "HGATEActorCritic must expose evaluate_actions_batched(...)"
    with torch.no_grad():
        bat_lp, bat_ent, bat_v = m.evaluate_actions_batched(
            obs_list, masks, actions)
    for name, seq_t, bat_t in [
        ("new_log_prob", seq_lp_t,  bat_lp ),
        ("entropy",      seq_ent_t, bat_ent),
        ("value",        seq_v_t,   bat_v  ),
    ]:
        assert tuple(bat_t.shape) == (B,), \
            f"evaluate_actions_batched {name} shape = {tuple(bat_t.shape)}, expected ({B},)"
        diff = (seq_t - bat_t).abs().max().item()
        assert diff < 1e-5, f"evaluate_actions_batched {name} drift: {diff:.2e}"


def test_step5perf_act_batched_deterministic_matches_sequential() -> None:
    """deterministic=True act_batched must match N sequential
    deterministic act() calls — actions, log_probs, entropies, values
    all bit-identical."""
    torch.manual_seed(3)
    m = HGATEActorCritic(hidden_dim=24, num_procs=7,
                          num_heads=2, num_gat_layers=1)
    m.eval()
    obs_list = _make_diverse_obs_list(N=5, num_procs=7)
    masks = [
        np.array([True, True, False, True, True, False, True], dtype=bool)
        for _ in obs_list
    ]

    seq_act, seq_lp, seq_ent, seq_v = [], [], [], []
    with torch.no_grad():
        for go, mk in zip(obs_list, masks):
            o = m.act(go, mk, deterministic=True)
            seq_act.append(int(o["action"]))
            seq_lp .append(o["log_prob"].view(()))
            seq_ent.append(o["entropy" ].view(()))
            seq_v  .append(o["value"   ].view(()))
    seq_act_t = torch.tensor(seq_act, dtype=torch.long)
    seq_lp_t  = torch.stack(seq_lp )
    seq_ent_t = torch.stack(seq_ent)
    seq_v_t   = torch.stack(seq_v  )

    assert hasattr(m, "act_batched"), \
        "HGATEActorCritic must expose act_batched(graph_obs_list, masks)"
    with torch.no_grad():
        bat = m.act_batched(obs_list, masks, deterministic=True)
    for k in ("actions", "log_probs", "entropies", "values"):
        assert k in bat, f"act_batched output missing {k!r}; got {sorted(bat.keys())}"
    assert tuple(bat["actions"].shape) == (5,), \
        f"act_batched actions shape wrong: {tuple(bat['actions'].shape)}"
    # Bit-identical for deterministic path
    assert torch.equal(bat["actions"], seq_act_t), \
        f"deterministic actions differ: seq={seq_act_t.tolist()} bat={bat['actions'].tolist()}"
    for name, seq_t, bat_t in [
        ("log_probs", seq_lp_t,  bat["log_probs"]),
        ("entropies", seq_ent_t, bat["entropies"]),
        ("values",    seq_v_t,   bat["values"   ]),
    ]:
        diff = (seq_t - bat_t).abs().max().item()
        assert diff < 1e-5, f"act_batched {name} drift: {diff:.2e}"


def test_step5perf_act_batched_mask_honored_stochastic() -> None:
    """act_batched with deterministic=False must respect masks across
    all envs in the batch (no leakage of masked-out actions)."""
    torch.manual_seed(4)
    m = HGATEActorCritic(hidden_dim=16, num_procs=5,
                          num_heads=2, num_gat_layers=1)
    m.eval()
    N = 8
    obs_list = _make_diverse_obs_list(N=N, num_procs=5)
    # Different mask per env
    masks = []
    for i in range(N):
        mk = np.array([False, True, True, False, True], dtype=bool)
        if i % 2 == 1:
            mk = np.array([True, False, True, True, False], dtype=bool)
        masks.append(mk)
    valid_sets = [set(np.where(mk)[0].tolist()) for mk in masks]

    with torch.no_grad():
        for _ in range(200):
            out = m.act_batched(obs_list, masks, deterministic=False)
            acts = out["actions"].cpu().numpy().tolist()
            for n, a in enumerate(acts):
                assert int(a) in valid_sets[n], (
                    f"env {n} sampled invalid action {a} not in {valid_sets[n]} "
                    f"(mask = {masks[n].tolist()})")


# =====================================================================
# Device discipline regression tests (HK-3.1.1 pattern, carried over)
# =====================================================================
_PRE_FIX_ERROR = "not on the expected device cpu"


def test_encoder_respects_meta_device() -> None:
    """Mac-runnable: move encoder to 'meta', forward must not raise the
    'expected device cpu' bug signature.  Catches the HK-3.1.1-class
    device leak in local CI without needing CUDA."""
    enc = HGATEEncoder(hidden_dim=16, num_layers=1).to("meta")
    assert next(enc.parameters()).device.type == "meta", \
        "test setup failed: encoder did not move to meta"
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    try:
        task_embs, proc_embs = enc(obs)
    except RuntimeError as e:
        msg = str(e)
        assert _PRE_FIX_ERROR not in msg, (
            f"encoder forward has CPU-tensor leak (HK-3.1.1 signature):\n  {msg}")
        return   # other RuntimeError (e.g. meta unsupported in PyG) ok
    # If forward succeeded, outputs must be on meta
    assert task_embs.device.type == "meta", \
        f"task_embs.device = {task_embs.device}, expected meta"
    assert proc_embs.device.type == "meta", \
        f"proc_embs.device = {proc_embs.device}, expected meta"


def test_encoder_respects_cuda_device() -> None:
    """V100-only regression — model on cuda, forward emits cuda outputs."""
    if not torch.cuda.is_available():
        raise _Skip("CUDA not available on this host")
    enc = HGATEEncoder(hidden_dim=16, num_layers=1).to("cuda")
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    task_embs, proc_embs = enc(obs)
    assert task_embs.device.type == "cuda"
    assert proc_embs.device.type == "cuda"
    assert torch.isfinite(task_embs).all() and torch.isfinite(proc_embs).all()


# =====================================================================
# Driver
# =====================================================================
def main() -> int:
    print("=" * 72)
    print("HGATE-PPO — per-step acceptance tests")
    print("=" * 72)

    print("\n-- Step 1: HGATEEncoder (__init__ + forward + adapter) --")
    _run("init: task_proj / proc_proj shapes",       test_step1_init_module_shapes)
    _run("init: layer count + norms",                test_step1_layer_count_and_norms)
    _run("forward: output shape + finiteness",       test_step1_forward_output_shape)
    _run("forward: gradient flows to all params",    test_step1_gradient_flow)
    _run("adapter: drops RC edge attr + p2p edges",  test_step1_drops_rc_edge_attribute)

    print("\n-- Step 2: HGATEActorCritic (__init__ + forward) --")
    _run("init: encoder + actor_score + critic attached", test_step2_init_attaches_encoder_actor_critic)
    _run("init: actor_score input dim = 2*hidden (Path B)", test_step2_actor_score_input_dim_is_two_hidden)
    _run("init: critic is single scalar head",            test_step2_critic_is_single_scalar_head)
    _run("forward: (logits, value) shapes + finiteness",  test_step2_forward_logits_and_value_shape)
    _run("forward: gradient flows through full ActorCritic", test_step2_gradient_flow_through_full_actor_critic)
    _run("ActorCritic respects meta device (no cpu leak)", test_step2_actor_critic_respects_meta_device)

    print("\n-- Step 3: HGATEActorCritic.act + get_value --")
    _run("act: returns expected dict keys",            test_step3_act_returns_expected_dict_keys)
    _run("act: mask honored over 1000 samples",        test_step3_act_mask_honored_1000_samples)
    _run("act: deterministic=True is repeatable",      test_step3_act_deterministic_repeatable)
    _run("get_value: returns scalar finite tensor",    test_step3_get_value_returns_scalar_finite)
    _run("get_value agrees with forward()[1]",         test_step3_get_value_agrees_with_forward)
    _run("act + get_value respect meta device",        test_step3_act_get_value_respect_meta_device)

    print("\n-- Step 8: HGATEPPOScheduler (eval-time wrapper) --")
    _run("scheduler loads ckpt + schedules valid action", test_step8_scheduler_loads_ckpt_and_schedules)
    _run("scheduler deterministic=True is repeatable",    test_step8_scheduler_deterministic_repeatable)

    print("\n-- Step 4 (HK-4.5 perf-fix): _ppo_update minibatch + new metrics --")
    _run("ppo_update returns approx_kl + clip_frac + entropy",
          test_step4perf_ppo_update_returns_kl_clipfrac_entropy)
    _run("ppo_update does one optim.step per minibatch",
          test_step4perf_ppo_update_one_optim_step_per_minibatch)

    print("\n-- Step 5 (HK-4.5.2 batched-forward): bit-identical batched API --")
    _run("forward_batched matches sequential forward",
          test_step5perf_forward_batched_matches_sequential)
    _run("get_value_batched matches sequential get_value",
          test_step5perf_get_value_batched_matches_sequential)
    _run("evaluate_actions_batched matches sequential per-action evaluate",
          test_step5perf_evaluate_actions_batched_matches_sequential)
    _run("act_batched deterministic=True matches sequential act",
          test_step5perf_act_batched_deterministic_matches_sequential)
    _run("act_batched stochastic respects per-env masks",
          test_step5perf_act_batched_mask_honored_stochastic)

    print("\n-- Device discipline (HK-3.1.1 carryover) --")
    _run("encoder respects meta device (no cpu leak)", test_encoder_respects_meta_device)
    _run("encoder respects cuda device (V100 path)",   test_encoder_respects_cuda_device)

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
