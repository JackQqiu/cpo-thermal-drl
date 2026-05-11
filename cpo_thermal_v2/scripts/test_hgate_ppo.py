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
