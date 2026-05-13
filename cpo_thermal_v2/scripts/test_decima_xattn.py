#!/usr/bin/env python3
"""
test_decima_xattn.py — unit tests for D2 baseline (HK-5.0)
==========================================================

D2 = Decima encoder (homogeneous GCN) + cross-attention actor + PPO.

Test plan mirrors test_hgate_ppo.py's per-step layout:
  Step 1 — DecimaXAttnEncoder (init / forward / adapter / grad flow)
  Step 2 — DecimaXAttnActorCritic (init / forward shapes / grad flow)
  Step 3 — act / get_value (sampling, mask, determinism)
  Step 4 — _ppo_update minibatch + new metrics (KL, clip_frac, entropy)
  Step 5 — batched API bit-identical to N sequential forwards
  Step 6 — best.pt rolling-mean gate (shared HK-4.6 contract)
  Step 7 — DecimaXAttnScheduler (eval-time wrapper) loads ckpt + acts

Run:
    PYTHONPATH=. python cpo_thermal_v2/scripts/test_decima_xattn.py
"""
from __future__ import annotations

import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cpo_thermal_v2.baselines.decima_xattn import (
    DecimaXAttnEncoder, DecimaXAttnActorCritic, DecimaXAttnScheduler,
)


try:
    import pytest                                              # noqa: F401
    _PYTEST = True
except ImportError:                                           # pragma: no cover
    _PYTEST = False


class _Skip(Exception):
    pass


# =====================================================================
# Fixture: synthetic graph_obs (same schema as cpo_thermal_env)
# =====================================================================
def _make_dummy_obs(
    N_task:    int = 6,
    N_proc:    int = 5,
    task_in:   int = 8,
    proc_in:   int = 7,
    n_t2t:     int = 5,
    seed:      int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    edges_t2t: List[List[int]] = []
    for _ in range(n_t2t):
        u, v = rng.integers(0, N_task, size=2)
        if u != v:
            edges_t2t.append([int(u), int(v)])

    return {
        "proc_x":           rng.standard_normal((N_proc, proc_in)).tolist(),
        "task_x":           rng.standard_normal((N_task, task_in)).tolist(),
        "edges_t2t":        edges_t2t,
        "edges_t2t_attr":   [[0.5] for _ in edges_t2t],
        "edges_p2p":        [],
        "edges_p2p_attr":   [],
        "edges_t2p":        [],
        "edges_t2p_attr":   [],
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
# Step 1 — DecimaXAttnEncoder
# =====================================================================
def test_step1_init_module_shapes() -> None:
    enc = DecimaXAttnEncoder(task_in_dim=8, proc_in_dim=7,
                              hidden_dim=64, num_gcn_layers=3)
    assert isinstance(enc.task_proj, torch.nn.Linear)
    assert enc.task_proj.in_features  == 8
    assert enc.task_proj.out_features == 64
    assert isinstance(enc.proc_proj, torch.nn.Linear)
    assert enc.proc_proj.in_features  == 7
    assert enc.proc_proj.out_features == 64


def test_step1_gcn_stack_count() -> None:
    enc = DecimaXAttnEncoder(hidden_dim=64, num_gcn_layers=4)
    assert hasattr(enc, "gcn_layers")
    assert len(enc.gcn_layers) == 4, \
        f"gcn_layers has len {len(enc.gcn_layers)}, expected 4"


def test_step1_forward_output_shape() -> None:
    """forward(obs) returns (task_embs[T,H], proc_embs[P,H]) — split back."""
    enc = DecimaXAttnEncoder(hidden_dim=32, num_gcn_layers=2)
    obs = _make_dummy_obs(N_task=6, N_proc=5)
    task_embs, proc_embs = enc(obs)
    assert isinstance(task_embs, torch.Tensor)
    assert isinstance(proc_embs, torch.Tensor)
    assert task_embs.shape == (6, 32), \
        f"task_embs.shape = {tuple(task_embs.shape)}, expected (6, 32)"
    assert proc_embs.shape == (5, 32), \
        f"proc_embs.shape = {tuple(proc_embs.shape)}, expected (5, 32)"
    assert torch.isfinite(task_embs).all()
    assert torch.isfinite(proc_embs).all()


def test_step1_gradient_flow() -> None:
    """Every learnable param sees a gradient after backward."""
    enc = DecimaXAttnEncoder(hidden_dim=32, num_gcn_layers=2)
    obs = _make_dummy_obs(N_task=4, N_proc=5)
    task_embs, proc_embs = enc(obs)
    loss = task_embs.sum() + proc_embs.sum()
    loss.backward()
    missing = [n for n, p in enc.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad: {missing}"


def test_step1_drops_thermal_and_task_proc_edges() -> None:
    """D2 encoder contract: must NOT consume edges_p2p (no thermal physics)
    NOR edges_t2p (no task-proc affinity in the GCN trunk; the cross-
    attention actor is what scores task-proc pairs)."""
    obs_no_extras = _make_dummy_obs(N_task=4, N_proc=5)
    obs_with_extras = dict(obs_no_extras)
    obs_with_extras["edges_p2p"]      = [[0, 1], [1, 2]]
    obs_with_extras["edges_p2p_attr"] = [[0.5], [0.7]]
    obs_with_extras["edges_t2p"]      = [[0, 1], [1, 2]]
    obs_with_extras["edges_t2p_attr"] = [[0.5, 0.3], [0.7, 0.4]]

    torch.manual_seed(0); enc1 = DecimaXAttnEncoder(hidden_dim=32, num_gcn_layers=1)
    torch.manual_seed(0); enc2 = DecimaXAttnEncoder(hidden_dim=32, num_gcn_layers=1)
    t1, p1 = enc1(obs_no_extras)
    t2, p2 = enc2(obs_with_extras)
    assert torch.allclose(t1, t2), \
        "task_embs differ — D2 encoder is reading edges_p2p or edges_t2p"
    assert torch.allclose(p1, p2), \
        "proc_embs differ — D2 encoder is reading edges_p2p or edges_t2p"


# =====================================================================
# Step 2 — DecimaXAttnActorCritic
# =====================================================================
def test_step2_init_attaches_encoder_actor_critic() -> None:
    m = DecimaXAttnActorCritic(hidden_dim=64, num_procs=5,
                                num_heads=2, num_gcn_layers=2,
                                task_in_dim=8, proc_in_dim=7)
    assert hasattr(m, "encoder")
    assert isinstance(m.encoder, DecimaXAttnEncoder)
    assert m.encoder.hidden_dim     == 64
    assert m.encoder.num_gcn_layers == 2
    assert hasattr(m, "actor"), "ActorCritic must expose .actor (CrossAttentionActor)"
    from cpo_thermal_v2.models.cross_attention_actor import CrossAttentionActor
    assert isinstance(m.actor, CrossAttentionActor), \
        f"actor type wrong: {type(m.actor).__name__}"
    assert hasattr(m, "critic"), "ActorCritic must expose .critic (single scalar head)"


def test_step2_hidden_num_heads_divisibility() -> None:
    """hidden_dim must be divisible by num_heads (cross-attn head_dim contract)."""
    try:
        DecimaXAttnActorCritic(hidden_dim=64, num_procs=5,
                                num_heads=3, num_gcn_layers=1)
    except ValueError as e:
        assert "divisible" in str(e).lower(), \
            f"expected divisibility ValueError; got: {e}"
        return
    raise AssertionError("hidden_dim=64 num_heads=3 did not raise ValueError")


def test_step2_forward_logits_and_value_shape() -> None:
    m = DecimaXAttnActorCritic(hidden_dim=32, num_procs=5,
                                num_heads=2, num_gcn_layers=1)
    obs  = _make_dummy_obs(N_task=6, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    logits, value = m.forward(obs, mask)
    assert isinstance(logits, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert logits.shape == (5,), \
        f"logits.shape = {tuple(logits.shape)}, expected (5,)"
    assert value.dim() == 0 or value.numel() == 1, \
        f"value must be scalar; got shape {tuple(value.shape)}"
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_step2_gradient_flow_through_full_actor_critic() -> None:
    """Backward through (logits.sum() + value) must populate grads on every
    learnable param — except the delay head, which is auto_only-only and
    intentionally receives no gradient (it's instantiated for code reuse
    with the cross-attention actor but discarded at action time)."""
    m = DecimaXAttnActorCritic(hidden_dim=32, num_procs=5,
                                num_heads=2, num_gcn_layers=2)
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    logits, value = m.forward(obs, mask)
    loss = logits.sum() + value.sum()
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and p.grad is None]
    # The cross-attention actor's delay_mlp is built but not consumed
    # in auto_only — it must be in the missing-grad set, no other param
    # should be.
    delay_params = {n for n in missing if n.startswith("actor.delay_mlp.")}
    other_missing = [n for n in missing if not n.startswith("actor.delay_mlp.")]
    assert not other_missing, (
        f"non-delay params with no grad: {other_missing[:5]}"
        f"{' ...' if len(other_missing) > 5 else ''}")
    # delay_mlp params being grad-less is OK (auto_only contract)


def test_step2_critic_is_single_scalar_head() -> None:
    m = DecimaXAttnActorCritic(hidden_dim=64, num_procs=5,
                                num_heads=2, num_gcn_layers=1)
    linears = [mod for mod in m.critic.modules()
               if isinstance(mod, torch.nn.Linear)]
    assert len(linears) >= 1, "critic has no Linear layers"
    assert linears[0].in_features == 64, \
        f"critic in_features = {linears[0].in_features}, expected 64"
    assert linears[-1].out_features == 1, \
        f"critic out_features = {linears[-1].out_features}, expected 1"


# =====================================================================
# Step 3 — act / get_value
# =====================================================================
def test_step3_act_returns_expected_dict_keys() -> None:
    m = DecimaXAttnActorCritic(hidden_dim=32, num_procs=5,
                                num_heads=2, num_gcn_layers=1)
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    out = m.act(obs, mask, deterministic=False)
    for key in ("action", "log_prob", "entropy", "value"):
        assert key in out, f"act() output missing key {key!r}"
    a = out["action"]
    assert isinstance(a, int), f"action must be int; got {type(a).__name__}"
    for k in ("log_prob", "entropy", "value"):
        t = out[k]
        assert isinstance(t, torch.Tensor)
        assert torch.isfinite(t).all(), f"{k} is non-finite: {t}"


def test_step3_act_mask_honored_1000_samples() -> None:
    torch.manual_seed(42)
    m = DecimaXAttnActorCritic(hidden_dim=32, num_procs=7,
                                num_heads=2, num_gcn_layers=1)
    obs = _make_dummy_obs(N_task=4, N_proc=7)
    mask = np.array([False, True, False, True, True, False, True], dtype=bool)
    valid = set(np.where(mask)[0].tolist())
    for _ in range(1000):
        out = m.act(obs, mask, deterministic=False)
        a_int = int(out["action"])
        assert a_int in valid, \
            f"sampled invalid action {a_int} not in {valid}"


def test_step3_act_deterministic_repeatable() -> None:
    m = DecimaXAttnActorCritic(hidden_dim=32, num_procs=5,
                                num_heads=2, num_gcn_layers=1)
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([False, True, True, False, True], dtype=bool)
    a1 = int(m.act(obs, mask, deterministic=True)["action"])
    a2 = int(m.act(obs, mask, deterministic=True)["action"])
    a3 = int(m.act(obs, mask, deterministic=True)["action"])
    assert a1 == a2 == a3, \
        f"deterministic action varies: {a1}/{a2}/{a3}"


def test_step3_get_value_agrees_with_forward() -> None:
    m = DecimaXAttnActorCritic(hidden_dim=32, num_procs=5,
                                num_heads=2, num_gcn_layers=1)
    m.eval()
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    with torch.no_grad():
        v_from_get   = m.get_value(obs)
        _, v_from_fwd = m.forward(obs, mask)
    diff = (v_from_get.flatten() - v_from_fwd.flatten()).abs().max().item()
    assert diff < 1e-6, f"get_value and forward disagree by {diff:.2e}"


# =====================================================================
# Step 4 — _ppo_update minibatch + new metrics
# =====================================================================
def _make_dummy_rollout(M: int = 8, num_procs: int = 5, device="cpu"):
    from cpo_thermal_v2.training.train_decima_xattn import RolloutBatch

    rng = np.random.default_rng(7)
    graph_obs = [_make_dummy_obs(N_task=4, N_proc=num_procs, seed=i) for i in range(M)]
    masks     = [np.array([True] * num_procs, dtype=bool) for _ in range(M)]
    actions   = rng.integers(0, num_procs, size=M).astype(np.int64)
    old_log_probs = torch.full((M,), -float(np.log(num_procs)),
                                dtype=torch.float32, device=device)
    raw_adv = rng.standard_normal(M).astype(np.float32)
    raw_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
    advantages = torch.tensor(raw_adv, dtype=torch.float32, device=device)
    returns    = torch.tensor(rng.standard_normal(M).astype(np.float32),
                              dtype=torch.float32, device=device)
    return RolloutBatch(
        graph_obs=graph_obs, masks=masks, actions=actions,
        old_log_probs=old_log_probs, advantages=advantages, returns=returns,
    )


def test_step4_ppo_update_returns_kl_clipfrac_entropy() -> None:
    from cpo_thermal_v2.training.train_decima_xattn import _ppo_update

    m = DecimaXAttnActorCritic(hidden_dim=16, num_procs=5,
                                num_heads=2, num_gcn_layers=1)
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
        assert k in metrics, f"metrics missing {k!r}"
        v = metrics[k]
        assert isinstance(v, float), f"metrics[{k!r}] not float: {type(v).__name__}"
        assert np.isfinite(v), f"metrics[{k!r}] is non-finite: {v}"
    assert metrics["approx_kl"] >= -1e-6, \
        f"approx_kl unexpectedly negative: {metrics['approx_kl']}"
    assert 0.0 <= metrics["clip_frac"] <= 1.0, \
        f"clip_frac out of [0,1]: {metrics['clip_frac']}"
    assert metrics["entropy"] > 0.0, \
        f"entropy non-positive: {metrics['entropy']}"


def test_step4_ppo_update_one_optim_step_per_minibatch() -> None:
    from cpo_thermal_v2.training.train_decima_xattn import _ppo_update

    m = DecimaXAttnActorCritic(hidden_dim=16, num_procs=5,
                                num_heads=2, num_gcn_layers=1)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
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
    expected = 3 * (8 // 4)
    assert step_calls[0] == expected, \
        f"optimizer.step called {step_calls[0]} times, expected {expected}"


# =====================================================================
# Step 5 — batched API bit-identical
# =====================================================================
def _make_diverse_obs_list(N: int, num_procs: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    obs_list = []
    for i in range(N):
        T_i = int(rng.integers(3, 9))
        obs_list.append(_make_dummy_obs(N_task=T_i, N_proc=num_procs, seed=seed + i))
    return obs_list


def test_step5_forward_batched_matches_sequential() -> None:
    """forward_batched(N) must produce (logits, values) bit-identical
    (within 1e-5) to N sequential forward() calls."""
    torch.manual_seed(0)
    m = DecimaXAttnActorCritic(hidden_dim=32, num_procs=5,
                                num_heads=2, num_gcn_layers=2)
    m.eval()
    obs_list = _make_diverse_obs_list(N=6, num_procs=5)
    masks = [np.array([True] * 5, dtype=bool) for _ in obs_list]

    seq_logits: List[torch.Tensor] = []
    seq_values: List[torch.Tensor] = []
    with torch.no_grad():
        for go, mk in zip(obs_list, masks):
            lg, vl = m.forward(go, mk)
            seq_logits.append(lg)
            seq_values.append(vl.view(-1))
    seq_logits_t = torch.stack(seq_logits, dim=0)
    seq_values_t = torch.cat (seq_values)

    with torch.no_grad():
        out = m.forward_batched(obs_list, masks)
    bat_logits = out["logits"]
    bat_values = out["values"]
    assert tuple(bat_logits.shape) == (6, 5), \
        f"forward_batched logits shape = {tuple(bat_logits.shape)}, expected (6, 5)"
    assert tuple(bat_values.shape) == (6,), \
        f"forward_batched values shape = {tuple(bat_values.shape)}, expected (6,)"

    diff_logits = (seq_logits_t - bat_logits).abs().max().item()
    diff_values = (seq_values_t - bat_values).abs().max().item()
    assert diff_logits < 1e-4, \
        f"forward_batched logits drift: {diff_logits:.2e}"
    assert diff_values < 1e-4, \
        f"forward_batched values drift: {diff_values:.2e}"


def test_step5_get_value_batched_matches_sequential() -> None:
    torch.manual_seed(1)
    m = DecimaXAttnActorCritic(hidden_dim=24, num_procs=5,
                                num_heads=2, num_gcn_layers=2)
    m.eval()
    obs_list = _make_diverse_obs_list(N=5, num_procs=5)

    with torch.no_grad():
        seq = torch.stack([m.get_value(go).view(-1) for go in obs_list]).view(-1)
    with torch.no_grad():
        bat = m.get_value_batched(obs_list)
    assert tuple(bat.shape) == (5,)
    diff = (seq - bat).abs().max().item()
    assert diff < 1e-4, f"get_value_batched drift: {diff:.2e}"


def test_step5_evaluate_actions_batched_matches_sequential() -> None:
    """evaluate_actions_batched (used inside the PPO minibatch update)
    must yield (new_log_prob, entropy, value) bit-identical to B
    sequential forward + mask + log_softmax + gather calls."""
    torch.manual_seed(2)
    m = DecimaXAttnActorCritic(hidden_dim=32, num_procs=5,
                                num_heads=2, num_gcn_layers=2)
    m.eval()
    B = 7
    obs_list = _make_diverse_obs_list(N=B, num_procs=5)
    masks = [np.array([True, True, False, True, True], dtype=bool)
              for _ in obs_list]
    actions = np.array([1, 0, 3, 4, 1, 0, 3], dtype=np.int64)

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
            seq_v.append(vl.view(()))
    seq_lp_t  = torch.stack(seq_lp )
    seq_ent_t = torch.stack(seq_ent)
    seq_v_t   = torch.stack(seq_v  )

    with torch.no_grad():
        bat_lp, bat_ent, bat_v = m.evaluate_actions_batched(
            obs_list, masks, actions)
    for name, seq_t, bat_t in [
        ("new_log_prob", seq_lp_t,  bat_lp ),
        ("entropy",      seq_ent_t, bat_ent),
        ("value",        seq_v_t,   bat_v  ),
    ]:
        diff = (seq_t - bat_t).abs().max().item()
        assert diff < 1e-4, f"evaluate_actions_batched {name} drift: {diff:.2e}"


def test_step5_act_batched_deterministic_matches_sequential() -> None:
    torch.manual_seed(3)
    m = DecimaXAttnActorCritic(hidden_dim=24, num_procs=7,
                                num_heads=2, num_gcn_layers=1)
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

    with torch.no_grad():
        bat = m.act_batched(obs_list, masks, deterministic=True)
    assert torch.equal(bat["actions"], seq_act_t), \
        f"deterministic actions differ: seq={seq_act_t.tolist()} bat={bat['actions'].tolist()}"


def test_step5_act_batched_mask_honored_stochastic() -> None:
    torch.manual_seed(4)
    m = DecimaXAttnActorCritic(hidden_dim=16, num_procs=5,
                                num_heads=2, num_gcn_layers=1)
    m.eval()
    N = 8
    obs_list = _make_diverse_obs_list(N=N, num_procs=5)
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
                assert int(a) in valid_sets[n], \
                    f"env {n} sampled invalid action {a}"


# =====================================================================
# Step 6 — best.pt rolling-mean gate (HK-4.6 contract reuse)
# =====================================================================
def test_step6_best_ckpt_rolling_mean_gate_first_save() -> None:
    from cpo_thermal_v2.training.train_decima_xattn import _update_best_ckpt
    for n_eps in range(1, 50):
        new_best, should_save = _update_best_ckpt(
            rolling_returns=[10.0] * n_eps,
            best_so_far=-float("inf"), window=50,
        )
        assert not should_save, \
            f"early save fired at n_eps={n_eps}"
        assert new_best is None
    new_best, should_save = _update_best_ckpt(
        rolling_returns=[10.0] * 50,
        best_so_far=-float("inf"), window=50,
    )
    assert should_save
    assert new_best == 10.0


def test_step6_best_ckpt_window_smaller_than_returns() -> None:
    from cpo_thermal_v2.training.train_decima_xattn import _update_best_ckpt
    bad = [-1000.0] * 950
    good = [1000.0] * 50
    new_best, should_save = _update_best_ckpt(
        rolling_returns=bad + good, best_so_far=-float("inf"), window=50)
    assert should_save
    assert 900 < new_best < 1100, \
        f"new_best={new_best:.1f}, expected ~1000"


# =====================================================================
# Step 7 — DecimaXAttnScheduler (eval-time wrapper)
# =====================================================================
def test_step7_scheduler_loads_ckpt_and_schedules() -> None:
    import tempfile, os
    torch.manual_seed(0)
    inner = DecimaXAttnActorCritic(hidden_dim=16, num_procs=5,
                                    num_heads=2, num_gcn_layers=1,
                                    task_in_dim=8, proc_in_dim=7)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "tiny.pt")
        torch.save({"model": inner.state_dict(),
                    "global_step": 0,
                    "metrics_summary": {"ep_ret_mean": 0.0}}, p)

        sch = DecimaXAttnScheduler(
            ckpt_path=p, num_nodes=5, deterministic=True, device="cpu",
            hidden_dim=16, num_gcn_layers=1, num_heads=2,
        )
        assert sch.name == "D2"
        assert sch.action_mode == "auto_only"

        obs = _make_dummy_obs(N_task=4, N_proc=5)
        info = {
            "graph_obs":   [obs],
            "action_mask": np.array([True, False, True, True, False], dtype=bool),
        }
        action = sch.schedule(None, info)
        assert isinstance(action, int), \
            f"schedule must return int (auto_only); got {type(action).__name__}"
        assert 0 <= action < 5, f"action {action} out of range"
        assert info["action_mask"][action], \
            f"schedule returned masked-out action {action}"


def test_step7_scheduler_deterministic_repeatable() -> None:
    import tempfile, os
    torch.manual_seed(1)
    inner = DecimaXAttnActorCritic(hidden_dim=16, num_procs=5,
                                    num_heads=2, num_gcn_layers=1)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "tiny.pt")
        torch.save({"model": inner.state_dict()}, p)

        sch = DecimaXAttnScheduler(
            ckpt_path=p, num_nodes=5, deterministic=True, device="cpu",
            hidden_dim=16, num_gcn_layers=1, num_heads=2,
        )
        obs = _make_dummy_obs(N_task=4, N_proc=5)
        info = {"graph_obs": [obs],
                "action_mask": np.array([False, True, True, True, True], dtype=bool)}
        a1 = sch.schedule(None, info)
        a2 = sch.schedule(None, info)
        a3 = sch.schedule(None, info)
        assert a1 == a2 == a3, \
            f"deterministic scheduler varies: {a1}/{a2}/{a3}"


# =====================================================================
# Device discipline (HK-3.1.1 carryover)
# =====================================================================
_PRE_FIX_ERROR = "not on the expected device cpu"


def test_encoder_respects_meta_device() -> None:
    enc = DecimaXAttnEncoder(hidden_dim=16, num_gcn_layers=1).to("meta")
    assert next(enc.parameters()).device.type == "meta"
    obs = _make_dummy_obs(N_task=4, N_proc=5)
    try:
        task_embs, proc_embs = enc(obs)
    except RuntimeError as e:
        msg = str(e)
        assert _PRE_FIX_ERROR not in msg, \
            f"encoder has CPU-tensor leak:\n  {msg}"
        return
    assert task_embs.device.type == "meta"
    assert proc_embs.device.type == "meta"


def test_actor_critic_respects_meta_device() -> None:
    m = DecimaXAttnActorCritic(hidden_dim=16, num_procs=5,
                                num_heads=2, num_gcn_layers=1).to("meta")
    obs  = _make_dummy_obs(N_task=4, N_proc=5)
    mask = np.array([True] * 5, dtype=bool)
    try:
        logits, value = m.forward(obs, mask)
    except RuntimeError as e:
        assert _PRE_FIX_ERROR not in str(e), \
            f"actor-critic has CPU-tensor leak:\n  {e}"
        return
    assert logits.device.type == "meta"
    assert value.device.type == "meta"


# =====================================================================
# Driver
# =====================================================================
def main() -> int:
    print("=" * 72)
    print("D2 (Decima encoder + cross-attention actor) — unit tests")
    print("=" * 72)

    print("\n-- Step 1: DecimaXAttnEncoder --")
    _run("init: task_proj / proc_proj shapes",            test_step1_init_module_shapes)
    _run("init: gcn_layers count",                        test_step1_gcn_stack_count)
    _run("forward: (task_embs, proc_embs) shape + finite", test_step1_forward_output_shape)
    _run("forward: gradient flows through all params",    test_step1_gradient_flow)
    _run("adapter: drops thermal + task-proc edges",      test_step1_drops_thermal_and_task_proc_edges)

    print("\n-- Step 2: DecimaXAttnActorCritic --")
    _run("init: encoder + actor + critic attached",       test_step2_init_attaches_encoder_actor_critic)
    _run("init: hidden % num_heads divisibility check",   test_step2_hidden_num_heads_divisibility)
    _run("forward: (logits, value) shapes + finite",      test_step2_forward_logits_and_value_shape)
    _run("forward: gradient flows through actor-critic",  test_step2_gradient_flow_through_full_actor_critic)
    _run("init: critic is single scalar head",            test_step2_critic_is_single_scalar_head)

    print("\n-- Step 3: act + get_value --")
    _run("act: dict keys",                                test_step3_act_returns_expected_dict_keys)
    _run("act: mask honored across 1000 samples",         test_step3_act_mask_honored_1000_samples)
    _run("act: deterministic is repeatable",              test_step3_act_deterministic_repeatable)
    _run("get_value agrees with forward()[1]",            test_step3_get_value_agrees_with_forward)

    print("\n-- Step 4: _ppo_update minibatch + new metrics --")
    _run("ppo_update returns approx_kl + clip_frac + entropy",
          test_step4_ppo_update_returns_kl_clipfrac_entropy)
    _run("ppo_update does one optim.step per minibatch",
          test_step4_ppo_update_one_optim_step_per_minibatch)

    print("\n-- Step 5: batched API bit-identical --")
    _run("forward_batched matches sequential",            test_step5_forward_batched_matches_sequential)
    _run("get_value_batched matches sequential",          test_step5_get_value_batched_matches_sequential)
    _run("evaluate_actions_batched matches sequential",   test_step5_evaluate_actions_batched_matches_sequential)
    _run("act_batched deterministic matches sequential",  test_step5_act_batched_deterministic_matches_sequential)
    _run("act_batched stochastic respects per-env masks", test_step5_act_batched_mask_honored_stochastic)

    print("\n-- Step 6: best.pt rolling-mean gate --")
    _run("first save fires at window=50",                 test_step6_best_ckpt_rolling_mean_gate_first_save)
    _run("window honoured with long history",             test_step6_best_ckpt_window_smaller_than_returns)

    print("\n-- Step 7: DecimaXAttnScheduler (eval-time wrapper) --")
    _run("scheduler loads ckpt + schedules valid action", test_step7_scheduler_loads_ckpt_and_schedules)
    _run("scheduler deterministic is repeatable",         test_step7_scheduler_deterministic_repeatable)

    print("\n-- Device discipline (HK-3.1.1 carryover) --")
    _run("encoder respects meta device",                  test_encoder_respects_meta_device)
    _run("actor-critic respects meta device",             test_actor_critic_respects_meta_device)

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
