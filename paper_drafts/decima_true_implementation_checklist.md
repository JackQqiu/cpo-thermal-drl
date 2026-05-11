# Decima TRUE Implementation Checklist — Mao 2019 SIGCOMM

**Status:** BLOCKED — scaffold only.
**Effort estimate:** 5-7 days, single developer (per CLAUDE.md §3 Stage 2).
**Files:**
- `cpo_thermal_v2/baselines/decima_true.py` (stub: NotImplementedError throughout — shared by both variants)
- `cpo_thermal_v2/configs/train_decima_true_vanilla.yaml` (stub: `_BLOCKED_DO_NOT_RUN:` guard)
- `cpo_thermal_v2/configs/train_decima_true_thermal.yaml` (stub: `_BLOCKED_DO_NOT_RUN:` guard)
- `cpo_thermal_v2/scripts/train_decima_true_vanilla.sbatch` (refuses to launch while guard is present)
- `cpo_thermal_v2/scripts/train_decima_true_thermal.sbatch` (refuses to launch while guard is present)
- `paper_drafts/decima_true_implementation_checklist.md` (this file — draft, untracked)

**Reference:** Mao et al., "Learning Scheduling Algorithms for Data Processing Clusters", SIGCOMM 2019.

---

## Two-variant strategy

The reward-design question is resolved by training **both** variants
through the same codebase + checklist, differing only in yaml config:

| # | Variant | Yaml | Reward | What it isolates in §5 cube |
|---|---|---|---|---|
| 1 | Decima vanilla | `train_decima_true_vanilla.yaml` | makespan-only (Mao 2019 faithful) | Avoids cherry-pick critique; pairs with thermal to isolate reward |
| 2 | Decima thermal | `train_decima_true_thermal.yaml` | thermal-aware (SAME as Ours) | Controlled comparison vs decima_fair (algo+GNN) and Ours (RC edge) |

§5 ablation chain (paper):

```
Decima vanilla  ── isolates reward signal ──>  Decima thermal
                                                      │
                                                      │ isolates algo + GNN type
                                                      v
                                              decima_fair (HK-2.2 PPO + Hetero)
                                                      │
                                                      │ isolates RC edge attribute
                                                      v
                                                    Ours
```

**Train order:** vanilla first. If vanilla doesn't converge at all, the
thermal variant won't either (same model class, same algo, only reward
differs) — saves 18h on a dead end.

**Total V100 budget:** ~36h (18h × 2).

The model class in `decima_true.py` is identical across variants.

---

## 10-step implementation plan

### Step 1 — Homogeneous GCN encoder
- [ ] Replace `DecimaTruePolicy.__init__` body
- [ ] Stack `torch_geometric.nn.GCNConv` × `num_gcn_layers` (default 4)
- [ ] Mean aggregation (NOT GAT attention)
- [ ] Single node type: concat `[task_features, proc_features]` + one-hot type indicator
- [ ] Edges: DAG precedence + self-loop ONLY; **NO** task-proc affinity, **NO** thermal coupling

**Acceptance:** `model.forward(dummy_obs)` returns logits without error; `model(obs).shape` matches expected `(num_dags + num_pairs,)`.

### Step 2 — Two-stage Mao score function
- [ ] Stage A: scalar score per ready DAG → softmax → DAG selection distribution
- [ ] Stage B: scalar score per (ready_task, processor) pair within chosen DAG → softmax → pair distribution
- [ ] Use separate linear heads on top of the GCN final-layer node embeddings
- [ ] Hierarchical sampling: sample DAG first, then condition Stage B on chosen DAG's task subset

**Acceptance:** `policy.forward(obs)` returns `(stage_A_logits, stage_B_logits)`; both finite, no NaN; sampled action is a valid `(task_idx, proc_idx)` pair under `info["action_mask"]`.

### Step 3 — Action sampling + masking
- [ ] Implement `select_action(state, deterministic)` with proper masking (mask invalid procs / non-ready tasks before softmax)
- [ ] Wrap output as auto_only `int` action via `BaseScheduler._wrap_action`

**Acceptance:** 1000 sampled actions all pass `info["action_mask"]`; `deterministic=True` gives argmax; `deterministic=False` gives variance.

### Step 4 — REINFORCE agent setup
- [ ] Replace `DecimaTrueAgent.__init__` body
- [ ] Optimizer: Adam or RMSProp (Mao used RMSProp lr=1e-3; we start with Adam lr=5e-4)
- [ ] Moving-average baseline buffer (`baseline_window=50` episodes, ring buffer of returns)
- [ ] No GAE, no clipping, no minibatching, no entropy bonus by default (consider adding if collapse, see Step 9)

**Acceptance:** `agent.update(dummy_rollout)` returns `loss` finite and >0 (cross-entropy > 0); baseline tracker updates correctly across 100 dummy episodes.

### Step 5 — Training loop integration
- [ ] Add `algorithm: reinforce` dispatch branch in `cpo_thermal_v2/training/train.py` (currently PPO-only)
- [ ] Add `architecture: decima_true` registry entry in `cpo_thermal_v2/models/__init__.py`
- [ ] Single gradient step per episode (NOT per minibatch) — REINFORCE does not minibatch
- [ ] Loss: `loss = -mean(log_prob_t * (G_t - baseline))` where `G_t` is discounted return; optionally normalize `(G_t - baseline)` by std for numerical stability (note in paper if used)

**Acceptance:** `python -m cpo_thermal_v2.training.train --config configs/train_decima_true_vanilla.yaml --total-steps 1000` runs without error; tensorboard shows `train/loss`, `train/baseline`, `train/ep_return` keys. Repeat for `train_decima_true_thermal.yaml`.

### Step 6 — Reward + observation stripping (env-side)
**This step is required for both variants.**  Currently `reward_mode`
is **not** present anywhere in `envs/` or `training/` (verified: zero
grep matches as of scaffold time), so the dispatch must be added from
scratch.

- [ ] Implement env-side `reward_mode` dispatch in `envs/reward_shaping.py`
  supporting two values:
  - `'makespan_only'` — Mao 2019 original; only base_step_reward,
    truncate_pen (still needed for termination semantics), hard_wall_pen,
    dag_done_bonus.  Drop all `cool_*`, `anticipation_*`,
    `under_anticipate_*`, and soft_wall terms.
  - `'thermal_aware'` — full Ours reward (eq 14); pass-through to the
    existing channel-routed reward path.
- [ ] Implement env-side `observation_keys` filter in `envs/cpo_thermal_env.py`
  — drop everything except listed keys before returning obs.  Same filter
  applies to both variants (both strip thermal obs; only reward differs).
- [ ] Verify `truncate_pen` and `hard_wall_pen` still fire under both
  modes (they're termination semantics, not "thermal" per se, but the
  episode-end behavior is critical for return computation).

**Acceptance (vanilla):** `env(reward_mode='makespan_only').step(...)`
reward stream contains no `cool_*`, no `anticipation_*`, no
`under_anticipate_*`, no soft-wall terms; truncate / hard-wall penalties
still fire on thermal trips; obs dict has only `task_x` + `edges_t2t`.

**Acceptance (thermal):** `env(reward_mode='thermal_aware').step(...)`
reward equals the existing Ours reward stream bit-for-bit (regression
test against an unchanged Ours rollout); obs dict has only
`task_x` + `edges_t2t`.

### Step 7 — Eval-time scheduler (both variants share the same code)
- [ ] Replace `DecimaTrueScheduler.__init__` body — load checkpoint, instantiate `DecimaTruePolicy`, set `eval()`
- [ ] Implement `schedule()` — convert `info["graph_obs"]` to homogeneous graph format, forward, hierarchical argmax, wrap as auto_only `int`
- [ ] Register in `evaluation/runner.py` so `--scheduler decima_true_vanilla` and `--scheduler decima_true_thermal` both work (or pass `--ckpt-path` to disambiguate)

**Acceptance:** loading a smoke-test checkpoint from EITHER variant and running 5 episodes from `evaluate.py` completes without error and produces an `episodes.csv` with the expected columns; the two ckpts produce visibly different schedules (sanity that reward really did affect training).

### Step 8 — Smoke test (100 steps local) — RUN BOTH VARIANTS
- [ ] Run vanilla with `total_steps=100` on CPU; verify `ep_return` rises
- [ ] Run thermal with `total_steps=100` on CPU; verify `ep_return` rises
- [ ] Verify baseline tracker non-zero after first episode for each

**Acceptance (per variant):** `ep_return(50)` final mean > `ep_return(50)` initial mean by some non-trivial margin (e.g., > 5% relative improvement). If flat or decreasing → bug in loss / sign / log_prob (same code for both variants, so a bug here likely affects both).

### Step 9 — 1M-step pilot — VANILLA FIRST (V100, ~3-4h)
- [ ] Submit `train_decima_true_vanilla.sbatch` after removing `_BLOCKED_DO_NOT_RUN:` from the vanilla yaml
- [ ] Watch for divergence: `loss → ±inf` or `ep_return → -inf` are bug signals
- [ ] Watch for collapse: `H(stage_A)` or `H(stage_B)` falling below 0.3 prematurely indicates need for entropy regularization
- [ ] If unstable: add entropy bonus (`+ beta * H(policy)`, beta ~ 0.01); document in paper §4.x as one deviation from Mao

**Acceptance:** `ep_return` rises monotonically (or saturates without diverging) over 1M steps; final return ≥ random-policy baseline by a clear margin.

**Decision gate after vanilla pilot:**
- Converged: proceed to Step 10 for vanilla AND submit `train_decima_true_thermal.sbatch` pilot in parallel
- Failed: do NOT submit thermal — same model + algo will fail too. Report negative result per HANDOFF §7 ("REINFORCE struggles in this regime, motivating PPO"). Do NOT silently swap to Decima++ (Liu 2022) without footnoting the deviation.

### Step 10 — Full 5M-step training + checkpoint eval — BOTH VARIANTS
- [ ] Resubmit `train_decima_true_vanilla.sbatch` with full `total_steps: 5_000_000`
- [ ] Resubmit `train_decima_true_thermal.sbatch` with full `total_steps: 5_000_000`
- [ ] Save `best.pt` by validation `ep_return` separately for each variant (different `checkpoint_dir`)
- [ ] Run full eval matrix per CLAUDE.md §3 Stage 5 (4 ambient × 5 N × 500 ep) for each ckpt; both rows appear in §5 main results table

**Acceptance:** `best.pt` saved for both variants; both produce `episodes.csv` with N=17 main scaling + horizon scan rows; results integrate into `compare_*.py` paired analysis without column-name surprises; `Decima vanilla` and `Decima thermal` show as separate rows.

---

## Risks (from CLAUDE.md §7)

1. **REINFORCE notoriously unstable in thermal env.** Truncated episodes give unbounded return variance; baseline_window=50 may be too small. Mitigations: entropy regularization (Step 9), normalize advantages, increase window.
2. **Homogeneous GCN cannot represent RC structure** — this is the **point** of the baseline. Ours-NoThermal vs Decima true is one rung in the §5.3 ablation chain ("CPO-specific RC edge value"). Don't accidentally fix this by adding hetero typing.
3. **Mao reward in thermal-truncated episodes may dominate signal.** If `truncate_pen=20` fires often during early training, the policy may learn to avoid action altogether (degenerate solutions). Monitor truncation rate during pilot.
4. **HGATE-PPO may train to ≥ Ours-NoThermal** (CLAUDE.md §7 item 4). If this happens, RC edge attribute's contribution is in question — surface it, don't cover it up.

---

## Time budget estimate

| Step | Days |
|---|---|
| 1-3 (model architecture) | 2 |
| 4-5 (REINFORCE + train.py wiring) | 1.5 |
| 6 (env reward_mode dispatch + obs filter — BOTH branches) | 0.75 |
| 7 (eval-time scheduler — works for both ckpts) | 0.5 |
| 8-9 (smoke both + vanilla pilot + tuning) | 1-2 |
| 10 (full training × 2 + eval × 2) | wallclock 36h, integration < 0.5 |
| **Total** | **~5-7 days dev + 36h V100 training** |
