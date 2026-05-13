# Findings — Paper §5 Evaluation Plan Context

## What's Already Validated

### HK-4.7 5-way HOT eval (50 ep paired) — DONE 2026-05-12
- **CSV**: `eval_results/hgate_final_5way_hot/episodes.csv` (250 rows = 5 schedulers × 50 ep)
- **Schedulers**: HEFT, Decima-vanilla, Decima-thermal, HGATE-PPO (from `checkpoints/hgate_ppo_N17/best.pt`), Ours-auto_only (from `checkpoints/stage1_auto_only_N17/best.pt`)
- **Regime**: HOT [60, 75] °C, N=17, seed_base=100000
- **Key results**:
  - HEFT: peak_T 90.67°C, viol_rate 0.960, completion 0.320, ep_ret +80.71
  - Decima-vanilla: peak_T 90.68°C, viol_rate 1.000, completion 0.325, ep_ret +90.08
  - Decima-thermal: peak_T 80.21°C, viol_rate 0.480, completion 0.686, ep_ret +224.91
  - HGATE-PPO: peak_T 88.94°C, viol_rate 0.980, completion 0.414, ep_ret +117.96
  - Ours-auto_only: peak_T 67.50°C, viol_rate 0.000, completion 1.000, ep_ret +341.66
- **Reusable for Phase A**: yes (5/6 schedulers; just append D2 row when ckpt lands)

### Stage 2 Hybrid Case Study — DONE
- **CSVs**: under `eval_results/stage2_v{2,3,3final}_validation_{warm,hot}/episodes.csv`
- **Schedulers**: Ours-hybrid-v1, v2, v3-best, v3-final, paired vs Ours-auto_only-s1
- **Regime**: WARM [50, 65] + HOT [60, 75], N=17 (v1/v2) or N=33 (v3), 50 ep paired
- **Key finding**: all 4 hybrid variants failed to Pareto-dominate Stage-1 placement-only
- **Paper §5.X snippet ready**: `paper_drafts/section5X_hybrid_case_study.tex` (committed on `paper-draft` as HK-paper-2 + HK-paper-3a)

### Stage 1 validation
- **CSV**: `eval_results/stage1_validation_*/episodes.csv`
- **Ours-auto_only-s1**: the Pareto anchor referenced everywhere

## Existing Checkpoints (Local Disk)

| Scheduler | Checkpoint Path | Status |
|---|---|---|
| Ours-auto_only (Stage 1) | `checkpoints/stage1_auto_only_N17/best.pt` | Trained, validated |
| Ours-hybrid v1 | `checkpoints/stage2_hybrid_N17/best.pt` | Trained, case study |
| Ours-hybrid v2 | `checkpoints/stage2_hybrid_v2_N17/best.pt` | Trained, case study |
| Ours-hybrid v3 | `checkpoints/stage2_hybrid_v3_stress_N33/{best,final}.pt` | Trained, case study |
| Decima-vanilla | `checkpoints/decima_true_vanilla_N17/best.pt` | Trained |
| Decima-thermal | `checkpoints/decima_true_thermal_N17/best.pt` | Trained |
| HGATE-PPO | `checkpoints/hgate_ppo_N17/best.pt` | Trained (H collapsed early; ep_ret_mean +136.80 @ step 1.12M) |
| **D2** (Decima encoder + cross-attention) | `checkpoints/decima_xattn_N17/best.pt` | **PENDING** — V100 24h after sbatch |

## Round-1 Reviewer Obligations

- **Throttled-HEFT** as like-for-like classical baseline — code shipped (`cpo_thermal_v2/baselines/throttled_heft.py` for hybrid + agent_only modes), eval missing from §5
- **Decima true reproduction** (Mao 2019 SIGCOMM, homogeneous GCN + REINFORCE) — done, ckpts above
- **HGATE-PPO 2025 reproduction** (Wu 2025 IoT-J, hetero GAT + PPO) — done, ckpt above; training H-collapsed but reported honestly

## Audit (HK-paper-3a) Findings — Status

### Gate-blockers PRE-Phase-2 (closed this round)
- ✓ Abstract 3-tier vs §1 4-link contradiction — fixed in HK-paper-3a abstract realignment
- ✓ Abstract dual-critic overclaim — fixed in HK-paper-3a (half-sentence deleted)
- ✓ §1 9786/10000 priming for HEFT failure rate — fixed in HK-paper-3a
- ✓ §6.3 dual-critic Option B reframe (HK-paper-2) — done
- ✓ §5.X echo-chamber line removed (HK-paper-2) — done

### Gate-blockers DEFERRED to Phase G (after eval data lands)
- §1 four-link ablation promise vs §5 absence → resolved by section5_main_results.tex plug-in
- Statistical test type (Fisher exact → paired Wilcoxon + Holm-Bonferroni)
- Training hyper-params incomplete in §4 (rollout/batch/optimizer/gamma/5 reward coefs)
- Ours-NoThermal training budget 2.5× asymmetric — methodology note in §4 or footnote
- Throttled-HEFT absent from §2 + §5 — §5 added via plug-in; §2 mention as part of §5 plug-in commit
- Closest-prior repositioning — surfaces automatically via §5.1 Component 5 narrative
- §5.4 "emergent autonomy" tone vs §6.3 — user opted to keep §5.4 unchanged, §5.X carries the negative tone
- Counter-evidence weighting — user opted not to touch §1/§2 narrative

### False positives ruled out
- §1 LaTeX rendering failures (5 items) — markdown-strip artefact, source is fine
- Orphan `fig:cpo-arch` — defined at line 481

## Stale §5 Data Inconsistency

**Critical context (user-confirmed 2026-05-13)**: The existing §5 in `draft/draft.tex` is from a previous experimental run that does NOT match current code. Numbers there (9786/10000 HEFT, 408×/4.8× decomposition, emergent autonomy at 372 ms cooling, etc.) are STALE. Phase G must wholesale replace §5 with `section5_main_results.tex` + `section5X_hybrid_case_study.tex` Phase 2 plug-in, populated from the Phase A-F eval CSVs.

**Implication for §1 9786 priming added in HK-paper-3a**: HK-4.7 current-code data shows HEFT trunc_rate = 0.960, i.e., ~9600/10000 not 9786. Phase G must reconcile (either re-run a 10k-ep HEFT eval to get the actual current number, or change §1 to "≈9600/10000" or similar).

## Repository Structure Notes

- **Branches**: `main` (code + operational planning), `paper-draft` (paper-private LaTeX edits)
- **Paper files on `paper-draft` only**: `draft/draft.tex`, `paper_drafts/section5*.tex`, `paper_drafts/deletion_plan.md`, `paper_drafts/headline_numbers_to_sync.txt`
- **This eval plan lives on `main`** — operational, no paper prose
- **Switch protocol**: when editing paper, `git checkout paper-draft`; when editing code/eval, `git checkout main`

## Available Tools / Resources

- `cpo_thermal_v2/evaluation/evaluate.py` — main eval entrypoint (already supports D2 via HK-5.0)
- `cpo_thermal_v2/evaluation/runner.py` — episode runner
- `cpo_thermal_v2/evaluation/plots.py` — has D2 palette as of HK-5.0
- `cpo_thermal_v2/baselines/{decima_true,hgate_ppo,decima_xattn,throttled_heft}.py` — all baseline scheduler implementations
- Skill: `paper-audit` (review_results/ artifacts from HK-paper-3a session)
- Skill: `latex-paper-en` for Phase G compile/citation/format checks
- Skill: `planning-with-files` (this skill, governs this plan)
