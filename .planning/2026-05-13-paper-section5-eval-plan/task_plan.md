# Task Plan: Paper §5 Evaluation Plan (Post-D2)

## Goal

Produce the complete evaluation evidence base that backs the paper §5
wholesale rewrite (replacing the stale existing §5 in draft.tex with
the Phase-2 plug-in snippet `paper_drafts/section5_main_results.tex` +
`paper_drafts/section5X_hybrid_case_study.tex`). Specifically: a
6-row main table at N=17 HOT, a scaling sweep over N, an ambient
sweep, Throttled-HEFT baseline coverage (round-1 reviewer
obligation), and proper paired-Wilcoxon + Holm-Bonferroni statistics
at higher episode count.

## Current Phase

**Phase X-A — D2 sanity 6-way smoke** (Phase 0 complete; D2 ckpt landed)

## Phases

### Phase 0: D2 V100 retrain ✅ COMPLETE (2026-05-13)
- [x] User sbatched `cpo_thermal_v2/scripts/train_decima_xattn.sbatch`
- [x] ~17h wallclock; D2 finished 5M steps
- [x] best.pt at step 4,628,480, ep_ret_mean=+832.79 (hot-stage policy; HGATE comparison: HGATE's best.pt was +136.80 — D2 6× better)
- [x] final.pt at step 5,000,000
- [x] Synthetic sanity check via DecimaXAttnScheduler load + schedule() returns valid action
- **Status:** complete

### Phase X-A: D2 sanity 6-way smoke (gate to grand matrix)
- [ ] Write `cpo_thermal_v2/configs/eval_main_6way_hot_n17.yaml`
- [ ] Run 6-scheduler eval × N=17 × HOT × 50 ep
- [ ] Verify D2 row CSV columns + no NaN/error rows
- [ ] Measure Mac CPU per-ep timing (calibrates Phase X-B/C wallclock)
- **Cells:** {HEFT, Decima-vanilla, Decima-thermal, HGATE-PPO, **D2**, Ours-auto_only} × N=17 × HOT × 50 ep = 300 ep
- **Wallclock target:** ~10 min Mac CPU
- **Deliverable:** `eval_results/main_6way_hot_n17/episodes.csv` + Mac CPU rate estimate
- **Status:** in_progress

### Phase X-B: Grand scaling + ambient matrix (CLAUDE.md Stage 5 original scope)
- [ ] Write `cpo_thermal_v2/configs/eval_grand_matrix.yaml`
- [ ] Run 5 N × 4 ambient × 500 ep × 10 schedulers (Throttled-HEFT × 2 modes counts as 2 rows)
- **N values:** {9, 13, 17, 24, 33}
- **Ambient ranges:** cold [25, 40], warm [40, 65], hot [60, 75], extreme [70, 80]
- **Schedulers (10 rows):** HEFT, Thermal-HEFT, RoundRobin, Throttled-HEFT-hybrid, Throttled-HEFT-agent_only, Decima-vanilla, Decima-thermal, HGATE-PPO, **D2**, Ours-auto_only
- **Total cells:** 5 × 4 × 10 = 200 cells × 500 ep = 100,000 ep
- **Wallclock estimate (Mac CPU):** ~21h at HK-4.7-pace (~0.7 sec/(sched,ep)) — but Mac CPU may be slower than Linux; calibrate from Phase X-A
- **Alternative:** user sbatches to V100/A100 partition for ~7-10× speedup (~2-3h elapsed)
- **Deliverable:** `eval_results/grand_matrix/episodes.csv`
- **Status:** pending — blocks on X-A timing measurement + user partition choice

### Phase X-C: Horizon scan (paper §5.4 horizon-sensitivity figure)
- [ ] Write `cpo_thermal_v2/configs/eval_horizon_scan_v3.yaml`
- [ ] Run 4 H × 500 ep × N=17 × 10 schedulers
- **Horizon values (dags_per_episode):** {20, 50, 100, 200}
- **Same 10 schedulers, N=17, HOT only**
- **Total cells:** 4 × 10 = 40 cells × 500 ep = 20,000 ep
- **Wallclock estimate:** ~4h Mac CPU (extrapolated from X-A)
- **Deliverable:** `eval_results/horizon_scan_v3/episodes.csv`
- **Status:** pending — blocks on X-A

### Phase X-D: Throttled-HEFT plug-in (round-1 reviewer obligation)
- **Folded into Phase X-B/X-C** as additional 2 scheduler rows (hybrid + agent_only mode)
- **Status:** subsumed by X-B/X-C

### Phase G: Aggregate + paper §5 plug-in (after A-F done)
- [ ] Run `cpo_thermal_v2/scripts/compose_paper_section5.py` (new) — joins A-F CSVs, computes Wilcoxon+Holm, emits booktabs/pgfplots data for paper §5
- [ ] Update `paper_drafts/section5_main_results.tex` D2 row + Holm-adjusted p-values
- [ ] Switch to `paper-draft` branch; plug-in `section5_main_results.tex` and `section5X_hybrid_case_study.tex` into `draft/draft.tex` (wholesale §5 replace)
- [ ] Add §4 training hyper-params block (deferred audit item from HK-paper-3a)
- [ ] Compile 3-pass, verify clean (0 undef refs, ?? count ≤ 0 after plug-in)
- **Wallclock:** ~1 h
- **Deliverable:** updated `draft/draft.tex` on `paper-draft` branch
- **Status:** pending — blocks on Phase A

### Phase H: Post-merge audit re-run (verify gate-blockers cleared)
- [ ] Re-run paper-audit deep-review on the post-Phase-G `draft/draft.tex`
- [ ] Confirm: §1 4-link / §5 6-row consistency, Wilcoxon (not Fisher), Holm-Bonferroni applied, Throttled-HEFT present, D2 row real, abstract reflects Phase 2 reality
- [ ] Address any new findings
- **Wallclock:** ~30 min audit + variable revision
- **Deliverable:** clean audit report
- **Status:** pending

## Sequencing Summary

**Can start NOW in parallel with Phase 0 (D2 training)**:
- Phase B (partial, 5/6 schedulers across all N)
- Phase C (partial, 5/6 schedulers across all ambients)
- Phase D (Throttled-HEFT) — fully independent
- Phase E (partial)
- Phase F (Ours only, fully D2-independent)

**Total parallel wallclock**: ~5 h CPU (can be split across overnight runs).

**Sequential post-D2**: ~30 min CPU.

**Total elapsed D2-sbatch → §5 ready**: ~26 h (D2 training dominates).

## Statistical Methodology (audit gate-blocker fix)

- **Pairing**: `seed_base=100000` shared across A-G; episode k is the same DAG / initial-T / arrival draw across every scheduler.
- **Test**: paired Wilcoxon signed-rank (two-sided). Replaces the Fisher exact in the existing draft.tex §5 (methodology gate-blocker).
- **Multi-comparison**: Holm-Bonferroni across the 5-link ablation chain (HEFT→Decima-v, Decima-v→Decima-t, Decima-t→HGATE, HGATE→Ours, plus Decima-t→D2). Family-wise α = 0.05.
- **Reporting per comparison**: Δ ± std, raw p, Holm-adjusted p, sig markers (`*`/`**`/`***`).
- **Metrics**: `total_makespan_ms`, `peak_temp_episode`, `viol_rate`, `dag_completion_rate`, `episode_return`, `cooling_total_ms`.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 50 ep for Phases A-D, 500 ep for E-F | 50 ep is fast enough for fig data; 500 ep needed for paired Wilcoxon detection power |
| 6 schedulers in main table (drop Round-Robin / Thermal-HEFT) | Stays tight for MDPI Electronics page budget |
| Throttled-HEFT included | Round-1 reviewer obligation; literature reviewer flag |
| `seed_base=100000` invariant across A-G | Keeps paired-Wilcoxon valid across cells |
| Holm-Bonferroni not Bonferroni | Less conservative; standard for ablation chains |
| Ambient sweep: 4 ranges {cold, warm, hot, extreme} | Matches §1 "30-75 °C ambient envelope" claim + adds extreme for bounded-claim panel |
| Phase F restricted to Ours only | §1 contribution 4 is about Ours's bounded claim, not baselines |

## Errors Encountered

| Error | Resolution |
|-------|------------|
| _(none yet — populate as encountered)_ | — |

## Open Decisions (need user input before execution)

1. **Tier 5 episode count**: 500 ep / scheduler (~3h) or 1,000 ep (~6h, tighter CIs)?
2. **Holm-Bonferroni scope**: 5-link chain only (5 tests) or include Throttled-HEFT comparisons (6+ tests)?
3. **Plot regeneration**: reuse `cpo_thermal_v2/evaluation/plots.py` (has D2 palette) or new MDPI 2-col fig scripts?
4. **9-scheduler grand matrix** (CLAUDE.md Stage 5 original, 180k ep, 2-3 days CPU): include or defer to journal extension?

## Files Created / Modified by This Plan

- `cpo_thermal_v2/configs/eval_main_6way_hot_n17.yaml` (new) — Phase A
- `cpo_thermal_v2/configs/eval_scaling_sweep_hot.yaml` (new) — Phase B
- `cpo_thermal_v2/configs/eval_ambient_sweep_n17.yaml` (new) — Phase C
- `cpo_thermal_v2/configs/eval_stat_500ep_hot_n17.yaml` (new) — Phase E
- `cpo_thermal_v2/configs/eval_bounded_n9_extreme.yaml` (new) — Phase F
- `cpo_thermal_v2/scripts/compose_paper_section5.py` (new) — Phase G aggregator
- `paper_drafts/section5_main_results.tex` (modify, Phase G) — D2 row + Holm-adjusted p-values
- `draft/draft.tex` (modify on `paper-draft` branch, Phase G) — wholesale §5 replace

## Cross-Reference

- Paper draft (private): `paper-draft` branch
- Stale §5 in current `draft/draft.tex` will be replaced wholesale at Phase G
- Phase 2 plug-in snippets ready at `paper_drafts/section5_main_results.tex` + `paper_drafts/section5X_hybrid_case_study.tex`
- Audit report: `review_results/draft/review_report.md` (HK-paper-3a session)
- Pending audit items deferred to Phase G: §4 hyper-params block, Holm-Bonferroni note in §5.3
