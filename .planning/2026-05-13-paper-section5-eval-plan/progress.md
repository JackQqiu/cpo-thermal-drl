# Progress Log

## Session: 2026-05-13 — Plan Initialization

### Current Status
- Plan created (this session).
- Phase 0 (D2 V100 training) is **pending user sbatch**.
- Phases A-F can mostly run pre-D2 (Throttled-HEFT, scaling+ambient sweeps for 5/6 schedulers, Phase F is Ours-only).
- Phase G (paper §5 wholesale plug-in) blocks on A-F + D2.

### Done This Session (HK-paper-3a)
- HK-paper-2 (`a5fb98a` on `paper-draft`): §6.3 dual-critic Option B reframe + RC-edge new para + section5_main_results.tex / section5X_hybrid_case_study.tex Phase 2 plug-in prep
- HK-paper-3a (`f55b5be` on `paper-draft`): abstract realignment (4-link, dual-critic delete) + §1 9786 priming
- paper-audit deep-review (5/5 committee reviewers): 67 issues, 22 gate-blockers; pre-Phase-2 fixes closed except §4 hyper-params (deferred)
- Plan initialized via `planning-with-files` skill

### Branch State at Plan Init
- On `main` (HK-5.0 D2 code committed `d654744`)
- `paper-draft` branch ahead by 2 commits (HK-paper-1, HK-paper-2, HK-paper-3a)
- Working tree clean wrt paper files (they only exist on paper-draft)

### Next Actions
1. **User**: sbatch `cpo_thermal_v2/scripts/train_decima_xattn.sbatch` on V100/A100 to kick off Phase 0 (~24h)
2. **Parallel start (optional, can wait until D2 sbatched)**: write eval YAML configs for Phases B/C/D/E/F + start their CPU runs locally
3. **Decision needed before Phase E start**: episode count 500 vs 1000 (open decision #1 in task_plan.md)

### Errors / Blockers This Session
_(none yet)_

---

## Session: 2026-05-13 — D2 Training Progress Update

### D2 V100 training checkpoint @ step 745,472 (~15% of 5M, 152.7 min elapsed)
- mean_ep_ret = +837.35 (cold-curriculum stage)
- H = 1.908 (DROPPING from log(17)=2.833 — clear entropy collapse, unlike HGATE's frozen 2.833)
- KL = 0.2088 (policy updating substantively)
- clipfrac = 0.039 (clip engaging sometimes)
- loss_pg = +0.0074, loss_v = +350.66 (both finite)
- 81 step/s sustained
- **best.pt updated**: rolling_avg(50 eps)=+826.95 at episode 4687, step 745,472

### Interpretation
- Training is healthy. All 5 signals (loss_pg, loss_v, H, KL, clipfrac) green; entropy is collapsing (vs HGATE-PPO H卡死 failure mode).
- Currently in **cold curriculum stage** (cold→warm switch at step 1M per train_decima_xattn.yaml curriculum schedule). Cold-stage ep_ret is typically much higher than warm/hot stages because no thermal violations.
- best.pt at +826 is the cold-stage peak; once warm/hot stages kick in (step 1M+/3M+), ep_ret typically drops and rolling-50 mean gate may or may not fire. Two scenarios for final ckpt:
  - Scenario A: D2 hot-stage learns thermal-anticipation → rolling-50 keeps improving → best.pt updates further
  - Scenario B: D2 hot-stage ep_ret never recovers above +826 → best.pt locked at this cold-stage value (HGATE-style failure)
- Won't know which until full run completes (~14.5h more from this checkpoint).

### ETA
- 5M steps − 745k = 4.255M remaining
- @ 81 step/s = 52,531 sec = ~14.6h
- Total D2 wallclock will be ~17h (faster than HGATE's 24h)

### Next Update
- When user shares step ≥ 1M progress, capture warm-stage entry into mean_ep_ret trajectory.
- When user shares step ≥ 3M, capture hot-stage entry.
- When D2 finishes, kick Phase A.

---

## Session: 2026-05-13 — D2 Training COMPLETE + Phase X Setup

### D2 final state (`checkpoints/decima_xattn_N17/`)
- **best.pt**: global_step=4,628,480, ep_ret_mean=+832.79 (rolling-50 mean at hot-stage gate fire)
- **final.pt**: global_step=5,000,000, ep_ret_mean=+832.79 (= best_ep_ret in metrics_summary; weights differ)
- Best.pt firing in hot stage (step >3M) means D2 rolling-50 mean DROPPED in warm/hot then CLIMBED BACK above the cold-stage peak — strong signal of learned thermal anticipation (vs HGATE where best.pt locked at cold peak and never recovered)
- Sanity check: `DecimaXAttnScheduler` loads cleanly, `schedule()` returns valid action int

### Plan revisions (user decisions on 4 open questions)
1. Phase E ep count: **500 ep** (~3h)
2. Holm scope: **5-link main chain (5 tests)**
3. Figures: **reuse `cpo_thermal_v2/evaluation/plots.py`** (D2 palette already wired)
4. 9-scheduler grand matrix: **run now (2-3 days)** — this triggers the Phase X-A/B/C structure (see task_plan.md)

### Phase X plan structure (replaces original A-F)
- **X-A**: 6-way sanity at N=17 HOT 50 ep (~10 min Mac CPU target) — IN PROGRESS
- **X-B**: 10-scheduler grand matrix × 5 N × 4 ambient × 500 ep = 100k ep (~20h Mac CPU est)
- **X-C**: horizon scan, 4 H × N=17 × HOT × 500 ep × 9 sched = 18k ep (~3-4h Mac CPU est)
- **G**: paper §5 wholesale plug-in (after X-A/B/C)

### Eval YAMLs + drivers created
- `cpo_thermal_v2/configs/eval_main_6way_hot_n17.yaml` (X-A)
- `cpo_thermal_v2/configs/eval_grand_matrix.yaml` (X-B HOT slice)
- `cpo_thermal_v2/configs/eval_grand_matrix_{cold,warm,extreme}.yaml` (X-B sibling ambient slices)
- `cpo_thermal_v2/scripts/eval_grand_matrix.sh` (X-B driver, all 4 ambients sequentially)
- `cpo_thermal_v2/configs/eval_horizon_scan_phaseXC.yaml` (X-C base)
- `cpo_thermal_v2/scripts/eval_horizon_scan_phaseXC.sh` (X-C driver, 4 horizons via override)

### Pipeline schema notes
- Eval runner supports `num_nodes_list × action_mode_list × schedulers` sweep but NOT `initial_temp_range` or `dags_per_episode` sweep — those need separate yaml or driver overrides (hence the 4 grand-matrix sibling yamls + horizon-scan driver)
- All Phase X yamls maintain `seed_base=100000` for pairing across cells

### X-A timing measurement (Mac CPU 2026-05-13 09:00)
- **Wallclock**: 170 sec (2.83 min) for 300 ep total (6 sched × 50 ep × N=17)
- **Avg per-(sched, ep)**: 0.567 sec
- **Per-sched breakdown** (50 ep each):
  - HEFT / RR / ThermalHEFT (classical): ~10 sec
  - Throttled-HEFT-agent_only: ~15 sec (estimated, not in X-A run)
  - Decima-vanilla / Decima-thermal / HGATE-PPO / D2: ~30-35 sec
  - Ours-auto_only: ~45 sec (1.1 ep/s)

### X-A data quality
- 300 rows, 6 schedulers, 0 schedulers with NaN issue beyond expected hybrid-only fields
- Headline §5 ablation chain confirmed (table below)
- NaN count = 300 but localised to 1 column (likely `agent_delay_total_ms` 100% empty for auto_only mode runs); Phase G aggregator must ignore

### D2 vs §5 ablation chain (HK-5.0 deliverable confirmed)
| Scheduler | peak_T | viol_rate | completion | ep_ret |
|---|---:|---:|---:|---:|
| HEFT | 90.67 | 0.960 | 0.320 | +80.71 |
| Decima-vanilla | 90.68 | 1.000 | 0.325 | +90.08 |
| Decima-thermal | 80.21 | 0.480 | 0.686 | +224.91 |
| HGATE-PPO | 88.94 | 0.980 | 0.414 | +117.96 |
| **D2** | **76.16** | **0.380** | **0.772** | **+257.66** |
| Ours-auto_only | 67.50 | 0.000 | 1.000 | +341.66 |

D2 strictly improves over Decima-thermal AND over HGATE-PPO on every safety metric. RC-coupling edge (Ours - D2) closes the residual 8.66°C peak_T gap.

### Phase X-A A100 server variant (next, for X-B/C wallclock decision)
- `cpo_thermal_v2/configs/eval_main_6way_hot_n17_cuda.yaml` (device=cuda variant)
- `cpo_thermal_v2/scripts/eval_phaseXA_a100.sbatch` (sbatch wrapper)
- User to git push + ssh server + sbatch → expected ~30-60 sec elapsed → drives X-B/C partition decision
- Speedup estimate: 3-5× overall (neural-scheduler model forwards go to GPU; env step + classical schedulers stay CPU)
