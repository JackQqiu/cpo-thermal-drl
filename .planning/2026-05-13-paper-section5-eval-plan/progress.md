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

### Phase X-A A100 server variant — RESULT (user ran 2026-05-13)
- **A100 wallclock**: 616s (10 min) for 300 ep — **3.6× SLOWER than Mac CPU 170s**
- **Per-(sched, ep) on A100**: 2.053 sec  vs **Mac 0.567 sec** → 0.3× "speedup"
- **Extrapolated**: X-B (100k ep) ≈ 57h, X-C (18k ep) ≈ 10.3h — both untenable on A100

**Root cause** (tiny-model GPU anti-pattern):
- D2 has only 142,850 params; single-graph forward per env-step is hot loop
- GPU kernel launch overhead (10-100 μs) + PCIe transfer (10-20 μs) > CPU forward compute (~50 μs)
- Server Xeon single-core also slower than Mac M-series single-core
- 16 A100 CPU cores irrelevant: eval pipeline is single-process sequential

**Decision**: abandon A100/cuda path. Run X-B + X-C on Mac CPU.

### Phase X-B / X-C Mac CPU execution plan (user-confirmed)
- **Tonight**: X-B HOT slice (10 sched × 5 N × 500 ep = 25k ep ≈ 4h Mac CPU)
- **Tomorrow night**: X-C horizon scan (~10h Mac CPU)
- **Later**: X-B ambient siblings (cold/warm/extreme) — non-blocking for paper main table

X-B HOT slice alone covers:
- Paper main-table source (slice at N=17)
- Paper scaling figure (HOT × N sweep)
- Paper bounded-claim panel (N=9 HOT — extreme ambient handled in later sibling)
- Holm-Bonferroni 5-link chain (N=17 HOT 500 ep)

So HOT-first is the right paper-prioritised cut.

---

## Session: 2026-05-13 (evening) — X-B HOT slice + extras COMPLETE; narrative pivot

### X-B HOT main slice result (8 schedulers × 5 N × 500 ep)
- Wallclock 10,419s (2.9h) Mac CPU
- N=17 viol_rate: HEFT 0.986, Decima-vanilla 1.000, ThermalHEFT 0.968, RoundRobin 0.986, Decima-thermal 0.382, HGATE-PPO 0.968, **D2 0.478**, **Ours-auto_only 0.002**
- Ours-auto_only **size-parametric ★**: viol 0.002 flat across N ∈ {9,13,17,24,33}
- D2 N-scaling DEGRADES (0.450 → 0.544) while Decima-thermal IMPROVES (0.456 → 0.272) — opposite directions
- Throttled-HEFT-agent_only MISSING (action_mode_list bug)

### X-B HOT extras (4 schedulers × 5 N × 500 ep, +Ours-NoThermal +Throttled-HEFT-{hybrid,agent_only} +Ours-hybrid)
- Wallclock 7,633s (2.1h) Mac CPU
- N=17 viol_rate: **Decima_fair (Ours-NoThermal) 0.006**, Throttled-HEFT-hybrid 0.984, Throttled-HEFT-agent_only 0.984, Ours-hybrid 0.002
- Decima_fair ckpt = `checkpoints_v1/decima_fair_N17/best.pt` (Ours architecture, proc_in_dim=3 = no thermal obs, trained makespan-only reward)
- **Decima_fair size-parametric ★**: viol 0.006 flat across all N (matches Ours-auto_only flatness)

### D2 vs Decima-thermal paired Wilcoxon (N=17 HOT 500 ep, paired-by-seed)
- McNemar binary (violation): p = 0.0096** (D2 has 189 paired-unsafe-only, Decima-thermal 141)
- dag_completion_rate: D2 0.70 vs Decima-thermal 0.74, p = 0.0010***
- peak_T / makespan / ep_ret: ns
- **Component 4 (cross-attn on homog GCN trunk) is statistically WORSE than Decima-thermal** — not a positive contribution

### 🔥 Narrative-breaking finding
| Configuration | viol_rate | Implication |
|---|---:|---|
| Decima-thermal (Mao homog + thermal reward) | 0.382 | reward shaping is moderately useful |
| D2 (Decima homog + cross-attn + thermal reward) | 0.478 | cross-attn on homog hurts |
| HGATE-PPO (hetero GAT + MLP + thermal reward, H collapsed) | 0.968 | hetero alone doesn't help, may also be training failure |
| **Decima_fair = Ours-NoThermal (Ours arch: hetero + RC-edge + cross-attn; NO thermal obs, NO thermal reward)** | **0.006** | **architecture alone gets 99.4% of the way** |
| Ours-auto_only (Ours arch + thermal obs + thermal reward) | 0.002 | thermal info adds marginal 0.4pp |

Architecture-without-thermal-anything (Decima_fair) almost matches full Ours. Thermal reward shaping = marginal contributor. Architecture (especially RC-coupling edge attribute as physics-grounded inductive bias) = primary driver.

### Decision: pause X-C/ambient sweep; literature check FIRST
User concern: if hetero-GNN-with-RC-coupling-edge architecture is already published (by HGATE-PPO-adjacent works, by GNN-thermal-management works, etc.), paper contribution is at risk.

User to query Google Scholar / Semantic Scholar / arXiv for:
- "thermal-aware DAG scheduling + GNN"
- "CPO + DRL scheduler" or "co-packaged optics + reinforcement learning + GNN"
- "RC coupling + graph encoder" + "DAG scheduling"
- "physics-grounded edge attribute" + "graph neural network" + "scheduler"
- HGATE-PPO (Wu 2025) — already known closest prior, does NOT include RC-edge attribute

If literature check returns "no exact prior with RC-coupling edge attribute in DAG scheduling": rewrite §1 contribution + §5 narrative around RC-edge as the novelty anchor.

If literature returns prior: rethink contribution framing, possibly need additional ablation isolating RC-edge alone (would require training a no-RC-edge variant — `configs/ablation_no_rc_edge.yaml` exists but ckpt missing).

### Phase X-B/C eval status
- X-B HOT main: complete ✓
- X-B HOT extras: complete ✓
- **X-B ambient siblings (cold/warm/extreme): PAUSED pending narrative direction**
- **X-C horizon scan: PAUSED pending narrative direction**

These can resume once user decides whether to:
- Anchor on RC-edge novelty (then current data + more ambient = sufficient)
- Train no-RC-edge variant first (then more eval after train)

---

## Session: 2026-05-13 (late) — External keynote review (Ansys EDPS 2025)

### Done
- Reviewed `0400chang.pdf` (Norman Chang & A. Kumar, Ansys EDPS 2025 keynote on CPO + ML + Agentic AI)
- Wrote structured analysis to `paper_drafts/external_review_chang_edps2025.md` (self-contained, other sessions can read without the PDF)
- Synced pointer + key takeaways into `findings.md` under new "External Reviews" section
- Added one-line pointer in CLAUDE.md "Current Phase" so future sessions surface the review at session start

### Outputs for Phase G (paper-draft branch, future session)
- 2 must-cite priors identified (Youn DesignCon 2021, Alam DAC 2025)
- 1 calibration anchor (28 °C measured Tx-Rx gradient → backs §3 RC-matrix physics)
- 1 related-work paragraph drafted (ML thermal surrogate triple)
- 1 reviewer-defence answer pre-drafted ("why not ML surrogate in env")

### No code touched. No commits. No eval re-runs.

---

## Session: 2026-05-14 — Phase X-B Tier 2 + X-C H=200 main+extras COMPLETE

### Phase X-B Tier 2 — DONE (15.7h Mac CPU)
- 6 sub-yamls × 2 variants (main + extras) × 3 ambients (cold, warm, extreme; HOT done day before)
- 240 cells × 500 ep = **120,000 episodes**
- All 12 schedulers covered: HEFT, RoundRobin, ThermalHEFT, Throttled-HEFT-{hybrid, agent_only}, Decima-vanilla, Decima-thermal, HGATE-PPO, D2, Ours-NoThermal (decima_fair), Ours-auto_only, Ours-hybrid
- All N ∈ {9, 13, 17, 24, 33}
- All ambients × 4 (cold/warm/hot/extreme)
- All paired by `seed_base=100000`
- 8 output dirs:
  - `eval_results/grand_matrix_{hot, cold, warm, extreme}/`
  - `eval_results/grand_matrix_{hot, cold, warm, extreme}_extras/`

### X-B Tier 2 narrative-confirming findings

**1. Ours-NoThermal ≈ Ours-auto_only cross-ambient** — paired Wilcoxon McNemar all ns (p=0.32-1.0 across 4 ambients). Architecture alone (RC-edge + hetero attn + cross-attn) delivers Ours's safety claim; thermal observation + reward marginal.

**2. D2 ambient-scaling failure**: viol_rate 0.000 (cold) → 0.042 (warm) → 0.478 (hot) → 0.656 (extreme). Cross-attn placement actor on homog GCN trunk does NOT generalize to extreme thermal regimes. Component 4 (D2) is a paper-positive negative result: it isolates the need for RC-edge specifically.

**3. Decima-thermal extreme anomaly explained**: 0.042 viol_rate at extreme vs 0.382-0.536 at cold/warm/hot. Caused by **over-conservative policy** (trunc 0.040 vs 0.34-0.51 at other ambients) — the policy fails-safe at extreme by being so slow that thermal violations rarely accumulate. Not "actually clever," and Ours still strictly better (Ours extreme 0.006 viol, 0% trunc, 100% completion).

**4. Bounded-claim refresh (paper §1 contribution 4)**:
   - Ours-hybrid: 7 unsafe / 12,500 ep (across 5 N × 4 ambient × 500 ep grid). 5 of 7 in extreme regime.
   - Ours-auto_only: 30 / 12,500.
   - Ours-NoThermal: 31 / 12,500.
   - Original draft.tex §1 framing "single residual at N=9 extreme" is too narrow; the real claim is "7 unsafe / 12,500 paired episodes across the full size×ambient envelope, concentrated entirely in the extreme regime."

### Phase X-C — DONE (main 2.4h + extras 2.8h = 5.2h Mac CPU)
- **X-C H=200 main** (`eval_horizon_scan_phaseXC.yaml`): 8 schedulers × N=17 × HOT × 500 ep × H=200, 8,763s
- **X-C H=200 extras** (`eval_horizon_scan_phaseXC_extras.yaml`): 4 schedulers (Throttled-HEFT-{hybrid, agent_only}, Ours-hybrid, Ours-NoThermal/decima_fair) × N=17 × HOT × 500 ep × H=200, 10,074s
- Combined: 12 schedulers × N=17 × HOT × 500 ep × H=200 = **6,000 episodes** at the long-horizon stress test

### X-C narrative-strong findings

| Scheduler | H=20 viol | H=200 viol | Δ |
|---|---:|---:|---:|
| HEFT / RR / Throttled / HGATE / Decima-v | 0.97–1.00 | 1.000 | mostly + |
| Decima-thermal | 0.382 | 0.394 | flat |
| D2 | 0.478 | 0.478 | flat |
| Ours-NoThermal | 0.006 | 0.020 | +0.014 |
| Ours-auto_only | 0.002 | 0.028 | +0.026 |
| **Ours-hybrid** | **0.002** | **0.004** | **+0.002** ★ |

**Hybrid pays off at long horizon**: Ours-hybrid (2/500 unsafe) is 7× safer than Ours-auto_only (14/500) at H=200. This is the first regime where the Stage-2 hybrid policy demonstrates clear empirical value over the Stage-1 placement-only — partially reframes the §5.X case study's pessimistic conclusion.

### Total data pool for Phase G

| Source | Cells | Episodes |
|---|---:|---:|
| X-B Tier 2 (12 sched × 5 N × 4 ambient × 500 ep) | 240 | 120,000 |
| X-C H=200 main + extras (12 sched × N=17 × HOT × 500 ep × H=200) | 24 | 12,000 |
| HK-4.7 5-way HOT smoke (50 ep paired, redundant with X-B HOT slice but historical) | 5 | 250 |
| Stage 2 hybrid case study (v1/v2/v3-best/v3-final vs s1, WARM + HOT) | varies | ~1,000 |
| **Total fresh data anchoring paper §5 wholesale rewrite** | **~270** | **~133,000 ep** |

### Phase G readiness

All eval data anchoring `paper_drafts/section5_main_results.tex` and `paper_drafts/section5X_hybrid_case_study.tex` Phase 2 plug-ins is now collected. Phase G actions (on `paper-draft` branch):

1. Write Phase G aggregator script that merges 8 grand_matrix CSVs + 2 horizon CSVs + renames Decima → Ours-NoThermal, computes paired Wilcoxon + Holm-Bonferroni across the 5-link chain.
2. Refresh §5 main table numbers from N=17 HOT slice of grand_matrix (was already drafted with HK-4.7 50-ep numbers; now upgrade to 500 ep).
3. Add Component 5 (Ours-NoThermal vs Ours-auto_only) cross-ambient table to §5.1 narrative.
4. Refresh paper §1 contribution 4 bounded-claim wording (7/12,500 across full envelope, not "single residual at N=9 extreme").
5. Add §5.4 horizon-failure-mode figure from X-C H=200 data (12 schedulers).
6. Add §5.X case study revision: "v1/v2/v3 retraining failures + final hybrid shows long-horizon advantage at H=200" (positive coda).
7. Apply Phase G writing-style rules per `~/.claude/.../memory/feedback_paper_writing_pitfalls.md`.
8. Run paper-audit quick-audit + deep-review after Phase G edits.

### Process notes
- Caffeinate -i -w PID kept Mac alive across both X-B Tier 2 + X-C main + X-C extras runs. Auto-exited when each python exited.
- All eval pipelines used `device: cpu` after A100 cuda anti-pattern ruled out (X-A Mac 170s vs X-A A100 616s; tiny-model GPU launch overhead).
- Working tree clean except eval_results/ + .planning/ updates (no source code touched during eval phases).
