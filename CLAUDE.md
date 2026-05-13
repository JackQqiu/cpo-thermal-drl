# CPO 论文 SOTA 复现 + Env Fix Handoff (v2)

**目标**：在 Claude Code 中接续工作。本文档自洽——包含所有 context、已做决策、阶段化执行计划，让 Claude Code session 能独立推进，不需要回 chat 问背景。

---

## ⏱ Current Phase (last updated: 2026-05-13)

**Active**: D2 (Decima encoder + cross-attention) baseline — **Phase 0 pending**:
user-side sbatch of `cpo_thermal_v2/scripts/train_decima_xattn.sbatch` on
V100/A100 (~24 h wallclock). Once ckpt lands at
`checkpoints/decima_xattn_N17/best.pt`, Phase A-G eval pipeline executes
per the plan file.

**Active plan file (planning-with-files skill)**:
`.planning/2026-05-13-paper-section5-eval-plan/task_plan.md`
- Read at the start of every session to recover state.
- Companion files in the same dir: `findings.md` (validated data +
  audit findings + ckpt locations), `progress.md` (session log).
- The skill's hooks auto-inject the plan head into tool-call context, so
  the plan is in attention without manual re-reads.

**Last commits**:
- `d654744` (HK-5.0): D2 baseline code on `main` — encoder + actor-critic
  + sbatch + train + eval factory + 27 unit tests pass + 1024-step Mac
  CPU smoke shows H collapsing 2.832 → 2.797 (vs HGATE 卡死)
- `f55b5be` (HK-paper-3a) on `paper-draft`: abstract realignment
  (3-tier → 4-link match with §1) + dual-critic overclaim removal +
  §1 9786 HEFT-failure priming
- `a5fb98a` (HK-paper-2) on `paper-draft`: §6.3 dual-critic Option B
  echo-chamber fix + new RC-edge paragraph + Phase 2 plug-in snippets

**Pending (per task_plan.md phases)**:
- Phase 0: user sbatches D2 V100 (24 h)
- Phases B/C/D/E/F (mostly D2-independent): can start any time —
  scaling sweep, ambient sweep, Throttled-HEFT, 500-ep Wilcoxon,
  bounded-claim N=9 extreme
- Phase A: 6th row (D2) after Phase 0
- Phase G: §5 wholesale plug-in into `draft/draft.tex` on `paper-draft`
- Phase H: paper-audit re-run

**Paper §5 main table**: 5/6 rows ready in
`eval_results/hgate_final_5way_hot/episodes.csv`; D2 row pending.

**Stage 状态** (磁盘真相，不是计划):
- ✅ Stage 0 — env fixes + sanity (commit `8802a3d`)
- ✅ Stage 1 — `checkpoints/stage1_auto_only_N17` (commit `HK-1.*`)
- ✅ Stage 2 retrain — `checkpoints/stage2_hybrid_v3_stress_N33` (commit `HK-1.5.9`)
- ✅ Stage 2 (真 Decima 复现) — `checkpoints/decima_true_vanilla_N17` (commit `HK-3.1.2`)
- ✅ Stage 3 (HGATE-PPO) — 代码 + 训练完毕 (HK-4.1..HK-4.7); H 训练时早 collapse, best.pt
  step=1.12M, paper §5.1 诚实报告 (HGATE-PPO peak_T 88.94°C)
- ✅ Stage 4 (Throttled-HEFT 代码) — commit `HK-2.1` / `HK-2.2`; **eval 仍未跑**
  (round-1 reviewer obligation — list 在 task_plan.md Phase D)
- 🟡 Stage 5 (D2 + eval pipeline) — 代码 + smoke 完毕 (HK-5.0), V100 训练待
  user sbatch
- ⬜ Stage 6 — Paper §5 wholesale rewrite (Phase G in task_plan.md);
  Phase 2 plug-in snippets ready at `paper_drafts/section5_main_results.tex`
  + `paper_drafts/section5X_hybrid_case_study.tex` on `paper-draft` branch

> ⚠ 下方 §1 / §3 还保留原始 handoff 视角 (写于 Stage 0 之前)，对历史 stage 的"执行计划"不再准确。当成"参考文档 + 未来 stage 的指南"读，**真实状态以本节为准**。本节 + `.planning/2026-05-13-paper-section5-eval-plan/` 是 cross-session 状态来源。

---

## 工作环境约定（必读）

- **Claude Code 跑在本地（用户的 Mac/PC）**，不在 GPU 服务器上。Claude Code 不能 ssh，不能直接在远程开训练。
- 所有 GPU-heavy task（Stage 1 / 2 / 3 训练、Stage 5 eval matrix）由用户手动在服务器上跑。Claude Code 不要尝试运行 `python -m ... train ...` 等长跑命令。
- **Claude Code 的职责**：
  - 写代码（env fix、decima_true.py、hgate_ppo.py、训练脚本）
  - 跑本地 sanity check（CPU 够用的）
  - 写 smoke test：小规模配置（N=9、50 ep、200 train steps）能本地跑通的版本，证明代码不崩
- Stage 0 sanity check 是纯 CPU 任务，**必须在本地全部四个 check PASS** 才能告知用户进 Stage 1。
- Stage 1+ 让用户开训练前，先在本地跑对应的 smoke test 验证代码可运行，再让用户上服务器跑正经训练。
- 同步机制：git。详细规则见下方 "Git 工作流" 章节。

### Cross-session 进度持久化协议（planning-with-files skill）

跨 session 的大块任务（multi-phase eval、paper revision、ablation chain）
用 `planning-with-files` skill 持久化进度，不依赖 chat history 或 user
re-brief。文件位置约定：

- **Active plan**：`.planning/<YYYY-MM-DD>-<slug>/{task_plan.md,findings.md,progress.md}`
- **当前激活 plan 指针**：`.planning/.active_plan` 内容为 plan 目录名
- **CLAUDE.md "⏱ Current Phase" 段** 必须引用 active plan 路径（已经引用就行；
  换 phase 时把整个 plan dir 路径更新一下）
- **Auto-memory** (`~/.claude/projects/.../memory/MEMORY.md`) 同步指针，
  避免 CLAUDE.md 漂移时丢线索

skill 配的 hooks 会在每次 `Write|Edit|Bash|Read|Glob|Grep` tool call 前把
`task_plan.md` 头 30 行注入 context — **不需要主动 re-read plan**，
attention window 里会自动出现。

**Session 开局** (任何新 session 触碰任何 tool 之后)：

1. Hooks 已经把 plan head 推进 context → 已经知道 active phase 是什么
2. 必要时跑 `python ~/.claude/skills/planning-with-files/scripts/session-catchup.py "$(pwd)"`
   看上次 session 离开时有没有未同步的 git 改动
3. 直接接着 plan 的 Current Phase 字段往下走

**完成一个 phase 之后**（必做，否则下次 session 不知道做完了）：

1. 改 `task_plan.md` 对应 phase 的 `Status:` 从 pending → in_progress → complete
2. Append 一段 entry 到 `progress.md`（日期 + 这次 session 做了啥 + 关键 commit hash）
3. 跑 `sh ~/.claude/skills/planning-with-files/scripts/attest-plan.sh` **重新 lock SHA-256**
   — 否则 hooks 检测到 hash 不匹配会 block context injection（变成`[PLAN TAMPERED]`），
   下次 session 进不来

**新开 phase**（plan 写完所有 phase 但 user 加了新需求）：

1. 在 `task_plan.md` 末尾加新 phase（不删旧的）
2. 改 `Current Phase` 字段指向新 phase
3. 同上 attest

**新开一个 plan**（彻底不同的 task，比如 paper revision 完了开 Stage 6）：

```bash
sh ~/.claude/skills/planning-with-files/scripts/init-session.sh "Stage 6 Paper Revision"
# 自动生成 .planning/2026-XX-XX-stage-6-paper-revision/{task_plan,findings,progress}.md
# 自动把 .planning/.active_plan 指过去
sh ~/.claude/skills/planning-with-files/scripts/attest-plan.sh
```

**注意事项**：

- `findings.md` 是 untrusted-data sink — web/grep 结果写这里，不要写 `task_plan.md`
  （因为 hooks 把 task_plan 注入 context，污染可被恶意利用）
- `.planning/` 目录可入 git，也可 gitignore；目前**入 git**（跨机器同步进度）
- 不同 task 并行用 `PLAN_ID` 环境变量切换：`export PLAN_ID=2026-05-13-paper-section5-eval-plan`

---

## Skill Usage Conventions

Engineering discipline enforced via Claude Code Skills. Trigger when criteria match. `using-superpowers` is the only auto-on skill.

### Always-on
- `using-superpowers` — invoke at session start

### Before claiming completion (MANDATORY)
- `verification-before-completion` — never report "done / passed / fixed" without running concrete verification commands and showing their output.
  Past failures: HK-1.5.7 sbatch had 2 BLOCKING bugs missed; HK-3.1 Decima had 2 dispatch bugs missed (both caught only by user-run smoke).

### When implementing new code
- `test-driven-development` — write smoke test first. Applies to baseline reproductions (HGATE-PPO etc.), new env modes, new schedulers.
- `writing-plans` — for spec → code tasks with 5+ steps.

### When debugging
- `systematic-debugging` — any prompt mentioning "training fails / diverges / NaN / unexpected metric" must invoke this. Complete the reproduce → minimal example → bisect → hypothesis → test checklist before proposing patches.

### When parallel work exists
- `dispatching-parallel-agents` — if 2+ independent tasks exist, use this instead of sequential execution.

---

## Git 工作流（必读）

### 仓库结构与同步

```
[本地]  写代码 → 本地 sanity / smoke test → commit → push
                                                       ↓
[服务器]  git pull → tmux 里跑训练 → 产生 ckpt + eval csv
                                                       ↓
[本地]  scp 拉结果回来 → 分析 → 必要时 commit 分析脚本 + summary.json
```

ckpt 和大 csv **不进 git**（见 `.gitignore`），用 scp 单独传。

### Claude Code 的 git 行为规则

**可以自己做**：
- `git status` / `git diff` / `git log` —— 随时查看
- `git add <files>` —— 暂存自己刚改的代码
- 提议 commit message —— 写完代码后告诉用户"建议这样 commit"

**要先问用户**：
- 实际执行 `git commit` —— 让用户 review diff 再 confirm
- 创建 / 切换 / 删分支
- 任何 destructive 操作（`reset --hard`、`clean -fd`、`checkout` 覆盖未提交改动）

**绝对不做**：
- `git push` —— 永远由用户来推
- `git rebase` 或 force push
- 把 ckpt / 大 csv 加进 git。即使 `.gitignore` 漏配，看到 `*.pt` / 大 `*.csv` 被 stage 时应停下提醒用户

### 每 stage 的 commit checkpoint

每个 stage 的 sanity / smoke test 全 PASS 后，commit 一次。message 格式：

```
Stage <N>: <one-line summary>

- 关键变更 1
- 关键变更 2
- Sanity/smoke check: PASS（贴关键数字）
```

示例（Stage 0 完成时）：

```
Stage 0: env fixes — _maybe_precool predicted_peak loop + temp_rise constants

- _maybe_precool 改用 _simulate_execution_peak 估 predicted peak，
  循环退出条件 predicted_peak <= thermal_guardband
- temp_rise_per_ms_asic: 0.5 → 0.08 (matches RC matrix calibration)
- temp_rise_per_ms_oe:   0.1 → 0.18 (OE is the bottleneck, not ASIC)
- 新增 scripts/sanity_check_env_fixed.py
- Sanity check 4/4 PASS:
  A: 常数生效 ✓
  B: predicted_peak = 77.4 ✓ (旧值 75.0)
  C: HEFT extreme — cooling 132 ms/ep, peak_T 82.3, truncate 22% ✓
  D: 旧 ckpt hybrid vs auto_only 可区分 ✓
```

建议 tag 关键节点（用户来打）：
- Stage 0 完成：`git tag stage0-env-fixed`
- Stage 1 服务器训完：`git tag stage1-retrained-v1`
- 所有 baseline 训完：`git tag baselines-ready`
- 投稿前：`git tag submission-v1`

### 必入 git

- `CLAUDE.md` —— 项目记忆，绝不能丢
- `cpo_thermal_v2/` —— 所有源码
- `configs/*.yaml` —— 训练配置
- `*.tex` —— paper 源
- 每 stage 的 summary `.json`（headline 数字，方便回看进度）

### 不进 git（见 `.gitignore`）

- `*.pt`、`*.pth`、`checkpoints/` —— 大 binary
- `eval_results/*.csv`、`runs/`、`tensorboard_logs/` —— 大 CSV / 训练日志
- `__pycache__/`、`.ipynb_checkpoints/` —— 缓存
- LaTeX build 产物（`*.aux`、`*.log`、`*.synctex.gz` 等）

### 服务器端 workflow

服务器 clone 同一个 repo，**只 pull 不 push 代码**（避免在服务器上意外改了代码）：

```bash
# 服务器上
cd /path/to/cpo_project
git pull                         # 拉本地最新代码
tmux new -s train                # 进 tmux
python -m cpo_thermal_v2.scripts.train --config ...  # 跑训练
# Ctrl+B D detach；训练在后台跑
```

训完后 ckpt 和 eval csv 用 scp 拉回本地：
```bash
# 本地
scp user@server:/path/to/cpo_project/checkpoints/stage1_v2.pt ./checkpoints/
scp user@server:/path/to/cpo_project/eval_results/*.csv ./eval_results/
```

---

## 0. 项目 Context

- **论文**：CPO（Co-Packaged Optics）数据中心 thermal-aware microservice DAG scheduling，PPO + hetero-GATv2 actor-critic
- **当前 draft**：`draft__6_.tex`（3672 行），已经处理了一轮 reviewer critique，但 baseline 太弱
- **Reviewer 要求 + 自加扩展**：
  1. 真 Decima 复现（同构 GCN + REINFORCE，Mao 2019 SIGCOMM）— **reviewer 直接要求**
  2. like-for-like classical baseline（Throttled-HEFT，已有代码）— **reviewer 直接要求**
  3. 2025 SOTA 同期对比 → HGATE-PPO（Wu 2025 IEEE IoT Journal）— **自加，强化 contribution**
- **没有 deadline 压力** — 这次目标是做对，不是做完。

---

## 1. 当前状态

### 1.1 已就绪
- Ours-hybrid / Ours-auto_only / Ours-NoThermal 模型 + checkpoints — **但是在 buggy env 上训的，要重训**
- HEFT / Thermal-HEFT / RoundRobin 三个 classical baselines
- Throttled-HEFT v5 augmented（已写好，未跑全 eval）
- Paper draft `draft__6_.tex`

### 1.2 已发现的 env bugs（两个，必须同时修）

**Bug 1: `_maybe_precool` 退出条件错**
- 已在前序 session 修复 (`cpo_thermal_env.py:798-875`, predicted-peak loop + caller-managed snapshot/restore)。本 session 通过 Check D 的 +333 ms cooling delta 独立验证仍在工作。

**Bug 2: `temp_rise_per_ms_*` 跟 RC 矩阵物理实际不一致**
- 当前 default：`temp_rise_per_ms_asic = 0.5`, `temp_rise_per_ms_oe = 0.1`
- 实际 matrix calibration（70°C 起点 + 持续 active power）：ASIC ≈ **0.08** °C/ms, OE ≈ **0.18** °C/ms
- 问题：6× / 2× 数值偏差让所有用 `est_dT` 的 logic（GNN edge feature, Thermal-HEFT, Throttled-HEFT trigger）都被 mislead — 以为 ASIC 是 hot path，**其实 OE 才是 bottleneck**（OE 散热弱 G_ENV_OE=1.5，cross-coupling 累积升温到 85°C 触发 truncate）
- Fix：直接改两个 scalar default

### 1.3 还没真复现的 baselines
- **真 Decima**（Mao 2019）：当前 `decima.py` 是 mask 热特征的 hack，`decima_fair.py` 是 HeteroEncoder + PPO（架构不对），**两个都不是 Decima**
- **HGATE-PPO**（Wu 2025）：完全没动，从零写

---

## 2. 已确认的关键决策（Claude Code 不需要再问）

1. **`_maybe_precool` 修复策略**：选最干净版本 — loop 内 re-simulate execution peak，cool until `predicted_peak ≤ guardband`。理由：算法跟 paper claim 一致，不引入 magic number。
2. **重训范围**：完整 stage1（auto_only，~6h V100）+ stage2（hybrid 从 stage1 warm-start，~3h V100）。理由：reward channel 设计是为 hybrid mode 准备的，buggy env 下训出的 stage2 严格说不是论文设计的算法。
3. **真 Decima 复现**：必须做。**不能**用 HeteroEncoder 简化版顶替（之前两次都是这样错的）。
4. **HGATE-PPO 复现**：必须做，从零写。
5. **`oe_active_power` 决策**：保留 code 现状 = **40 W**，**改 paper §3.2 那句**对齐到 40。理由：code 跑过几百小时训练，grounded 在实测；改 code 反而引入新风险。
   - ⚠ 如果用户改主意要改 code 到 15 W：需要重新 calibrate matrix + 全部重训。**默认走 40。**

---

## 3. 阶段化执行计划

依赖关系：**Stage 0 必须 100% 完成且通过 sanity check 才能进 Stage 1。Stage 1 跑完后，Stage 2 和 Stage 3 可以并行**（不同代码、互不依赖）。

---

### Stage 0：Env fixes（阻塞所有后续）

#### Step 0.1：定位代码

env 文件路径：`cpo_thermal_v2/envs/cpo_thermal_env.py`（重组后）。两个 fix 都在这一个文件。

```bash
grep -n "temp_rise_per_ms_asic\|temp_rise_per_ms_oe" cpo_thermal_v2/envs/cpo_thermal_env.py
grep -n "def _maybe_precool\|def _simulate_execution_peak" cpo_thermal_v2/envs/cpo_thermal_env.py
```

#### Step 0.2：Fix 1 — 常数对齐到矩阵物理

`__init__` 默认值改成：
```python
temp_rise_per_ms_asic: float = 0.08,   # was 0.5 — matches RC matrix calibration
temp_rise_per_ms_oe:   float = 0.18,   # was 0.1 — OE is the bottleneck, not ASIC
```

⚠ **同时检查全项目的 explicit override**：
```bash
grep -rn "temp_rise_per_ms_asic=\|temp_rise_per_ms_oe=" .
grep -rn "temp_rise_per_ms_asic:\|temp_rise_per_ms_oe:" configs/
```
如果 yaml configs 或 train scripts 传了 explicit 值，**必须同步**——否则 default 不生效。

#### Step 0.3：Fix 2 — `_maybe_precool` 重写

新 logic：

```python
def _maybe_precool(self, target_proc_idx: int,
                   exec_time_ms: float,
                   traffic_load: float) -> tuple[float, bool]:
    """Pre-cool until the predicted execution peak is safe.

    Loop:
      1. Simulate executing this dispatch at current state, get predicted_peak.
      2. If predicted_peak <= thermal_guardband: exit (no more cooling needed).
      3. Else: advance env by 1 ms idle (no power), accumulate cooling_used.
      4. If cooling_used >= max_cooling_steps_ms: exit (give up, episode will likely truncate).
    """
    cooling_used = 0.0
    fired = False
    while True:
        predicted_peak = self._simulate_execution_peak(
            target_proc_idx, exec_time_ms, traffic_load
        )
        if predicted_peak <= self.thermal_guardband:
            break
        if cooling_used >= self.max_cooling_steps_ms:
            break
        # Advance env by 1 ms with idle power (no dispatch)
        self._idle_step(dt_ms=1.0)
        cooling_used += 1.0
        fired = True
    return cooling_used, fired
```

注意：
- `_simulate_execution_peak` 内部应已有 snapshot/restore（不会污染外部 state）。如果没有，先加。
- `_idle_step` 是真实推进 env state（不是模拟），cooling_used 是真实 ms。
- 别忘了把 `cooling_used` 计入 episode metric（`env_cooling_ms`），reward channel 会用。

#### Step 0.4：Sanity check 脚本

新文件 `cpo_thermal_v2/scripts/sanity_check_env_fixed.py`，验证 4 个 check：

```python
# Check A: 常数已生效
assert env.temp_rise_per_ms_asic == 0.08
assert env.temp_rise_per_ms_oe   == 0.18

# Check B: _simulate_execution_peak 单步行为
# 设置 T[0]=75, T[1..]=65, dispatch 30ms ASIC
# 期望 predicted_peak ≈ 77-79 (用新常数)
# 旧版会返回 75.0（fire 不了 _maybe_precool）

# Check C: HEFT in extreme regime, 50 ep
# 旧 env: cooling≈0, peak_T≈91.5, truncate=96%
# 新 env: cooling>100 ms/ep, peak_T<85, truncate<30%

# Check D: 旧 Ours-hybrid ckpt vs 旧 Ours-auto_only ckpt on fixed env, 100 ep
# 行为应该可区分（即使旧 ckpt 不是最优）
# 如果 hybrid vs auto_only 数据完全一样 → fix 没生效（或 reward channel 还有别的 bug）
```

四个 check 全 PASS 才能进 Stage 1。

#### Step 0.5：Paper §3.2 同步（不阻塞，但要做）

- Line ~896 `ASIC dissipates 10× hotter` → `dissipates 10× more heat`（单位错）
- §3.2 thermal model 里 OE per-engine power 描述对齐到 `40 W`

---

### Stage 1：重训 Ours（fixed env 上）

#### Step 1.1：Stage 1（auto_only）

```bash
python -m cpo_thermal_v2.scripts.train --config configs/stage1_auto_only.yaml \
    --output checkpoints/stage1_auto_only_v2.pt
```

预算：5×10⁶ env steps，curriculum cold/warm/hot 在 1e6 / 3e6 切换，~6h V100。

**TensorBoard watchlist**：
- `train/return` 平稳上升，hot stage 后 ≥ 旧 buggy ckpt 的 return
- `train/truncate_rate` 显著低于旧版
- `eval/peak_T` < 85
- `train/env_cooling_ms` 在 hot stage 应 > 0（旧版几乎 = 0）

#### Step 1.2：Stage 2（hybrid，warm-start）

```bash
python -m cpo_thermal_v2.scripts.train --config configs/stage2_hybrid.yaml \
    --warm_start checkpoints/stage1_auto_only_v2.pt \
    --output checkpoints/stage2_hybrid_v2.pt
```

预算：1.5×10⁶ env steps（lr=1e-4），~3h V100。

**核心 sanity（必查）**：训完后跑 100 ep eval，确认 **hybrid vs auto_only 数据真的拉开**（这是论文核心 claim）。如果几乎相同 → 停下来 debug，**不要**继续。这是 reward channel 设计或 dual critic 的信号。

---

### Stage 2：真 Decima 复现（Mao 2019 SIGCOMM）

⚠ **不能用 HeteroEncoder 简化**。Reviewer 要求的是 controlled comparison：**同样 reward shaping**，**只换 encoder + RL algo**，分离出 "hetero edge typing 的价值"。

#### Step 2.1：架构（新文件 `cpo_thermal_v2/baselines/decima_true.py`）

```python
"""
Mao et al. 2019 SIGCOMM faithful reproduction.

Architecture differences from Ours:
- Homogeneous GCN encoder (single node type) — NOT hetero GAT
- Edges = DAG dependencies ONLY — NOT thermal coupling, NOT task-proc affinity
- Two-stage policy:
    Stage A: scalar score per ready DAG → softmax → pick DAG
    Stage B: score per (ready task, proc) pair within chosen DAG → softmax → pick pair
- REINFORCE with moving-average baseline — NOT PPO, NOT GAE, NOT clipping
- One gradient step per episode — NOT minibatched

Reward shaping = SAME as Ours (eq 14). Only encoder + algo differ.
This isolates 'homogeneous GNN + REINFORCE' contribution.
"""

from torch_geometric.nn import GCNConv

class DecimaGCNEncoder(nn.Module):
    """3-layer GCN, mean aggregation."""
    # task feature + proc feature 都拼到统一 node feature dim
    # node type 用 one-hot indicator concatenate 进 feature
    # edge index = DAG precedence + self-loop only

class DecimaTwoStagePolicy(nn.Module):
    """Stage A (DAG selection) + Stage B (task-proc pair selection)."""

class DecimaScheduler(BaseScheduler):
    """Trained-policy wrapper for evaluation."""
```

**关键细节**：
- Node features：task 和 proc 都当同一种 node，feature 拼接 + one-hot type 标识
- Edges：**只**保留 DAG precedence + self-loop。**不**加 task-proc affinity edge，**不**加 thermal coupling edge
- GCN aggregation：**mean**（不是 GAT attention）

#### Step 2.2：训练 loop（新文件 `cpo_thermal_v2/scripts/train_decima_true.py`）

REINFORCE pseudocode：
```python
moving_avg_baseline = 0.0
for episode in range(total_episodes):
    states, actions, log_probs, rewards = run_episode(env, policy)
    G_t = discounted_returns(rewards, gamma=0.99)
    advantages = G_t - moving_avg_baseline
    loss = -mean(log_probs * advantages.detach())
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    moving_avg_baseline = 0.99 * moving_avg_baseline + 0.01 * G_t.mean().item()
```

预算：5×10⁶ env steps（match Ours stage 1）。如果 REINFORCE 不收敛——这本身是数据点（"REINFORCE struggles in this regime"）。如果**完全**爆炸，备选 Decima++（Liu 2022 PPO 改进）但要在 paper 注明。

#### Step 2.3：注册

```python
# baselines/__init__.py
from .decima_true import DecimaTrueScheduler
```

`scripts/evaluate.py` 加 `--scheduler decima_true` 选项。

---

### Stage 3：HGATE-PPO 复现（Wu 2025 IEEE IoT Journal）

#### Step 3.1：参考资源

- Paper: Wu et al. "Dependency-Aware Task Offloading Strategy via Heterogeneous Graph Neural Network and Deep Reinforcement Learning", IEEE IoT Journal 2025, vol 12 no 13, pp 22915-22933, doi 10.1109/JIOT.2025.3549441
- Code: https://github.com/JM-Wu-BIT/HGATE-PPO

#### Step 3.2：复现规则（critical）

- ✅ 保留原架构：hetero-GAT + 标准 PPO
- ❌ **不要**加 RC coupling edge attribute — 这是 reviewer 想看的对比："generic hetero-GAT vs CPO-specific hetero-GAT"。偷偷加 thermal physics → 对比失效
- 改 action space：原版 vehicular cloud 是连续 `Discrete(N)`（哪个 server），我们 hybrid mode 是 `MultiDiscrete([N, K_delay])`
- 写 input adapter：他们 graph format → 我们 `graph_obs` schema 的转换

#### Step 3.3：实现（新文件 `cpo_thermal_v2/baselines/hgate_ppo.py`）

```python
"""
Wu et al. 2025 IEEE IoT Journal HGATE-PPO faithful reproduction.

Architecture:
- Heterogeneous GAT encoder (task / processor node types)
- Standard PPO trainer (clipping, GAE, value bootstrap)
- Edges: DAG dependency + task-proc affinity, scalar weight only
- NO RC coupling edge attribute (this is the controlled-comparison point)

Adapted from their vehicular-cloud action space to our discrete dispatch.
"""
```

#### Step 3.4：训练（`cpo_thermal_v2/scripts/train_hgate_ppo.py`）

可以复用我们 PPO trainer 改造，预算 ~3-4h V100（架构跟我们 stage 1 接近）。Total steps 5×10⁶ 匹配 Ours/Decima。

---

### Stage 4：Throttled-HEFT augmented v5

代码已写好（`cpo_thermal_v2/baselines/throttled_heft.py`）。是 classical heuristic，不用训。Stage 5 eval matrix 里直接跑两行（hybrid + agent_only）。

---

### Stage 5：完整 eval matrix

跑 9 个 schedulers（10 行，因为 Throttled-HEFT 两个 mode）：

| # | Scheduler | Mode |
|---|---|---|
| 1 | HEFT | (auto) |
| 2 | Thermal-HEFT | (auto) |
| 3 | Round Robin | (auto) |
| 4 | Throttled-HEFT | hybrid |
| 5 | Throttled-HEFT | agent_only |
| 6 | Decima true | (REINFORCE) |
| 7 | HGATE-PPO | (PPO) |
| 8 | Ours-NoThermal | hybrid |
| 9 | Ours-auto_only | agent_only |
| 10 | Ours-hybrid | hybrid |

矩阵：
- **Main scaling**: 4 ambient × 5 N (9, 13, 17, 24, 33) × 500 ep = 10,000 ep / scheduler
- **Horizon scan**: 4 ambient × 4 H (20, 50, 100, 200) × 500 ep, N=17 = 8,000 ep / scheduler

总 ≈ 10 × 18,000 = 180,000 ep。CPU 评测可能 2-3 天。

**重要**：所有 scheduler 用**完全相同的 random seed pool**——paired-sample comparison 才有效。

---

### Stage 6：Paper revision

按 Stage 5 数据填表 + 改 narrative。Checklist：

- §3.2 line ~896：`10× hotter` → `10× more heat`
- §3.2：同步 `oe_active_power = 40 W` 描述
- §5.1.5 baseline policies：把 Throttled-HEFT 从 §6.5 limitations 移过来作 main baseline；加 Decima true 和 HGATE-PPO 的 architecture rationale
- §5.2 main results table：从 4 行扩到 10 行
- §5.3 two-channel decomposition → 重写为 **five-channel ablation chain**：
  ```
  HEFT → Throttled-HEFT       : reactive throttling 价值
  Throttled-HEFT → Decima     : RL + reward shaping 价值
  Decima → HGATE-PPO          : hetero edge typing 价值
  HGATE-PPO → Ours-NoThermal  : CPO-specific RC edge 价值
  Ours-NoThermal → auto_only  : real-time thermal obs 价值
  auto_only → hybrid          : anticipatory delay 价值
  ```
- §5.4 makespan trade-off：跟 Throttled-HEFT 比是 like-for-like
- §6 future work：删 HGATE-PPO 段（已做）
- §6.5 limitations：删 Throttled-HEFT 段（已做）；删 "no homogeneous-graph DRL comparison" 段（已做）
- Abstract：更新 headline 数字

---

## 4. 不要做的事（critical）

- ❌ **不要**用 Ours 的 HeteroEncoder 包一层 mask 当 Decima — 之前 `decima.py` / `decima_fair.py` 都这样错的
- ❌ **不要**在 HGATE-PPO 里加 RC edge attribute — 污染 controlled comparison
- ❌ **不要**在 Stage 0 sanity check 全 PASS 前进 Stage 1
- ❌ **不要**在 Stage 1 hybrid vs auto_only 数据可区分前出主表
- ❌ **不要**改 RC 矩阵 calibration（除非真要走 oe_active_power=15 那条路）— `generate_matrices.py` 物理是对的，bug 只在两个 scalar constants
- ❌ **不要**所有 scheduler 用不同 random seed — 必须 paired comparison

---

## 5. 文件 map

```
cpo_thermal_v2/
├── envs/
│   ├── cpo_thermal_env.py        ← Stage 0 修这里（两个 fix）
│   ├── rc_dynamics.py            ← 不动
│   └── reward_shaping.py         ← 不动
├── models/
│   ├── hetero_encoder.py         ← Ours 用
│   ├── cross_attention_actor.py  ← Ours 用
│   └── ...
├── baselines/
│   ├── heft.py                   ← 已就绪
│   ├── thermal_heft.py           ← 已就绪
│   ├── round_robin.py            ← 已就绪
│   ├── throttled_heft.py         ← 已就绪 (Stage 4)
│   ├── decima.py                 ← 旧 hack，删或重命名 .deprecated
│   ├── decima_fair.py            ← 旧错版，删或重命名 .deprecated
│   ├── decima_true.py            ← Stage 2 新写
│   └── hgate_ppo.py              ← Stage 3 新写
├── scripts/
│   ├── train.py                  ← Ours 用
│   ├── train_decima_true.py      ← Stage 2 新写
│   ├── train_hgate_ppo.py        ← Stage 3 新写
│   ├── sanity_check_env_fixed.py ← Stage 0 新写
│   └── evaluate.py               ← Stage 5 已有，要加 decima_true / hgate_ppo 注册
└── configs/
    ├── stage1_auto_only.yaml     ← 检查 temp_rise default 是否 override
    ├── stage2_hybrid.yaml        ← 同上
    └── ...
```

---

## 6. Stage 0 完成判定

四个 sanity check 全 PASS：

| Check | 期望 |
|---|---|
| A: 常数生效 | `env.temp_rise_per_ms_asic == 0.08`, `env.temp_rise_per_ms_oe == 0.18` |
| B: predicted_peak | OE 30ms dispatch from T uniform=65 → peak ∈ [67, 75]; plus rate calibration (ASIC from T=25 in [0.05, 0.12] K/ms; OE from T=70 in [0.10, 0.22] K/ms) |
| C: cooling fires | cooling_total > 50 ms/ep, P(cooling>0) > 70%（pre-fix ≈0 ms/ep）|
| D: 旧 ckpt hybrid vs auto_only | 行为可区分（旧 env 上几乎完全相同）|

四个 PASS = 可以开 Stage 1。任何 FAIL = 停下来 debug。

### Lessons learned (Stage 0)

- env fix 的责任只到 cooling fires，不负责让 greedy classical 存活
- HEFT/ThermalHEFT/RR 在 extreme 全部 fail (peak~91, trunc~95%) 是 scheduler-vs-cap 交互问题，不是 env 问题
- Check D (trained PPO) 才是 env 健康的判定
- Paper §5.1.5 备注: classical greedy 在 bounded idle budget 下根本无法利用 cooling，这是 RL 价值的 motivation 点

### Findings (Stage 3 HGATE-PPO)

- **HGATE-PPO**: 实现完毕 (HK-4.1 到 HK-4.5.8). 真瓶颈是 env.step
  (no thermal awareness → 触发 cooling 多), 不是 model. throughput 58 step/s
  on A100 是 fair baseline 数据.

---

## Known paper-code deviations (for Stage 6 paper revision)

1. Stage 1 training budget — paper §4.x 写 5×10⁶ steps,
   configs/stage1_auto_only.yaml 实际是 3×10⁶ steps。yaml 注释解释
   "cold stage ep_ret plateaued by step ~150k", 是有意的实验决定。
   Stage 6 处理方向 (默认 Option A): 改 paper 反映 yaml 实测, 加 footnote
   解释 plateau 现象。Option B (改 yaml 回 5M 重训) 不推荐 —— 实测已证
   cold 早 plateau, 多训没价值。

2. Stage 1 curriculum 切换点 — paper 写 cold→warm at 1×10⁶,
   stage1_auto_only.yaml 实际 4×10⁵。同上, Option A 推荐:
   改 paper 反映 yaml。

3. Stage 2 budget — paper 写 1.5×10⁶, yaml 1.5M, 一致 ✓

这三条记下不阻塞 Stage 0 / HK-1 commit, 也不阻塞 Stage 1 重训 (用
yaml 实际值跑就好)。Stage 6 paper revision 时一并处理。

---

## 7. 何时回报 chat 不继续推进

Claude Code 遇到下面情况时**停下来回报，不要硬推**：

1. Stage 0 Check B 的 OE peak 不在 [67, 75]，或 ASIC/OE rate 不在标定区间 — 可能 RC 矩阵跟新常数失配，超原诊断范围
2. Stage 1 重训完 hybrid vs auto_only 仍几乎一样（reward channel / dual critic 设计问题）
3. Stage 2 Decima REINFORCE 完全爆炸不收敛（需切 Decima++ 或重审超参）
4. Stage 3 HGATE-PPO 训出来 ≥ Ours-NoThermal（**真信号**，可能颠覆论文 main claim 之一，需重审 RC edge attribute 真实贡献，不要 cover up）
5. 出现 paper §3 物理描述跟 code 新行为冲突（除了已知 line 896 那句）

---

## 8. 启动 prompt（给 Claude Code 的）

把这份 HANDOFF.md 加进 Claude Code 项目根目录，然后第一句对话用：

```
读 HANDOFF.md。当前任务：执行 Stage 0（env fixes + sanity check）。
完成后跑 sanity check 把四个结果贴出来，等我确认再进 Stage 1。
不要进 Stage 1 之前的任何工作。
```

每个 stage 完成后让 Claude Code 暂停 + 报数据，你确认后再放下一阶段。这样比一次性放它跑完整链条安全得多。

> 通用 Claude 行为守则 (Think Before Coding / Simplicity / Surgical / Goal-Driven) 已搬到全局 `~/.claude/CLAUDE.md` —— 所有项目自动继承。
