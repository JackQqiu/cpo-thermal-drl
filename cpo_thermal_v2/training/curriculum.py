"""
training/curriculum.py — 3-stage curriculum scheduler
=====================================================

Reads the ``curriculum.schedule`` list from the config and exposes a
single :class:`CurriculumScheduler` object that:

1. tracks global step count (across all parallel envs);
2. determines which stage we're in based on ``until_step`` thresholds;
3. on stage transitions, calls ``env.set_curriculum_stage(...)`` on each
   wrapped env to update ``initial_temp_range`` and ``max_dag_size``.

Stage definition (from config)
------------------------------
Each schedule entry is a dict with::

    name:               str                — for logging only
    until_step:         int | None         — global step threshold
                                             (None ⇒ stage runs to the end)
    initial_temp_range: [float, float]
    max_dag_size:       int | None

Stages are applied in list order; the first stage whose ``until_step``
is greater than the current global step (or whose ``until_step`` is
None) is selected.

Why global step (not env episodes)
----------------------------------
Different curriculum stages have different episode lengths (cold has
small DAGs ⇒ short episodes, hot has large DAGs ⇒ long episodes).  If
we counted episodes, hot would get under-trained relative to its
difficulty.  Counting global env-steps keeps wall-clock cost roughly
constant across stages.

Reference
---------
Plan §5.  Defaults baked into ``configs/default.yaml`` (cold→warm→hot).

Example
-------
::

    sched = CurriculumScheduler(cfg["curriculum"])
    for step in range(total_steps):
        if sched.update(step):                     # True ⇒ stage just changed
            for env in vec_env.envs:
                env.set_curriculum_stage(**sched.current_kwargs)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CurriculumStage:
    """A single stage's parameters; immutable after parse."""
    name:               str
    until_step:         Optional[int]                # None ⇒ open-ended
    initial_temp_range: Tuple[float, float]
    max_dag_size:       Optional[int]


def _parse_schedule(raw: List[Dict[str, Any]]) -> List[CurriculumStage]:
    """Validate and freeze the schedule list from the YAML config."""
    if not raw:
        raise ValueError("curriculum.schedule must be a non-empty list")
    stages: List[CurriculumStage] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"stage[{i}] must be a dict, got {type(entry).__name__}")
        for required in ("name", "initial_temp_range"):
            if required not in entry:
                raise ValueError(f"stage[{i}] missing key {required!r}")
        # ``max_dag_size`` is optional; default = no limit
        # ``until_step`` is optional; default = None (open-ended)
        rng = entry["initial_temp_range"]
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise ValueError(
                f"stage[{i}] initial_temp_range must be [lo, hi], got {rng!r}"
            )
        stages.append(CurriculumStage(
            name              = str(entry["name"]),
            until_step        = entry.get("until_step", None),
            initial_temp_range= (float(rng[0]), float(rng[1])),
            max_dag_size      = entry.get("max_dag_size", None),
        ))

    # Validate ordering: until_step must be strictly increasing (None goes last)
    seen_open_ended = False
    last_threshold  = -1
    for st in stages:
        if seen_open_ended:
            raise ValueError(
                "Open-ended stage (until_step=None) must be the last entry"
            )
        if st.until_step is None:
            seen_open_ended = True
        else:
            if st.until_step <= last_threshold:
                raise ValueError(
                    f"stage thresholds must be strictly increasing; "
                    f"got {st.until_step} after {last_threshold}"
                )
            last_threshold = st.until_step

    return stages


class CurriculumScheduler:
    """Step-driven schedule.  See module docstring for usage."""

    def __init__(self, curriculum_cfg: Dict[str, Any]):
        if not curriculum_cfg.get("enabled", True):
            # Disabled curriculum: a single stage that uses whatever the
            # env was constructed with.  We still build a one-entry
            # schedule for uniform code paths.
            self._enabled = False
            self.stages = [CurriculumStage(
                name="static", until_step=None,
                initial_temp_range=(25.0, 25.0), max_dag_size=None,
            )]
        else:
            self._enabled = True
            self.stages = _parse_schedule(curriculum_cfg["schedule"])
        self._current_idx: int = 0
        self._global_step: int = 0

    # -----------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def current(self) -> CurriculumStage:
        return self.stages[self._current_idx]

    @property
    def current_kwargs(self) -> Dict[str, Any]:
        """Kwargs to pass to ``env.set_curriculum_stage(...)``."""
        st = self.current
        return {
            "initial_temp_range": st.initial_temp_range,
            "max_dag_size":       st.max_dag_size,
            "stage_name":         st.name,
        }

    @property
    def global_step(self) -> int:
        return self._global_step

    # -----------------------------------------------------------------
    def update(self, global_step: int) -> bool:
        """Advance the global step; return True iff stage just changed.

        ``global_step`` should be the cumulative count of env-steps across
        ALL parallel envs (not per-env episodes).  For T=256 / N=16,
        each rollout adds 4096 to this counter.

        Stage selection rule: pick the first stage whose ``until_step``
        is None or strictly greater than ``global_step``.  Once selected,
        we never go back.
        """
        self._global_step = int(global_step)
        if not self._enabled:
            return False

        # Find the stage we should be in NOW
        for i, st in enumerate(self.stages):
            if st.until_step is None or self._global_step < st.until_step:
                if i != self._current_idx:
                    # Transition!  Update index and signal caller.
                    prev = self._current_idx
                    self._current_idx = i
                    return True
                return False
        # Shouldn't happen (last stage is always open-ended), but defensively:
        return False

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(enabled={self._enabled}, "
            f"step={self._global_step}, current={self.current.name!r}, "
            f"stage_idx={self._current_idx}/{len(self.stages)})"
        )


# =====================================================================
# Self-test
# =====================================================================
def _self_test():
    """Verify schedule parsing, stage transitions, edge cases."""
    print("Running curriculum.py self-tests...\n")

    cfg = {
        "enabled": True,
        "schedule": [
            {"name": "cold", "until_step": 100,
             "initial_temp_range": [25, 40], "max_dag_size": 5},
            {"name": "warm", "until_step": 200,
             "initial_temp_range": [40, 65], "max_dag_size": 15},
            {"name": "hot",  "until_step": None,
             "initial_temp_range": [60, 75], "max_dag_size": None},
        ],
    }
    sch = CurriculumScheduler(cfg)
    assert sch.current.name == "cold"
    assert sch.current_kwargs["max_dag_size"] == 5
    print(f"  ✅ initial: {sch}")

    # Step within cold
    changed = sch.update(50)
    assert not changed and sch.current.name == "cold"
    # Cross into warm
    changed = sch.update(100)
    assert changed and sch.current.name == "warm", f"got {sch.current.name}"
    print(f"  ✅ at step 100: transitioned to {sch.current.name}")
    # Cross into hot
    changed = sch.update(200)
    assert changed and sch.current.name == "hot"
    print(f"  ✅ at step 200: transitioned to {sch.current.name}")
    # Stay in hot
    changed = sch.update(10_000_000)
    assert not changed and sch.current.name == "hot"
    print(f"  ✅ stays in hot indefinitely")

    # Disabled curriculum
    sch2 = CurriculumScheduler({"enabled": False, "schedule": []})
    assert not sch2.enabled
    assert sch2.current.name == "static"
    assert not sch2.update(1000)
    print(f"  ✅ disabled curriculum: {sch2.current.name!r}")

    # Validation: out-of-order thresholds
    bad_cfg = {"enabled": True, "schedule": [
        {"name": "a", "until_step": 200, "initial_temp_range": [25, 40]},
        {"name": "b", "until_step": 100, "initial_temp_range": [40, 65]},
    ]}
    try:
        CurriculumScheduler(bad_cfg)
        assert False, "should have raised on out-of-order thresholds"
    except ValueError as e:
        print(f"  ✅ raises on out-of-order: {e}")

    # Validation: open-ended not last
    bad_cfg = {"enabled": True, "schedule": [
        {"name": "a", "until_step": None, "initial_temp_range": [25, 40]},
        {"name": "b", "until_step": 100,  "initial_temp_range": [40, 65]},
    ]}
    try:
        CurriculumScheduler(bad_cfg)
        assert False, "should have raised on misplaced open-ended"
    except ValueError as e:
        print(f"  ✅ raises on misplaced open-ended: {e}")

    # Validation: missing required field
    bad_cfg = {"enabled": True, "schedule": [
        {"name": "a"},   # no initial_temp_range
    ]}
    try:
        CurriculumScheduler(bad_cfg)
        assert False, "should have raised on missing field"
    except ValueError as e:
        print(f"  ✅ raises on missing initial_temp_range: {e}")

    print("\nAll curriculum.py tests passed ✓")


if __name__ == "__main__":
    _self_test()
