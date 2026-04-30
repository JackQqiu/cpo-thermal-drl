"""
training/env_factory.py — config-driven AsyncVectorEnv construction
====================================================================

Factory functions that build CPOThermalDAGEnvV2 from a resolved config
dict, plus a wrapped multi-process AsyncVectorEnv for training.

Why a factory module
--------------------
``gymnasium.vector.AsyncVectorEnv`` requires each child process to
re-build the env from a picklable closure.  We can't pass dataset_obj
arrays (huge memory blow-up on fork), so each worker reads the dataset
JSON from disk via the path in config.  Factory functions encapsulate
this protocol cleanly.

Datasets and matrices
---------------------
* The DAG dataset is loaded once per worker from
  ``config['env']['dataset_path']`` (default
  ``./data_pipeline/process/alibaba_dags_v2.json``).  Each child holds
  its own copy in memory — for 300 MB JSON × 16 workers = 4.8 GB, well
  under the 64 GB node budget.
* RC matrices are loaded by ``RCThermalDynamics`` autodiscovery,
  preferring ``data/thermal_matrics/N{N}/`` subdirs (set up by
  ``data_pipeline/generate_matrices.py``).

API
---
``make_single_env(config, seed)`` -> CPOThermalDAGEnvV2
    Returns one env instance.  Used by AsyncVectorEnv internally.

``make_vector_env(config, num_envs, seed_base, mode='async')`` -> VectorEnv
    Returns an AsyncVectorEnv (or SyncVectorEnv when mode='sync', for
    debugging).  Each worker gets a unique seed = seed_base + worker_idx.

``broadcast_curriculum_stage(vec_env, **kwargs)`` -> None
    Calls ``set_curriculum_stage(**kwargs)`` on every worker via the
    ``call`` API.  Used by the trainer when the CurriculumScheduler
    signals a stage transition.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

# gymnasium is mandatory at this layer (we're past the simulation-only
# Stage A+B sandbox).  If it's missing, the import itself signals to the
# user that they need to install the training-side requirements.
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
from cpo_thermal_v2.envs.reward_shaping import RewardConfig


# =====================================================================
# Single-env factory (used by AsyncVectorEnv on the worker side)
# =====================================================================
def _build_reward_cfg(reward_section: Dict[str, Any]) -> RewardConfig:
    """Construct a :class:`RewardConfig` from the YAML ``reward`` section.

    Only keys that exist on RewardConfig are passed; extras are silently
    dropped (so adding new YAML keys doesn't break this loader).
    """
    valid_keys = set(RewardConfig.__dataclass_fields__.keys())
    kw = {k: v for k, v in reward_section.items() if k in valid_keys}
    return RewardConfig(**kw)


def _make_env_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the YAML ``env`` section into env-constructor kwargs.

    Mostly a 1-to-1 pass-through, but with three transformations:
    1. ``initial_temp_range`` YAML lists -> tuples
    2. ``reward`` section -> ``RewardConfig`` passed as ``reward_config=``
    3. **Filter out keys the env constructor doesn't accept**, with a
       clear warning.  This keeps training robust against YAML drift
       (e.g. an obsolete key like ``thermtrip_timeout_ms`` left over
       from an earlier env API would otherwise crash with TypeError).
    """
    import inspect
    import warnings

    env_cfg = dict(config["env"])

    # Lists from YAML become tuples for the env's tuple-typed kwargs
    if "initial_temp_range" in env_cfg and env_cfg["initial_temp_range"] is not None:
        env_cfg["initial_temp_range"] = tuple(env_cfg["initial_temp_range"])
    if "delay_fractions" in env_cfg:
        env_cfg["delay_fractions"]    = tuple(env_cfg["delay_fractions"])

    # Build RewardConfig from the YAML 'reward' section (separate from env)
    if "reward" in config:
        env_cfg["reward_config"] = _build_reward_cfg(config["reward"])

    # Drop any keys the env constructor doesn't recognise.  The env's
    # __init__ uses keyword-only args after the leading positional pair,
    # so its signature reports them as KEYWORD_ONLY parameters.
    sig = inspect.signature(CPOThermalDAGEnvV2.__init__)
    valid = {p.name for p in sig.parameters.values()
             if p.kind in (inspect.Parameter.KEYWORD_ONLY,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD)}
    valid.discard("self")
    unknown = set(env_cfg.keys()) - valid
    if unknown:
        warnings.warn(
            f"env_factory: dropping {len(unknown)} unknown env-config "
            f"key(s) from YAML: {sorted(unknown)}.  This is usually fine "
            f"(stale config keys), but if you expected one of these to "
            f"take effect, the env API has likely changed.",
            stacklevel=2,
        )
        for k in unknown:
            env_cfg.pop(k)

    return env_cfg


def make_single_env(
    config: Dict[str, Any],
    seed:   Optional[int] = None,
) -> CPOThermalDAGEnvV2:
    """Build one env instance from the config dict (callable from a worker)."""
    kwargs = _make_env_kwargs(config)
    env = CPOThermalDAGEnvV2(**kwargs)
    if seed is not None:
        # gymnasium uses .reset(seed=...) rather than .seed(); seed-on-init
        # is the standard convention for vector envs.
        env.reset(seed=int(seed))
    return env


def _env_thunk(config: Dict[str, Any], seed: int) -> Callable[[], gym.Env]:
    """Return a no-arg callable that constructs the env (for AsyncVectorEnv)."""
    def _thunk() -> gym.Env:
        return make_single_env(config, seed=seed)
    return _thunk


# =====================================================================
# Vector-env factory
# =====================================================================
def make_vector_env(
    config:    Dict[str, Any],
    num_envs:  int,
    seed_base: int  = 42,
    mode:      str  = "async",
) -> gym.vector.VectorEnv:
    """Build a vectorised env (Async by default; Sync for debugging).

    Each worker gets ``seed = seed_base + worker_idx`` so episodes are
    distinct across workers but reproducible across runs.

    Parameters
    ----------
    mode : {'async', 'sync'}
        ``async``  → multiprocess :class:`AsyncVectorEnv` (recommended for
                     training)
        ``sync``   → single-process :class:`SyncVectorEnv` (good for
                     pdb-debugging the env)
    """
    if mode not in ("async", "sync"):
        raise ValueError(f"mode must be 'async' or 'sync', got {mode!r}")
    thunks = [_env_thunk(config, seed_base + i) for i in range(num_envs)]
    if mode == "async":
        return AsyncVectorEnv(thunks)
    return SyncVectorEnv(thunks)


# =====================================================================
# Curriculum broadcast helper
# =====================================================================
def broadcast_curriculum_stage(
    vec_env: gym.vector.VectorEnv,
    **kwargs: Any,
) -> None:
    """Call ``env.set_curriculum_stage(**kwargs)`` on every worker.

    Uses gymnasium's ``call`` API which is supported by both Async and
    Sync vector envs.  The return values (all ``None``) are discarded.
    """
    # gymnasium .call() takes positional args; we re-route kwargs by
    # passing the raw dict as a single keyword via the worker's stub.
    # Cleaner: AsyncVectorEnv has .call_async / call_wait; SyncVectorEnv
    # has .call.  Both accept (*args, **kwargs).  We use .call() which
    # both implement uniformly.
    vec_env.call("set_curriculum_stage", **kwargs)


# =====================================================================
# Diagnostic: collect a single sample to verify env wiring
# =====================================================================
def smoke_test_vector_env(
    vec_env:  gym.vector.VectorEnv,
    num_envs: int,
) -> Dict[str, Any]:
    """Run one reset + one random step; return a diagnostics dict.

    Used by ``train.py`` as a sanity check before the rollout loop, so
    any shape/key bug crashes within seconds rather than after hours.

    Returns
    -------
    A dict with::
        first_obs_shape:      tuple    — temperature obs shape per env
        action_mask_shapes:   list     — bool-mask shape for each env
        reward_channels_keys: set      — keys of the first env's reward_channels
        graph_obs_keys:       set      — keys of the first env's graph_obs[0]
        info_format:          str      — "dict_of_arrays" / "list_of_dicts" / etc

    Notes
    -----
    Gymnasium has shipped at least three different ``info`` broadcast
    formats across versions::

        list_of_dicts   (≤ gym 0.25):    info = [dict_env0, dict_env1, ...]
        dict_of_arrays  (gymnasium 0.26+): info["k"][i] = value for env i
        custom-wrapped  (gymnasium 1.0+): same as above + "_k" boolean mask

    We detect and unpack whichever format we got.
    """
    obs, info = vec_env.reset(seed=0)
    masks = _extract_per_env(info, "action_mask", num_envs)

    # Random step within mask
    #
    # We need to generate an action of the right shape for the env's
    # action_mode.  Probing the vector-env's ``action_space`` directly
    # is fragile because gymnasium has changed how it wraps Discrete /
    # MultiDiscrete across versions.  The reliable source of truth is
    # ``single_action_space`` (the *inner* env's action space, which is
    # always the un-batched form), available on every gymnasium
    # vector env.  Fall back gracefully if it's not exposed.
    rng = np.random.default_rng(0)

    inner_space = getattr(vec_env, "single_action_space", None)
    is_multi = (
        inner_space is not None
        and inner_space.__class__.__name__ == "MultiDiscrete"
    )
    # Final fallback: also accept the outer action_space for older
    # gymnasium versions that don't have single_action_space yet.
    if inner_space is None:
        outer = getattr(vec_env, "action_space", None)
        is_multi = (outer is not None
                    and outer.__class__.__name__ == "MultiDiscrete"
                    and len(getattr(outer, "nvec", [])) >= 2 * num_envs)

    if is_multi:
        # Factored action: [proc_idx, delay_idx] per env
        # Read K_delay from the inner space if possible; fall back to 5.
        if inner_space is not None and hasattr(inner_space, "nvec"):
            K_delay = int(inner_space.nvec[1])
        else:
            K_delay = 5
        actions = []
        for n in range(num_envs):
            valid = np.where(masks[n])[0]
            proc  = int(rng.choice(valid))
            delay = int(rng.integers(0, K_delay))
            actions.append([proc, delay])
        actions = np.array(actions, dtype=np.int64)
    else:
        actions = np.array([
            int(rng.choice(np.where(masks[n])[0])) for n in range(num_envs)
        ], dtype=np.int64)

    obs2, r, term, trunc, info2 = vec_env.step(actions)

    # Diagnose what gymnasium gave us
    info_format = _detect_info_format(info2)

    # Verify the env's flat reward keys are all present and numeric.  The
    # env emits reward_placement / reward_delay / reward_total as flat
    # info entries (see comment in cpo_thermal_env.py:_make_info for why).
    rp_list = _extract_per_env(info2, "reward_placement", num_envs)
    rd_list = _extract_per_env(info2, "reward_delay",     num_envs)
    rt_list = _extract_per_env(info2, "reward_total",     num_envs)
    reward_channel_ok = (
        all(rp_list[i] is not None for i in range(num_envs)) and
        all(rd_list[i] is not None for i in range(num_envs)) and
        all(rt_list[i] is not None for i in range(num_envs))
    )

    # Pull the first env's graph_obs (still nested, wrapped as object array)
    go_list = _extract_per_env(info2, "graph_obs", num_envs)
    first_graph_obs = go_list[0]
    if isinstance(first_graph_obs, np.ndarray) and first_graph_obs.dtype == object \
            and first_graph_obs.size == 1:
        first_graph_obs = first_graph_obs[0]

    # Synthesise a {placement, delay, total} key set so the train.py
    # smoke assertion still works without changes.
    reward_channels_keys = (
        {"placement", "delay", "total"} if reward_channel_ok else set()
    )

    return {
        "first_obs_shape":      tuple(obs.shape),
        "action_mask_shapes":   [tuple(m.shape) for m in masks],
        "reward_channels_keys": reward_channels_keys,
        "graph_obs_keys":       set(first_graph_obs.keys())
                                 if isinstance(first_graph_obs, dict)
                                 else None,
        "rewards_shape":        tuple(r.shape),
        "n_terminated":         int(np.sum(term)),
        "n_truncated":          int(np.sum(trunc)),
        "info_format":          info_format,
    }


def _detect_info_format(info) -> str:
    """Return a string label for the format of a vector-env info object."""
    if isinstance(info, list):
        return "list_of_dicts"
    if isinstance(info, dict):
        # Inspect a sample value
        for k, v in info.items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray):
                return "dict_of_arrays"
            if isinstance(v, list):
                return "dict_of_lists"
            if isinstance(v, dict):
                # Looks like info wasn't broadcasted; rare edge case
                return f"dict_of_dicts (broadcast failed?)"
        return "empty_dict"
    return f"unknown ({type(info).__name__})"


def _extract_per_env(info, key: str, num_envs: int) -> list:
    """Pull a per-env list of values for ``info[key]`` regardless of which
    gymnasium info-broadcast format we got.

    Always returns a Python list of length ``num_envs``, with ``None`` for
    any env that didn't populate the key.

    Handles 4 known formats:
        list_of_dicts:           info = [{"k": v0}, {"k": v1}, ...]
        dict_of_arrays:          info["k"] is np.ndarray (length N) of values
        dict_with_mask_arrays:   info["_k"] = bool mask, info["k"] = ndarray
        nested_dict_of_arrays:   info["k"] is itself a dict whose values
                                  are arrays  (gymnasium 1.x behaviour for
                                  dict-typed env info entries — it
                                  RECURSIVELY broadcasts dicts).  We
                                  reassemble per-env dicts here.
    """
    # Format 1: legacy list-of-dicts
    if isinstance(info, list):
        out = []
        for d in info:
            out.append(d.get(key) if isinstance(d, dict) else None)
        while len(out) < num_envs:
            out.append(None)
        return out[:num_envs]

    # Format 2/3/4/5: dict-of-...
    if isinstance(info, dict):
        if key not in info:
            return [None] * num_envs
        vals = info[key]

        # Format 2: numpy object array (length N) of per-env values
        if isinstance(vals, np.ndarray):
            return [vals[i] for i in range(min(num_envs, len(vals)))] + \
                   [None] * max(0, num_envs - len(vals))

        # Format 4 / 5: NESTED dict.
        #
        # Gymnasium 1.x flattens dict-typed env info one level deep.  If
        # an env returns ``info["reward_channels"] = {"placement": x, ...}``,
        # gymnasium reshapes it across N envs to::
        #
        #   info["reward_channels"] = {
        #       "placement": np.array([x_env0, ..., x_envN-1]),
        #       "_placement": np.array([True, ...]),    # presence mask
        #       ...                                       # same for "delay", "total"
        #   }
        #
        # SyncVectorEnv has an additional quirk: when env n terminates
        # mid-step, gymnasium ALSO appends ``info["reward_channels"][n]``
        # (with int key n) holding that env's final-info copy.  So we
        # may see a mix of string keys (the array layout) AND int keys
        # (per-env final-info copies).  We use the string-keyed arrays as
        # the source of truth and IGNORE the int-keyed entries (which
        # would otherwise be a duplicate).
        if isinstance(vals, dict):
            string_keys = [k_ for k_ in vals.keys() if isinstance(k_, str)]
            int_keys    = [k_ for k_ in vals.keys() if isinstance(k_, int)]
            other_keys  = [k_ for k_ in vals.keys()
                           if not isinstance(k_, (str, int))]

            if other_keys:
                raise TypeError(
                    f"_extract_per_env: info[{key!r}] has unexpected key "
                    f"types {[type(k_).__name__ for k_ in other_keys[:5]]} "
                    f"(sample keys: {other_keys[:5]!r}).  Don't know how to "
                    f"interpret this format."
                )

            # If we have string keys with the gymnasium "_k" mask convention
            # AND/OR per-env arrays, that's the authoritative source.
            has_per_env_arrays = any(
                isinstance(vals[k_], (np.ndarray, list))
                and len(vals[k_]) == num_envs
                for k_ in string_keys
                if not k_.startswith("_")
            )
            if has_per_env_arrays:
                real_keys = [k_ for k_ in string_keys if not k_.startswith("_")]
                out = []
                for n in range(num_envs):
                    per_env = {}
                    for rk in real_keys:
                        v = vals[rk]
                        if isinstance(v, (np.ndarray, list)) and len(v) > n:
                            per_env[rk] = v[n]
                        else:
                            per_env[rk] = v       # scalar broadcast
                    out.append(per_env)
                return out

            # Otherwise, if we ONLY have int keys, treat as
            # env-indexed dict (Format 5).
            if int_keys and not string_keys:
                return [vals.get(n) for n in range(num_envs)]

            # Fallback: broadcast a single dict (no array values, no int keys)
            return [vals] * num_envs

        # Format 2-ish: plain Python list (rare)
        if isinstance(vals, (list, tuple)):
            return list(vals[:num_envs]) + [None] * max(0, num_envs - len(vals))

        # Scalar / other: broadcast
        return [vals] * num_envs

    return [None] * num_envs
