"""
training/config_loader.py — YAML config loader with inherit + CLI override
==========================================================================

Three-step config resolution::

    1. load YAML file
    2. follow ``_inherit:`` chain, deep-merging each level
    3. apply CLI overrides via dotted keys (e.g. ``training.learning_rate``)

Usage
-----
::

    # In train.py:
    import argparse
    from cpo_thermal_v2.training import load_config, merge_cli_overrides

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[],
                        help="Override config (e.g. --override training.lr=1e-4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args.override)

CLI override syntax
-------------------
Each ``--override`` arg is a single ``key=value`` string with dotted
keys::

    --override env.action_mode=hybrid
    --override training.total_steps=1000000
    --override training.learning_rate=5e-5
    --override env.initial_temp_range=[40,65]

Values are parsed with ``yaml.safe_load`` so YAML literals work
(numbers, booleans, lists, null).  String values that look like YAML
literals must be quoted.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import yaml


# Default path resolved relative to the package root
_PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = str(_PKG_ROOT / "configs" / "default.yaml")


# =====================================================================
# YAML loading with _inherit chain
# =====================================================================
def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        # Try relative to cwd first; if not there, try relative to pkg root
        if path.exists():
            path = path.resolve()
        else:
            alt = _PKG_ROOT / path
            if alt.exists():
                path = alt.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def _resolve_inherit(
    cfg: Dict[str, Any],
    base_dir: Path,
    seen: List[Path],
) -> Dict[str, Any]:
    """Recursively follow ``_inherit:`` and merge child onto parent.

    Cycle detection raises ``ValueError`` (else infinite recursion would
    just look like a hang on the cluster).
    """
    inherit = cfg.pop("_inherit", None)
    if inherit is None:
        return cfg

    # Resolve parent path: support both bare names ("default") and full
    # paths ("configs/default.yaml" or absolute paths).
    parent_path = Path(inherit)
    if not parent_path.suffix:                     # bare name → assume yaml
        parent_path = parent_path.with_suffix(".yaml")
    if not parent_path.is_absolute() and not parent_path.exists():
        parent_path = base_dir / parent_path
    parent_path = parent_path.resolve()

    if parent_path in seen:
        raise ValueError(
            f"Cyclic _inherit detected: {' -> '.join(str(p) for p in seen)} -> {parent_path}"
        )

    parent = _load_yaml(parent_path)
    parent = _resolve_inherit(parent, parent_path.parent, seen + [parent_path])

    return _deep_merge(parent, cfg)


def _deep_merge(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``child`` on top of ``parent``.

    Lists in ``child`` REPLACE the parent list (no concat).  Dicts merge.
    Scalars in child override parent.

    The ``parent`` is not mutated; a new dict is returned.
    """
    result = copy.deepcopy(parent)
    for k, v in child.items():
        if (k in result and isinstance(result[k], dict)
                and isinstance(v, dict)):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(
    path: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    """Load a config file and resolve any ``_inherit:`` chain."""
    path = Path(path)
    if not path.is_absolute():
        # Same lookup logic as _load_yaml
        if not path.exists():
            alt = _PKG_ROOT / path
            if alt.exists():
                path = alt.resolve()
        else:
            path = path.resolve()
    cfg = _load_yaml(path)
    cfg = _resolve_inherit(cfg, path.parent, [path])
    return cfg


# =====================================================================
# CLI override parsing
# =====================================================================
def _parse_value(s: str) -> Any:
    """Interpret a CLI-supplied value string via YAML semantics.

    Examples
    --------
    >>> _parse_value("0.001")
    0.001
    >>> _parse_value("hybrid")
    'hybrid'
    >>> _parse_value("true")
    True
    >>> _parse_value("[1, 2, 3]")
    [1, 2, 3]
    >>> _parse_value("null")        # yields Python None

    Note
    ----
    PyYAML's strict spec requires a dot for scientific notation
    (``5.0e-5`` parses as float, ``5e-5`` parses as a string).  We
    handle the no-dot case explicitly so users aren't bitten on the CLI.
    """
    # First try plain Python float for strict scientific notation
    # (handles "5e-5", "1e6", "-2.5E+3", ... that YAML would reject).
    if _looks_like_scientific_float(s):
        try:
            return float(s)
        except ValueError:
            pass
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s                          # fall back to raw string


_SCI_RE = None
def _looks_like_scientific_float(s: str) -> bool:
    """True iff ``s`` matches a Python-style float literal with an
    exponent (``[+-]?\\d+(\\.\\d*)?[eE][+-]?\\d+``)."""
    global _SCI_RE
    if _SCI_RE is None:
        import re
        _SCI_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")
    return bool(_SCI_RE.match(s.strip()))


def _set_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Walk ``cfg`` along ``dotted_key`` and set the leaf.

    Creates intermediate dicts if missing.  Refuses to overwrite a
    non-dict intermediate (a clear error beats silently shadowing it).
    """
    parts = dotted_key.split(".")
    node = cfg
    for k in parts[:-1]:
        if k not in node:
            node[k] = {}
        elif not isinstance(node[k], dict):
            raise ValueError(
                f"Cannot set '{dotted_key}': "
                f"intermediate '{k}' is {type(node[k]).__name__}, not a dict"
            )
        node = node[k]
    node[parts[-1]] = value


def merge_cli_overrides(
    cfg: Dict[str, Any],
    overrides: Iterable[str],
) -> Dict[str, Any]:
    """Apply a list of ``key=value`` strings to ``cfg`` (in-place + return).

    Each entry must contain exactly one ``=``.  The key uses dotted
    notation; the value is parsed with YAML semantics.
    """
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(
                f"--override must be of the form key=value; got: {ov!r}"
            )
        key, _, raw_val = ov.partition("=")
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            raise ValueError(f"--override has empty key: {ov!r}")
        _set_dotted_key(cfg, key, _parse_value(raw_val))
    return cfg


# =====================================================================
# Convenience: round-trip the resolved config to disk for reproducibility
# =====================================================================
def save_resolved_config(
    cfg: Dict[str, Any],
    out_path: Union[str, Path],
) -> None:
    """Dump the fully-resolved config to YAML for posterity.

    Each training run should write its resolved config alongside the
    checkpoint, so a future re-run can ``load_config`` it directly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


# =====================================================================
# Self-test
# =====================================================================
def _self_test() -> None:
    """Smoke-test the loader against the bundled configs."""
    print("Loading default.yaml ...")
    default = load_config(DEFAULT_CONFIG_PATH)
    assert default["env"]["action_mode"] == "auto_only"
    assert default["env"]["num_nodes"] == 17
    assert default["model"]["hidden"] == 128
    print(f"  ✅ default loaded: {len(default)} top-level sections")

    print("Loading stage1_auto_only.yaml (inherits default) ...")
    s1 = load_config(_PKG_ROOT / "configs" / "stage1_auto_only.yaml")
    assert s1["env"]["action_mode"] == "auto_only"
    assert s1["env"]["num_nodes"] == 17                  # inherited
    assert s1["logging"]["run_name"] == "stage1_auto_only_N17"
    print(f"  ✅ stage1 inherited: run_name={s1['logging']['run_name']}")

    print("Loading stage2_hybrid.yaml (inherits default) ...")
    s2 = load_config(_PKG_ROOT / "configs" / "stage2_hybrid.yaml")
    assert s2["env"]["action_mode"] == "hybrid"
    assert s2["training"]["learning_rate"] == 1.0e-4    # overridden
    assert s2["training"]["gamma"] == 0.99              # inherited
    assert s2["training"]["warm_start_path"] is not None
    # Curriculum schedule was overridden — must be 2 stages, not 3
    assert len(s2["curriculum"]["schedule"]) == 2
    assert s2["curriculum"]["schedule"][0]["name"] == "warm"
    print(f"  ✅ stage2 inherited & overrode: lr={s2['training']['learning_rate']}, "
          f"curriculum={[s['name'] for s in s2['curriculum']['schedule']]}")

    print("Testing CLI overrides ...")
    cfg = load_config(_PKG_ROOT / "configs" / "stage1_auto_only.yaml")
    cfg = merge_cli_overrides(cfg, [
        "training.learning_rate=5e-5",
        "env.num_nodes=33",
        "training.total_steps=100",
        "env.initial_temp_range=[60, 75]",
        "logging.run_name=test_override",
    ])
    assert cfg["training"]["learning_rate"] == 5e-5
    assert cfg["env"]["num_nodes"] == 33
    assert cfg["training"]["total_steps"] == 100
    assert cfg["env"]["initial_temp_range"] == [60, 75]
    assert cfg["logging"]["run_name"] == "test_override"
    print(f"  ✅ CLI overrides applied: lr=5e-5, num_nodes=33, "
          f"initial_temp={cfg['env']['initial_temp_range']}")

    print("Testing dotted-key creation of new intermediate ...")
    cfg2 = load_config(DEFAULT_CONFIG_PATH)
    cfg2 = merge_cli_overrides(cfg2, ["custom.section.value=42"])
    assert cfg2["custom"]["section"]["value"] == 42
    print(f"  ✅ created nested key: custom.section.value=42")

    print("Testing error case (empty key) ...")
    try:
        merge_cli_overrides({}, ["=value"])
    except ValueError as e:
        print(f"  ✅ raised ValueError on empty key: {e}")

    print("Testing error case (no =) ...")
    try:
        merge_cli_overrides({}, ["no_equals_sign"])
    except ValueError as e:
        print(f"  ✅ raised ValueError on missing =: {e}")

    print("\nAll config_loader tests passed ✓")


if __name__ == "__main__":
    _self_test()
