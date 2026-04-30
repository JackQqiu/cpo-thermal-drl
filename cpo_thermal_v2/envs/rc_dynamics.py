"""
rc_dynamics.py — v2 RC thermal state-space simulator
=====================================================

Drop-in replacement for the v1 module.  New in v2:

  * Optional explicit ``A / B / D`` injection — for unit tests, ablations,
    and synthetic-matrix experiments.
  * Graceful numba fallback (still fast on plain numpy if numba is absent).
  * Matrix directory is configurable in three ways, in priority order:

        1. ``matrix_dir=`` keyword argument
        2. ``$CPO_MATRIX_DIR`` environment variable
        3. autodiscovery: ``../data/thermal_matrics/`` then
           ``./data/thermal_matrics/``

  * ``snapshot()`` / ``restore()`` helpers for the env's lookahead-then-undo
    auto-delay logic — letting the env predict whether a placement would
    overflow ``T_pen`` *without* committing to the prediction.

State-space model::

    T(t+1) = A · T(t) + B · P(t) + D · T_amb
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------
# numba fallback
# ---------------------------------------------------------------------
try:
    from numba import njit

    @njit(fastmath=True, cache=True)
    def _fast_step(A, B, Tamb_term, T, P):
        return A @ T + B @ P + Tamb_term

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover — server has numba
    NUMBA_AVAILABLE = False

    def _fast_step(A, B, Tamb_term, T, P):
        """Pure numpy fallback. ~5-10× slower than the numba version."""
        return A @ T + B @ P + Tamb_term


# ---------------------------------------------------------------------
# Matrix discovery
# ---------------------------------------------------------------------
def _autodiscover_matrix_dir(num_nodes: Optional[int] = None) -> Optional[str]:
    """Locate the directory holding ``matrix_A.npy``, ``matrix_B.npy``, ``matrix_D.npy``.

    Search order
    ------------
    1. ``$CPO_MATRIX_DIR``  (explicit override; expected to point directly
       at the directory containing the .npy files)
    2. Size-aware: ``<root>/N{num_nodes}/``  for each candidate ``<root>``
       (this is what the parametric ``generate_matrices.py`` writes)
    3. Legacy flat layout: ``<root>/`` directly
       (matches the user's pre-refactor 9-node setup)

    Each ``<root>`` is checked in two locations:
        * sibling of this file: ``../data/thermal_matrics/``
        * cwd:                 ``./data/thermal_matrics/``
    """
    here = os.path.dirname(os.path.abspath(__file__))

    # Priority 1: env-var override
    env_dir = os.environ.get("CPO_MATRIX_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "matrix_A.npy")):
        return env_dir

    roots = [
        os.path.join(os.path.dirname(here), "data", "thermal_matrics"),
        os.path.join(os.getcwd(),           "data", "thermal_matrics"),
    ]

    # Priority 2: size-aware subdir
    if num_nodes is not None:
        for r in roots:
            sized = os.path.join(r, f"N{num_nodes}")
            if os.path.exists(os.path.join(sized, "matrix_A.npy")):
                return sized

    # Priority 3: legacy flat layout (backward compat for N=9 only)
    for r in roots:
        if os.path.exists(os.path.join(r, "matrix_A.npy")):
            return r

    return None


# ---------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------
class RCThermalDynamics:
    """RC state-space thermal model:  T(t+1) = A·T(t) + B·P(t) + D·T_amb.

    Parameters
    ----------
    num_nodes
        Number of thermal nodes (usually 9 = 1 ASIC + 8 OE).
    dt
        Simulation time step in seconds (typically 1e-3 = 1 ms).
    ambient_temp
        Ambient temperature in °C.
    matrix_dir
        Folder containing ``matrix_A.npy``, ``matrix_B.npy``, ``matrix_D.npy``.
        If ``None``, autodiscovery kicks in.
    A, B, D
        Direct injection of the state-space matrices.  Bypasses disk loading.
        Useful for unit tests and synthetic-matrix experiments.
        Either pass *all three* or *none*.
    """

    def __init__(
        self,
        num_nodes: int,
        dt: float,
        ambient_temp: float = 25.0,
        matrix_dir: Optional[str] = None,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ):
        self.num_nodes = num_nodes
        self.dt = dt
        self.ambient_temp = ambient_temp
        self.temperatures = np.full(num_nodes, ambient_temp, dtype=np.float32)

        # ---- Resolve A / B / D ----
        injected = sum(M is not None for M in (A, B, D))
        if injected == 3:
            self.A = np.asarray(A, dtype=np.float32)
            self.B = np.asarray(B, dtype=np.float32)
            self.D = np.asarray(D, dtype=np.float32)
        elif injected == 0:
            md = matrix_dir or _autodiscover_matrix_dir(num_nodes)
            if md is None:
                raise FileNotFoundError(
                    "Cannot locate RC matrices.  Pass matrix_dir=, set "
                    "$CPO_MATRIX_DIR, or place matrices under "
                    "data/thermal_matrics/.  Alternatively, pass explicit "
                    "A=, B=, D=."
                )
            self.A = np.load(os.path.join(md, "matrix_A.npy")).astype(np.float32)
            self.B = np.load(os.path.join(md, "matrix_B.npy")).astype(np.float32)
            self.D = np.load(os.path.join(md, "matrix_D.npy")).astype(np.float32)
        else:
            raise ValueError("Pass either all of A, B, D or none of them.")

        # Validate shapes
        for name, M in (("A", self.A), ("B", self.B), ("D", self.D)):
            if M.shape != (num_nodes, num_nodes):
                raise ValueError(
                    f"{name} must be ({num_nodes}, {num_nodes}), got {M.shape}"
                )

        # Pre-compute the constant ambient term D · T_amb · 1
        T_amb_vec = np.full(num_nodes, ambient_temp, dtype=np.float32)
        self.Tamb_term = self.D @ T_amb_vec

    # -----------------------------------------------------------------
    # Forward / reset
    # -----------------------------------------------------------------
    def reset(self, initial_temperatures: Optional[np.ndarray] = None) -> np.ndarray:
        if initial_temperatures is not None:
            self.temperatures = np.asarray(
                initial_temperatures, dtype=np.float32
            ).copy()
        else:
            self.temperatures = np.full(
                self.num_nodes, self.ambient_temp, dtype=np.float32
            )
        return self.temperatures

    def step(self, power_array: np.ndarray) -> np.ndarray:
        """Advance one ``dt`` and return the new temperature vector.

        Internal state is clamped to ``[-50, 250]°C`` to prevent
        catastrophic numerical divergence from a single bad step
        propagating forever (an unbounded ``temperatures`` would
        produce ``inf`` powers via the leakage exponential, leading to
        NaN logits downstream).  Under healthy operation this clamp is
        never near the boundary; it's pure safety.
        """
        P = np.asarray(power_array, dtype=np.float32)
        self.temperatures = _fast_step(
            self.A, self.B, self.Tamb_term, self.temperatures, P
        )
        # Clamp to physically meaningful bounds so a divergent single
        # step can't poison subsequent steps.
        self.temperatures = np.clip(self.temperatures, -50.0, 250.0)
        return self.temperatures

    # -----------------------------------------------------------------
    # Lookahead helpers (used by the env's auto-delay logic)
    # -----------------------------------------------------------------
    def snapshot(self) -> np.ndarray:
        """Save the current temperature vector for later restore."""
        return self.temperatures.copy()

    def restore(self, snap: np.ndarray) -> None:
        """Restore from a snapshot taken by :meth:`snapshot`."""
        self.temperatures = np.asarray(snap, dtype=np.float32).copy()
