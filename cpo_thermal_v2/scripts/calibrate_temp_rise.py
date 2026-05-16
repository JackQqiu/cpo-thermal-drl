"""
calibrate_temp_rise.py
======================

Empirically measure the actual per-ms temperature rise rate from the
RC matrices, by running long single-proc dispatches at known starting
temps and computing per-ms slopes.

The env was originally configured with:
    temp_rise_per_ms_asic = 0.5
    temp_rise_per_ms_oe   = 0.1
But diag_simulate_peak_internals showed a 30 ms ASIC dispatch at
T_pre=75 produces NEGATIVE rise (T drops to 74). The matrices and the
documented constants disagree.

This script measures the ACTUAL behaviour by:
  1. For each of {ASIC, OE} target type and several starting temps
     {25, 40, 55, 70, 80}, run a 200 ms dispatch.
  2. Record T[target] per ms.
  3. Find the steady-state T (where dT/dt → 0).
  4. Find the per-ms rise rate near each starting temp (slope).

Results give:
  - The actual asymptotic T at full power (the true "T_steady_asic")
  - The realistic per-ms rise rate at each start point
  - Whether the matrix calibration is internally consistent

Use this to set ``temp_rise_per_ms_*`` values that ACTUALLY match
the matrices, so est_dT in graph_obs becomes meaningful.

Usage:
    PYTHONPATH=. python cpo_thermal_v2/scripts/calibrate_temp_rise.py
"""
from __future__ import annotations
import numpy as np


def measure(env, target_node: int, T_start_others: float, T_start_target: float,
            n_ms: int = 200):
    """Run single-proc dispatch for n_ms steps, return T[target] per step."""
    env.thermal_engine.temperatures[:] = T_start_others
    env.thermal_engine.temperatures[target_node] = T_start_target

    Ts = [float(env.thermal_engine.temperatures[target_node])]
    for _ in range(n_ms):
        P = env._compute_power(target_node, 100.0)
        env.thermal_engine.step(P)
        Ts.append(float(env.thermal_engine.temperatures[target_node]))
    return np.asarray(Ts)


def main():
    from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
    env = CPOThermalDAGEnvV2(
        num_nodes=17, action_mode="auto_only",
        initial_temp_range=(60.0, 75.0),
    )
    env.reset(seed=12345)

    print(f"\n{'='*72}")
    print("Calibration: actual per-ms temp rise rate")
    print(f"{'='*72}")
    print(f"  configured: temp_rise_per_ms_asic = "
          f"{getattr(env, 'temp_rise_per_ms_asic', 'n/a')}")
    print(f"  configured: temp_rise_per_ms_oe   = "
          f"{getattr(env, 'temp_rise_per_ms_oe', 'n/a')}")
    print()

    for proc_label, target in [("ASIC (proc 0)", 0), ("OE (proc 5)", 5)]:
        print(f"\n--- {proc_label} ---")
        print(f"  T_start | T_after_30ms | ΔT_30ms | rate_first_10ms (°C/ms) | rate_steady (°C/ms)")
        print(f"  --------|--------------|---------|-------------------------|--------------------")
        for T0 in [25.0, 40.0, 55.0, 70.0, 80.0]:
            Ts = measure(env, target_node=target,
                         T_start_others=T0,
                         T_start_target=T0,
                         n_ms=200)
            T_after_30 = Ts[30]
            dT_30 = T_after_30 - T0
            # rate_first_10 = avg slope from t=0 to t=10
            rate_first_10 = (Ts[10] - Ts[0]) / 10.0
            # rate_steady = slope from t=190 to t=200
            rate_steady = (Ts[200] - Ts[190]) / 10.0
            print(f"  {T0:6.1f}  |  {T_after_30:9.2f}   | "
                  f"{dT_30:+6.2f}  |        {rate_first_10:+7.4f}          |       "
                  f"{rate_steady:+7.4f}")

        # Asymptote: T at t=200 (mostly steady)
        print(f"  Asymptote (T at t=200): from T_start=25 → "
              f"{Ts[200]:.2f} °C  (= equilibrium for {proc_label})")
        # Idealised "T_steady" if rate_first_10 was the true rate:
        # T(t) = T_steady - (T_steady - T0) * exp(-t/τ)
        # but this is rough; just report empirical asymptote.

    print()
    print(f"{'='*72}")
    print("Implications")
    print(f"{'='*72}")
    print(f"  • If 'rate_first_10' values for ASIC are << 0.5 °C/ms (the")
    print(f"    configured temp_rise_per_ms_asic), the matrices and config")
    print(f"    are inconsistent.  est_dT in graph_obs is overestimating.")
    print(f"  • The 'asymptote' value tells you the equilibrium T at full ASIC")
    print(f"    power — if it's e.g. 73 °C, then no single ASIC dispatch can")
    print(f"    push T over 80 °C from a cold start.  Truncations come from")
    print(f"    other physics: cross-coupling, very long dispatches, or")
    print(f"    starting at T close to asymptote already.")
    print(f"  • Recommended: set temp_rise_per_ms_asic to the rate_first_10")
    print(f"    value at T0=70 (typical scheduling state), so est_dT predicts")
    print(f"    the actual per-ms-of-exec contribution accurately.")


if __name__ == "__main__":
    main()