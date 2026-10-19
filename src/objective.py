"""Training objective J (Sect. 3.4), to be minimised, for one 72-h training episode:
  J = CV(group delivery ratios, mean over 24-72 h)      — inequity
    + 0.5 * max(0, f - ratio of the worst-served group)  — shortfall of the worst-served group
    + 2.0 * S                                            — severity of level violations:
          S = sum_{t,k} (rise above the upper limit + drop below the lower limit) * dt / 72 h  [m]
    + 0.002 * total travel of the 7 check gates [m] + 0.02 * total travel of the 5 branch regulators [fraction]
If the canal dries out, J = 10 + 10 * (fraction of the 72 h that remained).
"""
import numpy as np
from metrics import level_bounds, gate_metrics

W_SHORT, W_LEVEL, W_TRAVEL, W_REG = 0.5, 2.0, 0.002, 0.02


def severity(env, rec):
    _, lo, up = level_bounds(env)
    hu = np.vstack([env.h_init[env.up_nodes], rec["hu"]])
    v = np.clip(hu - up - 1e-3, 0, None) + np.clip(lo - hu, 0, None)
    return float(v.sum() * (env.dt / 3600) / 72.0)


def objective(env, rec, f, t_from=24 * 3600):
    if "failed_at_h" in rec:
        return 10.0 + (72 - rec["failed_at_h"]) / 72 * 10
    sel = rec["t"] >= t_from
    s = rec["shares"][sel].mean(axis=0)
    G = gate_metrics(env, rec)
    return (s.std() / s.mean() + W_SHORT * max(0.0, f - s.min()) + W_LEVEL * severity(env, rec)
            + W_TRAVEL * G["check_travel_m"] + W_REG * G["reg_travel_frac"])
