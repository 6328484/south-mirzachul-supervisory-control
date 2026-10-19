"""
Level-safety and gate-effort indicators (Sects. 3.2 and 3.6)
============================================================
Level limits (no official operating limits are published; taken from the passport):
  upper: depth at Qmax (calibrated initial state)
  lower: passport depth at Qmin (196 m3/s); below it the offtakes cannot draw their design discharge
  drawdown rate: <= 0.15 m/h (Wu et al. 2021; an external benchmark, not an official limit of this canal)
Control points: upstream of the 9 check structures (8 internal + tail).
Gate effort: check gates in metres; branch and offtake regulators as fractions of full opening
(their opening heights are not known).
"""
import numpy as np
from gated_model import PASSPORT

DRAWDOWN_MAX = 0.15          # m/h
MOVE_TOL_A = 0.01            # m: a larger change counts as one gate operation
MOVE_TOL_U = 0.01            # fraction of full opening


def level_bounds(env):
    c = env.c
    reaches = list(c.struct_reach) + [9]
    h_max = np.asarray(PASSPORT["h"])[reaches]
    h_min = np.asarray(PASSPORT["h_min"])[reaches]
    # the calibrated initial depth (equal to the passport depth at Qmax) is used as the upper limit
    return env.hu0.copy(), h_min, h_max


def level_metrics(env, rec, dt=None):
    """rec['hu']: (n_t, 9) depths upstream of the structures. Returns a dictionary."""
    dt = dt or env.dt
    up, lo, _ = level_bounds(env)
    hu = np.vstack([env.h_init[env.up_nodes], rec["hu"]])     # including the initial state
    over = np.clip(hu - up - 1e-3, 0, None)          # 1 mm tolerance
    under = np.clip(lo - hu, 0, None)
    viol = (over > 0) | (under > 0)
    rate = -np.diff(hu, axis=0) / (dt / 3600)        # drawdown rate, m/h (positive = falling)
    return dict(
        n_points_violated=int(viol.any(axis=0).sum()),
        violation_hours=float(viol[1:].sum() * dt / 3600),                 # point-hours
        max_violation_hours=float(viol[1:].sum(axis=0).max() * dt / 3600),  # at the worst point
        max_under_m=float(under.max()),
        max_over_m=float(over.max()),
        max_drop_m=float((hu[0] - hu).max()),
        max_drawdown_rate=float(rate.max()),
        rate_violation_hours=float((rate > DRAWDOWN_MAX).sum() * dt / 3600),
    )


def travel_metrics(series, tol, dt_hours=1.0):
    """series: (n_t, k) control settings. Total travel, number of operations, maximum rate."""
    if series.size == 0:
        return dict(travel=0.0, moves=0, max_rate=0.0)
    d = np.abs(np.diff(series, axis=0))
    return dict(travel=float(d.sum()), moves=int((d > tol).sum()), max_rate=float(d.max() / dt_hours))


def gate_metrics(env, rec, ctrl_every=3600):
    """rec['a']: (n_t, 9) gate openings, rec['u']: (n_t, k) regulator openings.
    Records are made every dt; the control changes every ctrl_every."""
    step = int(ctrl_every / env.dt)
    a = np.vstack([env.a0, rec["a"]])[::step][:, env.ci]
    u0 = np.ones(rec["u"].shape[1]) if rec["u"].ndim == 2 and rec["u"].shape[1] else np.zeros(0)
    u = np.vstack([u0, rec["u"]])[::step] if u0.size else np.zeros((0, 0))
    g = travel_metrics(a, MOVE_TOL_A)
    r = travel_metrics(u, MOVE_TOL_U)
    return dict(check_travel_m=g["travel"], check_moves=g["moves"], check_max_rate=g["max_rate"],
                reg_travel_frac=r["travel"], reg_moves=r["moves"], reg_max_rate=r["max_rate"],
                n_controlled=len(env.ci) + (u.shape[1] if u.size else 0))
