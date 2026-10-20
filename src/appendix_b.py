"""Appendix B of the article: properties of the local FF/P layer, and the numbers quoted in Appendix A.
Output: results/appendix_b.json (Table B.1 and the checks below), printed summary.
  - Table B.1: pool length L, surface area F_k at 300 m3/s, outflow Q_k^out, travel times of small waves
    t_down = sum dx/(c+V), t_up = sum dx/(c-V), steady level error e_inf (Eq. B.6) for eps_mu = -0.1, +0.1,
    and the simulated decay of the level error per hour after a set-point step of -0.10 m at t = 48 h
    (f = 0.85, fixed rule; Eq. B.2 predicts 1 - dt_c/T_c = 0.75)
  - Levin-May bound 2 cos(d pi / (2d + 1)) and the largest root modulus of z^(d+1) - z^d + kappa (Eqs. B.4-B.5)
  - Courant and Froude numbers of the grid at 300 m3/s (Appendix A)
Usage:  python src/appendix_b.py      (about 2 min)"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from environment import Env
from supervisory import LocalLayer, _move_u
from gated_model import g
from paths import result_file

TC, DTC, KAPPA, T_STEP = 4 * 3600.0, 3600.0, 0.25, 48


def pools(e, L):
    c, h0, Q0 = e.c, e.h0, e.Q0
    idx, un = L.idx, e.up_nodes
    start = np.concatenate([[0], un[idx[:-1]] + 1])
    dxs = np.diff(c.x)
    rows = []
    for k, (s, ui) in enumerate(zip(start, un[idx])):
        segs = np.arange(s, ui)
        segs = segs[~c.is_struct[segs]]
        hs = 0.5 * (h0[segs] + h0[segs + 1])
        A = (c.b[segs] + c.m[segs] * hs) * hs
        B = c.b[segs] + 2 * c.m[segs] * hs
        V = 0.5 * (Q0[segs] + Q0[segs + 1]) / A
        cel = np.sqrt(g * A / B)
        F, Qout = L.pool_area[idx[k]], Q0[ui]
        einf = {str(eps): float(-eps / (1 + eps) * TC * Qout / F) for eps in (-0.1, 0.1)}
        rows.append(dict(gate=c.sname[idx[k]], L_km=float((c.x[ui] - c.x[s]) / 1000), F_km2=float(F / 1e6),
                         Q_out=float(Qout), t_down_min=float((dxs[segs] / (cel + V)).sum() / 60),
                         t_up_min=float((dxs[segs] / (cel - V)).sum() / 60), e_inf_m=einf))
    return rows


def step_decay(e, kk):
    """Decay factor per hour of h - h_set at gate kk after a -0.10 m set-point step (log-linear fit, 1-8 h)."""
    L = LocalLayer(e)
    base = L.up - 0.10
    clock = {"h": 0}

    def ctrl(o, a, u):
        t = clock["h"]; clock["h"] += 1
        hs = base.copy()
        if t >= T_STEP:
            hs[kk] -= 0.10
        u = _move_u(u, np.full(len(u), (o[0] + 1.0) ** 1.5))
        return L(o, a, hs, u), u

    rec = e.run(ctrl, f=0.85, T=(T_STEP + 16) * 3600)
    hu = rec["hu"][:, kk]
    err = np.array([hu[2 * m - 1] - (base[kk] - 0.10) for m in range(T_STEP, T_STEP + 13)])
    r = err / err[0]
    m = np.arange(1, 9)
    ok = r[1:9] > 0.02
    return float(np.exp(np.polyfit(m[ok], np.log(r[1:9][ok]), 1)[0]))


def main():
    e = Env()
    L = LocalLayer(e)
    table = pools(e, L)
    for row, kk in zip(table, list(e.ci) + [len(e.up_nodes) - 1]):
        row["decay_per_h"] = step_decay(e, kk)
    delay = []
    for d in range(8):
        p = np.zeros(d + 2); p[0], p[1] = 1, -1; p[-1] += KAPPA
        delay.append(dict(d=d, max_root=float(np.max(np.abs(np.roots(p)))),
                          bound=float(2 * np.cos(d * np.pi / (2 * d + 1)))))
    c, h0, Q0 = e.c, e.h0, e.Q0
    seg = ~c.is_struct
    hs = 0.5 * (h0[:-1] + h0[1:])
    A = (c.b[:-1] + c.m[:-1] * hs) * hs
    B = c.b[:-1] + 2 * c.m[:-1] * hs
    V = 0.5 * (Q0[:-1] + Q0[1:]) / A
    cel = np.sqrt(g * A / B)
    cr = ((V + cel) * e.dt / np.diff(c.x))[seg]
    out = dict(table_B1=table, sampled_prediction=1 - DTC / TC, delay_stability=delay,
               courant=[float(cr.min()), float(cr.max())], froude_max=float((V / cel)[seg].max()))
    with open(result_file("appendix_b.json"), "w") as f:
        json.dump(out, f, indent=1)
    for r in table:
        print(f"{r['gate']:40s} L={r['L_km']:6.2f} F={r['F_km2']:.3f} Q={r['Q_out']:6.1f} "
              f"t={r['t_down_min']:3.0f}/{r['t_up_min']:3.0f} min  e_inf={r['e_inf_m']['-0.1']:.2f}/"
              f"{r['e_inf_m']['0.1']:.2f} m  decay={r['decay_per_h']:.2f}")
    print("Courant", np.round(out["courant"], 1), "Froude max", round(out["froude_max"], 3))
    for d in delay:
        print(f"d={d['d']}: max|root|={d['max_root']:.3f}  bound={d['bound']:.3f}")


if __name__ == "__main__":
    main()
