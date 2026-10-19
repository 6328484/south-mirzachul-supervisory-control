"""
Calibrated canal and delivery ratios of the offtake groups
==========================================================
make_canal   — builds the Saint-Venant model of the canal and calibrates the gate openings to the
               passport depths at Qmax (Sect. 3.1).
reach_shares — delivery ratios of the nine offtake groups (Sect. 3.2): the offtakes of reaches 2..9
               and the outflow through the tail structure.
"""
import numpy as np
from gated_model import GatedCanal


def make_canal(dx=1000.0, n_scale=1.0):
    """The nominal canal is calibrated to the passport (gate openings and effective widths).
    If n_scale != 1 (model uncertainty), the calibration is kept at the nominal n, the roughness is then
    multiplied by n_scale, and the initial state is the Qmax steady state of this canal at the nominal
    gate openings. Returns: canal, nominal state (h, Q), initial state (h_init, Q_init)."""
    c = GatedCanal(dx=dx)
    c.open_struct[5] = True          # PK1035: negative head difference in the calibration, fully open
    h, Q = c.calibrate()
    c.offtake_mode = "orifice"
    h_init, Q_init = h.copy(), Q.copy()
    if n_scale != 1.0:
        c.n = c.n * n_scale
        h_init, Q_init = c.steady(Qin=300.0, a=c.a, h=h.copy(), Q=Q.copy(), dt=1800.0, n_steps=2000)
    return c, h, Q, h_init, Q_init


def reach_shares(c, h, Q, h0, Q0, qoff=None):
    """Delivery ratios (relative to the initial Qmax state) of 9 groups: the offtakes of reaches 2..9
    and the outflow through the tail structure (reach 10). Seepage and other distributed losses are
    excluded (they are not deliveries); reach 1 has no offtakes.
    qoff: current offtake setting."""
    xm = 0.5 * (c.x[1:] + c.x[:-1])
    qoff = c.q_off if qoff is None else qoff
    q0 = c.actual_offtake(h0, c.q_off - c.seepage)
    q1 = c.actual_offtake(h, qoff - c.seepage)
    out = []
    for i in range(1, 9):
        sel = (xm > c.edges[i]) & (xm <= c.edges[i + 1] + 1)
        out.append(q1[sel].sum() / q0[sel].sum())
    out.append(Q[-1] / Q0[-1])
    return np.array(out)
