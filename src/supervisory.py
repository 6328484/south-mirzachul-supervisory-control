"""
Two-layer control: local FF/P layer + supervisory layer (Sect. 3.3)
===================================================================
Local layer (the same for all strategies): the seven check gates and the tail gate hold the depth
upstream of them at the set-point h_set. Every hour, the target discharge of gate k combines a
feedforward of the measured water balance of the pool upstream of the gate with a proportional
correction of the level error (FF/P):
    Q_k = Q_in,k - sum(offtake withdrawals in the pool) + A_k * (h_k - h_set,k) / Tc,   Tc = 4 h
and the opening follows from the gate equation with the measured head difference:
    a_k = Q_k / (mu * W_k * sqrt(2 g dY_k)).
A gate moves by at most 0.30 m per hour and does not move if the required change is below 1 cm.
The integral term is disabled (ti = 1e9): the model and the controller share the gate equation, so the
feedforward leaves no steady-state error.

Supervisory layer, every hour:
  - h_set: set-points of the 7 check gates within [h_min(Qmin) + 0.10, h_max(Qmax) - 0.05];
           the tail-gate set-point is fixed at h_max - 0.10
  - u:     relative openings (0..1) of the 5 major branch regulators; a regulator moves by at most
           0.15 per hour and only in steps of at least 0.05.
Strategies:
  current practice — h_set = h_max - 0.10, u = 1 (branches fully open)
  rule             — h_set = h_max - 0.10, u = f^1.5
  neural           — h_set and u chosen by an MLP 11 -> 8 (tanh) -> 12
                     (inputs: f, the 9 depths, the tail outflow)
"""
import numpy as np
from environment import Env
from metrics import level_bounds

N_HID = 8
DA_MAX, DU_MAX = 0.30, 0.15
DEAD_A, DEAD_U = 0.01, 0.05          # regulators move only in steps of at least 5 % (operator practice)


class LocalLayer:
    """Local layer: feedforward (water balance) + level feedback.
    Target discharge of each gate k (water balance of the pool upstream of it):
        Q_k = Q_in,k - sum(offtake withdrawals in the pool)          (feedforward)
              + A_k * (h_u,k - h_set,k) / Tc  [+ integral, disabled]  (feedback)
    Q_in,k is the measured discharge entering the pool from the upstream gate (the head discharge for
    the first pool), so the travel time of the wave is taken into account automatically.
    A_k is the water-surface area of the pool upstream of the gate, Tc the correction time.
    The gate opening follows from the gate equation with the measured upstream and downstream levels:
        a_k = Q_k / (mu * W_k * sqrt(2 g dY_k)).
    All quantities are measurable by SCADA or known from the passport."""
    tc, ti = 4.0, 1e9           # hours (ti = 1e9: integral disabled; the feedforward removes the residual error)

    def __init__(self, env):
        self.env = env
        c = env.c
        _, self.lo, self.up = level_bounds(env)
        self.set_lo = self.lo + 0.10
        self.set_hi = self.up - 0.05
        self.idx = np.append(env.ci, len(env.a0) - 1)
        nodes = env.up_nodes                          # upstream nodes of the 9 gates
        self.seg_k = nodes[self.idx]                  # gate segment (N-1 for the tail gate)
        # pool areas: from the previous controlled gate to this gate
        bounds = np.concatenate([[0], nodes])
        self.pool_area = np.zeros(len(nodes))
        Tw0 = c.b + 2 * c.m * env.h0
        dxs = np.where(c.is_struct, 0.0, np.diff(c.x))
        segA = 0.5 * (Tw0[:-1] + Tw0[1:]) * dxs
        for k in range(len(nodes)):
            self.pool_area[k] = segA[bounds[k]:nodes[k]].sum()
        # the pool of the fully open PK1035 is merged with the next pool
        for k in range(len(nodes) - 1):
            if c.open_struct[k]:
                self.pool_area[k + 1] += self.pool_area[k]
        self.integ = np.zeros(len(self.idx))

    def hu_from_obs(self, o):
        return self.env.hu0 * (1.0 + o[1:10])

    def __call__(self, o, a, h_set, u):
        env, c = self.env, self.env.c
        h, Q = env.last_h, env.last_Q
        q_act = c.actual_offtake(h, env.qoff(u))
        cum = np.concatenate([[0], np.cumsum(q_act)])
        idx = self.idx
        e = (h[env.up_nodes] - h_set)[idx]
        self.integ = np.clip(self.integ + e / self.ti, -1.0, 1.0)          # m, anti-windup
        corr = self.pool_area[idx] * (e + self.integ) / (self.tc * 3600)
        # pool boundaries: downstream node of the previous controlled gate
        start = np.concatenate([[0], self.seg_k[:-1] + 1])
        Q_in = Q[start]
        end_seg = self.seg_k.copy(); end_seg[-1] = len(q_act)
        Qt = np.maximum(Q_in - (cum[end_seg] - cum[start]) + corr, 0.0)
        a_new = a.copy()
        W = c.W[idx]
        u_n = env.up_nodes[idx]
        dY = np.empty(len(idx))
        dY[:-1] = (c.z[u_n[:-1]] + h[u_n[:-1]]) - (c.z[u_n[:-1] + 1] + h[u_n[:-1] + 1])
        dY[-1] = h[-1]
        a_tgt = Qt / (c.mu * W * np.sqrt(2 * 9.81 * np.maximum(dY, 0.02)))
        a_tgt = np.clip(a_tgt, 0.05, env.a_max[idx])
        step = np.clip(a_tgt - a[idx], -DA_MAX, DA_MAX)
        step = np.where(np.abs(step) < DEAD_A, 0.0, step)
        a_new[idx] = a[idx] + step
        return a_new


def _move_u(u, u_tgt):
    du = np.clip(u_tgt - u, -DU_MAX, DU_MAX)
    du = np.where(np.abs(du) < DEAD_U, 0.0, du)
    return np.clip(u + du, 0.0, 1.0)


def practice_policy(env):
    """Current practice: set-points h_max - 0.10, branches fully open."""
    pi = LocalLayer(env); h_set = pi.up - 0.10

    def ctrl(o, a, u):
        return pi(o, a, h_set, u), u
    return ctrl


def rule_policy(env, power=1.5):
    """Rule: as current practice, but the five branch regulators are set to u = f^1.5."""
    pi = LocalLayer(env); h_set = pi.up - 0.10

    def ctrl(o, a, u):
        f = o[0] + 1.0
        u = _move_u(u, np.full(len(u), f ** power))
        return pi(o, a, h_set, u), u
    return ctrl


def n_params(env):
    n_out = len(env.ci) + len(env.majors)
    return env.n_obs * N_HID + N_HID + N_HID * n_out + n_out


def neural_policy(env, theta):
    """Neural supervisor: MLP 11 -> 8 (tanh) -> 12 with parameters theta (204 values)."""
    pi = LocalLayer(env)
    n_out = len(env.ci) + len(env.majors)
    i = 0
    W1 = theta[i:i + env.n_obs * N_HID].reshape(N_HID, env.n_obs); i += env.n_obs * N_HID
    b1 = theta[i:i + N_HID]; i += N_HID
    W2 = theta[i:i + N_HID * n_out].reshape(n_out, N_HID); i += N_HID * n_out
    b2 = theta[i:i + n_out]
    ng = len(env.ci)
    # at theta = 0: h_set = h_max - 0.10 and u = f^1.5, i.e. the rule (training starts from the rule)
    lo, hi = pi.set_lo[env.ci], pi.set_hi[env.ci]
    base_h = (pi.up - 0.10)[env.ci]
    z0_h = np.log((base_h - lo) / (hi - base_h))
    h_default = pi.up - 0.10

    def ctrl(o, a, u):
        z = W2 @ np.tanh(W1 @ (o * 5.0) + b1) + b2
        h_set = h_default.copy()
        h_set[env.ci] = lo + (hi - lo) / (1.0 + np.exp(-(z[:ng] + z0_h)))
        f = o[0] + 1.0
        u_tgt = 1.0 / (1.0 + np.exp(-(z[ng:] + np.log(f ** 1.5 / (1 - f ** 1.5 + 1e-6)))))
        u = _move_u(u, u_tgt)
        return pi(o, a, h_set, u), u
    return ctrl
