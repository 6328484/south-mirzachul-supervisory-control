"""Optimized staged rule (Sects. 3.3-3.4): a transparent 24-parameter supervisor that
is trained like the neural supervisor and serves as the fair benchmark of the network.
Same two-layer structure and local FF/P layer; the supervisor uses no level measurements:
  branch j:   f~_j = first-order lag of the supply ratio f with time constant tau_j (updated hourly)
              u_j* = f~_j ** p_j,            p_j = 1.5 exp(x_j),  tau_j = 24 / (1 + exp(4 - y_j)) h
  check k:    h_set,k = lo_k + (hi_k - lo_k) * sigma(c_k + 5 d_k (1 - f) + c0_k)
theta = (x_1..5, y_1..5, c_1..7, d_1..7). theta = 0 reproduces the fixed rule up to a lag of 0.4 h."""
import numpy as np
from supervisory import LocalLayer, _move_u


def n_params(env):
    return 2 * len(env.majors) + 2 * len(env.ci)


def decode(theta, nb=5, ng=7):
    x, y = theta[:nb], theta[nb:2 * nb]
    c, d = theta[2 * nb:2 * nb + ng], theta[2 * nb + ng:]
    return 1.5 * np.exp(x), 24.0 / (1.0 + np.exp(-(y - 4.0))), c, d


def staged_policy(env, theta):
    pi = LocalLayer(env)
    nb, ng = len(env.majors), len(env.ci)
    p, tau, c, d = decode(theta, nb, ng)
    lo, hi = pi.set_lo[env.ci], pi.set_hi[env.ci]
    base_h = (pi.up - 0.10)[env.ci]
    z0 = np.log((base_h - lo) / (hi - base_h))
    h_default = pi.up - 0.10
    alpha = 1.0 - np.exp(-1.0 / tau)
    state = {"ff": None}

    def ctrl(o, a, u):
        f = o[0] + 1.0
        ff = np.full(nb, f) if state["ff"] is None else state["ff"] + alpha * (f - state["ff"])
        state["ff"] = ff
        h_set = h_default.copy()
        h_set[env.ci] = lo + (hi - lo) / (1.0 + np.exp(-(c + d * 5.0 * (1.0 - f) + z0)))
        u = _move_u(u, np.clip(ff, 0, 1) ** p)
        return pi(o, a, h_set, u), u
    return ctrl
