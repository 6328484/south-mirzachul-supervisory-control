"""
Full-control benchmark (FCB): a theoretical reference (Sect. 3.5)
=================================================================
The FCB is NOT a description of real automated gates of the canal, and it is not an operating option for
the existing canal. It answers the question: if all 58 offtakes listed in the passport were remotely
controlled and the scenario were known in advance, how much more equity than with the 13-gate neural
supervisor would be possible?

Structure: the neural policy with the best training score after 100 nominal generations (seed 0) plus
open-loop corrections optimised separately for each test scenario:
  - a multiplier m_i,b in (0, 2) for each of the 58 offtakes in each of three 24-h blocks:
        u_i = clip(u_i^neural * m_i,b, 0, 1)   (u^neural = 1 for the offtakes other than the five branches)
  - an offset of up to +-0.3 m on each of the 7 check-gate set-points in each block
    (set-points stay within the allowed range)
With all corrections zero the FCB is the neural policy itself; the optimisation starts there and keeps the
best candidate, so the FCB scores at least as well as that policy on its objective.
Objective: delivery and level safety with the same weights as in training, but WITHOUT the gate-movement
penalty; gate effort is measured separately. The local FF/P layer and the actuator limits are the same as
for all other strategies.
Output: results/policies/fcb_<scenario>.npy, results/training_logs/fcb_<scenario>.csv
Usage:  python src/benchmark.py [scenario names]   (default: all four test scenarios, about 15 min each)
"""
import sys
import time
import warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from environment import Env, MAJOR_MIN_Q
from supervisory import LocalLayer, _move_u, N_HID
from objective import severity, W_SHORT, W_LEVEL
from scenarios import tag
from paths import policy_file, log_file

N_BLOCK = 3
SET_RANGE = 0.3
NN_SEED = 0
# The stored optimisation runs used the solver without step halving (see GatedCanal.max_halvings);
# keep 0 to reproduce them exactly. The evaluation of the benchmark (evaluate.py) uses the default.
SOLVER_HALVINGS = 0


class FullEnv(Env):
    """Environment in which all offtakes are controlled (Env.majors := all 58 offtakes)."""
    def __init__(self, n_scale=1.0):
        super().__init__(n_scale=n_scale)
        c = self.c
        self.major_pos = [i for i, o in enumerate(c.offtake_list) if o[3] >= MAJOR_MIN_Q]
        self.majors = list(c.offtake_list)
        self.q_minor = c.seepage.copy()                 # only the distributed losses remain uncontrolled


def n_params(env):
    return len(env.majors) * N_BLOCK + len(env.ci) * N_BLOCK


def make_fcb(env, nn_theta, phi):
    ng, n_off = len(env.ci), len(env.majors)
    n_out = ng + len(env.major_pos)
    i = 0
    W1 = nn_theta[i:i + env.n_obs * N_HID].reshape(N_HID, env.n_obs); i += env.n_obs * N_HID
    b1 = nn_theta[i:i + N_HID]; i += N_HID
    W2 = nn_theta[i:i + N_HID * n_out].reshape(n_out, N_HID); i += N_HID * n_out
    b2 = nn_theta[i:i + n_out]
    Z = phi[:n_off * N_BLOCK].reshape(N_BLOCK, n_off)
    Y = phi[n_off * N_BLOCK:].reshape(N_BLOCK, ng)
    pi = LocalLayer(env)
    lo, hi = pi.set_lo[env.ci], pi.set_hi[env.ci]
    base_h = (pi.up - 0.10)[env.ci]
    z0_h = np.log((base_h - lo) / (hi - base_h))
    h_default = pi.up - 0.10
    state = dict(hour=0)

    def ctrl(o, a, u):
        b = min(state["hour"] // 24, N_BLOCK - 1); state["hour"] += 1
        z = W2 @ np.tanh(W1 @ (o * 5.0) + b1) + b2
        h_set = h_default.copy()
        h_set[env.ci] = lo + (hi - lo) / (1.0 + np.exp(-(z[:ng] + z0_h)))
        h_set[env.ci] = np.clip(h_set[env.ci] + SET_RANGE * np.tanh(Y[b]), lo, hi)
        f = o[0] + 1.0
        u_major = 1.0 / (1.0 + np.exp(-(z[ng:] + np.log(f ** 1.5 / (1 - f ** 1.5 + 1e-6)))))
        base = np.ones(n_off); base[env.major_pos] = u_major
        u_tgt = np.clip(base * 2.0 / (1.0 + np.exp(-Z[b])), 0.0, 1.0)
        u = _move_u(u, u_tgt)
        return pi(o, a, h_set, u), u
    return ctrl


def objective_fcb(env, rec, f, t_from=24 * 3600):
    """As objective.objective, but without the gate-movement penalty."""
    if "failed_at_h" in rec:
        return 10.0 + (72 - rec["failed_at_h"]) / 72 * 10
    sel = rec["t"] >= t_from
    s = rec["shares"][sel].mean(axis=0)
    return s.std() / s.mean() + W_SHORT * max(0.0, f - s.min()) + W_LEVEL * severity(env, rec)


_env, _kw, _f, _nn = None, None, None, None


def _init(n_scale, kw, f):
    global _env, _kw, _f, _nn
    warnings.filterwarnings("ignore")
    _env, _kw, _f = FullEnv(n_scale=n_scale), kw, f
    _env.c.max_halvings = SOLVER_HALVINGS
    _nn = np.load(policy_file(f"nominal_g100_seed{NN_SEED}.npy"))


def _eval(phi):
    return objective_fcb(_env, _env.run(make_fcb(_env, _nn, phi), **_kw), _f)


def optimise(name, kw, f_ref, n_scale=1.0, gens=80, half_pop=12, sigma=0.15, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    env = FullEnv(n_scale=n_scale)
    P = n_params(env); phi = np.zeros(P)
    m = np.zeros(P); v = np.zeros(P); b1, b2 = 0.9, 0.999
    best = (np.inf, phi.copy())
    log = open(log_file(f"fcb_{tag(name)}.csv"), "w"); log.write("gen,score_mean,score_center,best\n")
    with Pool(2, initializer=_init, initargs=(n_scale, kw, f_ref)) as pool:
        for g in range(1, gens + 1):
            t0 = time.time()
            eps = rng.standard_normal((half_pop, P))
            cands = [phi + sigma * e for e in eps] + [phi - sigma * e for e in eps] + [phi]
            sc = np.array(pool.map(_eval, cands)); sc_c = sc[-1]; sc = sc[:-1]
            if sc_c < best[0]:
                best = (sc_c, phi.copy())
            ib = int(np.argmin(sc))
            if sc[ib] < best[0]:
                best = (sc[ib], cands[ib].copy())
            ranks = np.empty(2 * half_pop); ranks[np.argsort(sc)] = np.arange(2 * half_pop)
            util = -(ranks / (2 * half_pop - 1) - 0.5)
            grad = -((util[:half_pop] - util[half_pop:]) @ eps) / (half_pop * sigma)
            m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad**2
            phi = phi - lr * (m / (1 - b1**g)) / (np.sqrt(v / (1 - b2**g)) + 1e-8)
            log.write(f"{g},{sc.mean():.4f},{sc_c:.4f},{best[0]:.4f}\n"); log.flush()
            np.save(policy_file(f"fcb_{tag(name)}.npy"), best[1])
            print(f"FCB {name} gen {g:3d} mean {sc.mean():.3f} center {sc_c:.3f} best {best[0]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return best


if __name__ == "__main__":
    from scenarios import SCENARIOS
    which = sys.argv[1:] or [s[0] for s in SCENARIOS]
    for name, kw, f_ref, nsc in SCENARIOS:
        if name in which:
            optimise(name, kw, f_ref, n_scale=nsc)
