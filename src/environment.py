"""
Simulation environment with hourly control
==========================================
Gates operated by the controllers (13): seven check gates (PK145, PK400, PK623+86, PK746+62,
PK923+02, PK1087+23, PK1177+29) and the tail gate, which the local FF/P layer operates (see
supervisory.py), and the regulators of the five major branches: Right Bayaut, YuR-10, Central,
Kurgantepa and YuR-18 (Akbulak). PK1035 is fully open.
A regulated branch draws q = u * q0 * sqrt(h/h0), with u its relative opening (0..1); the other
offtakes are uncontrolled: q = q0 * sqrt(h/h0).

Observations (measurable by SCADA): f = Q_head/Q_head0, the relative depth deviations upstream of the
9 check structures and the relative tail outflow -> 11 inputs. The controller acts every hour.
"""
import numpy as np
from canal_setup import make_canal, reach_shares

MAJOR_MIN_Q = 10.0       # offtakes with a demand of at least 10 m3/s at Qmax: the five major branches


class Env:
    def __init__(self, dx=1000.0, dt=1800.0, n_scale=1.0):
        # h0, Q0: nominal (passport) state, used for the demands, the level limits and the normalisation
        # of the observations; h_init, Q_init: state at which the simulation starts (differs if n_scale != 1)
        self.c, self.h0, self.Q0, self.h_init, self.Q_init = make_canal(dx=dx, n_scale=n_scale)
        c = self.c
        self.dt = dt
        self.ctrl = ~c.open_struct & np.isfinite(c.a)
        self.ctrl[-1] = False
        self.ci = np.where(self.ctrl)[0]                      # the 7 check gates
        self.a0 = c.a.copy()
        self.a_max = np.maximum(c.Hg, c.a) * 1.3
        self.majors = [o for o in c.offtake_list if o[3] >= MAJOR_MIN_Q]
        self.q_minor = c.q_off.copy()
        for _, _, seg, q in self.majors:
            self.q_minor[seg] -= q
        self.up_nodes = np.append(c.u_nodes, c.N - 1)
        self.hu0 = self.h0[self.up_nodes]
        self.n_obs = 11
        self.n_act = len(self.ci) + len(self.majors)

    def qoff(self, u):
        q = self.q_minor.copy()
        for (_, _, seg, q0), uj in zip(self.majors, u):
            q[seg] += uj * q0
        return q

    def obs(self, h, Q, f):
        return np.concatenate([[f - 1.0], (h[self.up_nodes] - self.hu0) / self.hu0, [Q[-1] / self.Q0[-1] - 1.0]])

    def run(self, controller, f=0.8, T=72 * 3600, ctrl_every=3600, Qin_fn=None):
        """controller(obs, a, u) -> (a_new, u_new). Returns a dictionary of records."""
        c = self.c
        Qin_fn = Qin_fn or (lambda t: 300 * (1 - (1 - f) * min(max((t - 3600) / 3600, 0), 1)))
        h, Q = self.h_init.copy(), self.Q_init.copy()
        a = self.a0.copy(); u = np.ones(len(self.majors))
        rec = dict(t=[], shares=[], hu=[], a=[], u=[])
        k_ctrl = int(ctrl_every / self.dt)
        for n in range(1, int(T / self.dt) + 1):
            t = n * self.dt
            fq = Qin_fn(t) / Qin_fn(0)
            if controller is not None and (n - 1) % k_ctrl == 0:
                self.last_h, self.last_Q = h, Q          # local measurements (used by the local layer)
                a, u = controller(self.obs(h, Q, fq), a, u)
            qoff = self.qoff(u)
            h, Q = c.step(h, Q, self.dt, Qin_fn(t), qoff=qoff, a=a)
            if not np.isfinite(h).all() or h.min() < 0.05:
                rec["failed_at_h"] = t / 3600
                break
            rec["t"].append(t); rec["shares"].append(reach_shares(c, h, Q, self.h0, self.Q0, qoff))
            rec["hu"].append(h[self.up_nodes].copy()); rec["a"].append(a.copy()); rec["u"].append(u.copy())
        for k in ("t", "shares", "hu", "a", "u"):
            rec[k] = np.array(rec[k])
        return rec
