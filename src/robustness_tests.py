"""Measurement-noise and fault tests (Sect. V-G, Table VI), evaluation only.
Noise: Gaussian errors on every measured depth (sd 0.02 / 0.05 m) and a 2 % relative error on every measured
discharge, drawn independently each hour and seen by both control layers; three realisations each, at f = 0.75
and under variable inflow. Faults (f = 0.75): (a) check gate PK1087+23 stuck at its calibrated opening;
(b) Right Bayaut regulator stuck fully open; (c) 6-h SCADA outage at 12-18 h, all gates hold their positions.
Supervisors: fixed rule, neural (randomised_g140, 5 seeds), staged rule (staged_randomised_g140, 3 seeds).
Output: results/robustness_noise_faults.json      Usage: python src/robustness_tests.py"""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from environment import Env
from supervisory import rule_policy, neural_policy
from staged_rule import staged_policy
from evaluate import evaluate
from scenarios import variable_inflow
from indicators import delivery_indicators
from paths import policy_file, result_file

STAGED_SEEDS = [0, 1, 2]


def noisy(env, ctrl, sh, sq, rng):
    def c(o, a, u):
        h = env.last_h + rng.normal(0, sh, env.last_h.shape)
        Q = env.last_Q * (1 + rng.normal(0, sq, env.last_Q.shape))
        env.last_h, env.last_Q = np.maximum(h, 0.01), Q
        return ctrl(env.obs(env.last_h, Q, o[0] + 1.0), a, u)
    return c


def faulty(env, ctrl, kind):
    k_gate = list(env.ci).index(int(np.where(np.isclose(env.c.x[env.c.u_nodes], 108700, atol=600))[0][0]))
    clock = {"h": 0}

    def c(o, a, u):
        t = clock["h"]; clock["h"] += 1
        if kind == "outage" and 12 <= t < 18:
            return a, u
        a2, u2 = ctrl(o, a, u)
        if kind == "stuck_gate":
            g = env.ci[k_gate]; a2 = a2.copy(); a2[g] = env.a0[g]
        if kind == "stuck_reg":
            u2 = u2.copy(); u2[0] = 1.0
        return a2, u2
    return c


def policies(env):
    out = [("rule", lambda: rule_policy(env))]
    for s in range(5):
        th = np.load(policy_file(f"randomised_g140_seed{s}.npy"))
        out.append((f"neural_s{s}", lambda th=th: neural_policy(env, th)))
    for s in STAGED_SEEDS:
        th = np.load(policy_file(f"staged_randomised_g140_seed{s}.npy"))
        out.append((f"staged_s{s}", lambda th=th: staged_policy(env, th)))
    return out


def record(env, rec, m, extra):
    out = evaluate(env, rec, m, 5)
    if "failed_at_h" not in rec:
        out.update(delivery_indicators(rec["t"], rec["shares"]))
    out.update(extra)
    return out


def main():
    env = Env(); res = []
    for name, kw in (("f=0.75", dict(f=0.75)), ("variable", dict(f=0.8, Qin_fn=variable_inflow))):
        for sh in (0.0, 0.02, 0.05):
            for r in (range(3) if sh > 0 else [0]):
                for m, mk in policies(env):
                    ctl = mk() if sh == 0 else noisy(env, mk(), sh, 0.02, np.random.default_rng(100 * r + 7))
                    res.append(record(env, env.run(ctl, **kw), m, dict(case=name, test="noise", sd=sh, real=r)))
                    print(name, sh, r, m, res[-1].get("cv"), res[-1].get("failed_at_h"), flush=True)
    for kind in ("stuck_gate", "stuck_reg", "outage"):
        for m, mk in policies(env):
            res.append(record(env, env.run(faulty(env, mk(), kind), f=0.75), m, dict(case="f=0.75", test=kind)))
            print(kind, m, res[-1].get("cv"), res[-1].get("failed_at_h"), flush=True)
    json.dump(res, open(result_file("robustness_noise_faults.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
