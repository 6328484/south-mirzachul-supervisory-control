"""
Evaluation of all strategies (Sects. 3.6 and 4)
===============================================
1) The four test scenarios (scenarios.py; not used in training), strategies:
   none               — no control (gates keep their calibrated openings)
   practice           — current practice: FF/P local layer, set-points h_max - 0.10, branches fully open
   rule               — rule + FF/P: the five major branches at u = f^1.5
   neural_s0..4       — neural + FF/P, nominal training, 100 generations (train_nominal.py)
   neural_ftnom_s0..4 — + 40 generations of nominal training (control), 140 generations in total
   neural_ftdr_s0..4  — + 40 generations with randomised roughness (the variant reported in the paper)
   fcb                — full-control benchmark (benchmark.py)
2) Robustness: f = 0.8, roughness multiplier n/n0 = 0.8 ... 1.4
   (training values: nominal 1.0; randomised 0.9, 1.0, 1.1, 1.3)
3) Objective J of every policy on the 12 supply/roughness training combinations (Fig. 4b).
Indicators:
  delivery (mean over 24-72 h): group ratios, CV, worst group, demand-weighted delivery
  levels (0-72 h): violation point-hours, longest violation at one point, largest rise above / drop below
                   the limits, largest drawdown rate, time above the drawdown-rate limit
  gate effort: check gates and tail gate (m): total travel, number of operations, largest hourly move;
               regulators (fraction of full opening): total travel, number of operations
Numerical check: for every run, the largest residual of the accepted steps, the number of half-step
subdivisions and the number of unconverged steps (GatedCanal.diag).
Output: results/results.json, results/timeseries.npz, results/robustness.json, results/validation.json
Usage:  python src/evaluate.py     (about 10 min on one core)
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from environment import Env
from supervisory import practice_policy, rule_policy, neural_policy
from benchmark import FullEnv, make_fcb, NN_SEED
from metrics import level_bounds, level_metrics, travel_metrics, MOVE_TOL_A, MOVE_TOL_U
from objective import objective
from scenarios import SCENARIOS, COMBOS, N_TRAIN, tag
from paths import policy_file, result_file

SEEDS = [0, 1, 2, 3, 4]
ROBUST_N = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


def gate_metrics_all(env, rec, ctrl_every=3600):
    """All gates operated by the local layer (7 check gates + tail gate) and the regulators."""
    step = int(ctrl_every / env.dt)
    idx = np.append(env.ci, len(env.a0) - 1)
    a = np.vstack([env.a0, rec["a"]])[::step][:, idx]
    g = travel_metrics(a, MOVE_TOL_A)
    nu = rec["u"].shape[1] if rec["u"].ndim == 2 else 0
    if nu:
        u = np.vstack([np.ones(nu), rec["u"]])[::step]
        r = travel_metrics(u, MOVE_TOL_U)
    else:
        r = dict(travel=0.0, moves=0, max_rate=0.0)
    return dict(check_travel_m=g["travel"], check_moves=g["moves"], check_max_rate=g["max_rate"],
                reg_travel_frac=r["travel"], reg_moves=r["moves"], reg_max_rate=r["max_rate"])


def evaluate(env, rec, method, n_reg_controlled):
    out = dict(method=method)
    if "failed_at_h" in rec:
        out["failed_at_h"] = float(rec["failed_at_h"])
        return out
    sel = rec["t"] >= 24 * 3600
    s = rec["shares"][sel].mean(axis=0)
    c = env.c
    xm = 0.5 * (c.x[1:] + c.x[:-1])
    q0 = c.actual_offtake(env.h0, c.q_off - c.seepage)
    w = [q0[(xm > c.edges[i]) & (xm <= c.edges[i + 1] + 1)].sum() for i in range(1, 9)] + [env.Q0[-1]]
    w = np.array(w)
    out.update(shares=s.tolist(), cv=float(s.std() / s.mean()), worst=float(s.min()),
               mean=float(s.mean()), weighted=float(s @ w / w.sum()))
    out.update(level_metrics(env, rec))
    out.update(gate_metrics_all(env, rec))
    no_change = method in ("none",)
    out["n_gates"] = 0 if no_change else (len(env.ci) + 1 + n_reg_controlled)
    return out


def run_one(env, ctl, kw, method, nreg):
    env.c.reset_diag()
    rec = env.run(ctl, **kw)
    res = evaluate(env, rec, method, nreg)
    res.update(max_residual=env.c.diag["max_res"], substeps=env.c.diag["substeps"],
               unconverged=env.c.diag["unconverged"])
    return rec, res


def fmt(res):
    if "failed_at_h" in res:
        return f"tail dry after {res['failed_at_h']:.0f} h"
    return (f"CV {res['cv']:.3f} worst {res['worst']*100:5.1f}% violation {res['violation_hours']:6.1f} point-h "
            f"rise {res['max_over_m']:.2f} drop {res['max_under_m']:.2f} travel {res['check_travel_m']:.1f} m "
            f"operations {res['check_moves']}+{res['reg_moves']}")


def main():
    th_nom = {s: np.load(policy_file(f"nominal_g100_seed{s}.npy")) for s in SEEDS}
    th_ftnom = {s: np.load(policy_file(f"nominal_g140_seed{s}.npy")) for s in SEEDS}
    th_ftdr = {s: np.load(policy_file(f"randomised_g140_seed{s}.npy")) for s in SEEDS}
    variants = [("neural", th_nom), ("neural_ftnom", th_ftnom), ("neural_ftdr", th_ftdr)]
    results, series = [], {}
    envs, fenvs = {}, {}
    for name, kw, f_ref, nsc in SCENARIOS:
        env = envs.setdefault(nsc, Env(n_scale=nsc))
        fenv = fenvs.setdefault(nsc, FullEnv(n_scale=nsc))
        runs = [("none", env, None, 0), ("practice", env, practice_policy(env), 0),
                ("rule", env, rule_policy(env), len(env.majors))]
        for vn, th in variants:
            runs += [(f"{vn}_s{s}", env, neural_policy(env, th[s]), len(env.majors)) for s in SEEDS]
        base = np.load(policy_file(f"nominal_g100_seed{NN_SEED}.npy"))
        runs.append(("fcb", fenv, make_fcb(fenv, base, np.load(policy_file(f"fcb_{tag(name)}.npy"))), len(fenv.majors)))
        for method, e, ctl, nreg in runs:
            rec, res = run_one(e, ctl, kw, method, nreg)
            res.update(scenario=name, f_ref=f_ref)
            results.append(res)
            for k in ("t", "hu", "shares"):
                series[f"{tag(name)}__{method}__{k}"] = rec[k]
            print(f"{name:9s} {method:16s} {fmt(res)}  [residual {res['max_residual']:.0e}, "
                  f"half-steps {res['substeps']}, unconverged {res['unconverged']}]", flush=True)
        series[f"bounds__{nsc}"] = np.vstack([env.hu0, *level_bounds(env)[1:]])
    json.dump(results, open(result_file("results.json"), "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(result_file("timeseries.npz"), **series)

    # --- robustness to the roughness of the whole canal ---
    robust = []
    for n in ROBUST_N:
        env = Env(n_scale=n)
        runs = [("rule", rule_policy(env))]
        for vn, th in variants:
            runs += [(f"{vn}_s{s}", neural_policy(env, th[s])) for s in SEEDS]
        for method, ctl in runs:
            rec, res = run_one(env, ctl, dict(f=0.8), method, len(env.majors))
            res.update(n_scale=n)
            robust.append(res)
            print(f"robust n x{n:.1f} {method:16s} {fmt(res)}  [residual {res['max_residual']:.0e}, "
                  f"unconverged {res['unconverged']}]", flush=True)
    json.dump(robust, open(result_file("robustness.json"), "w"), indent=1, ensure_ascii=False)

    # --- objective J of every policy on the 12 randomisation combinations (validation set) ---
    venvs = {n: Env(n_scale=n) for n in N_TRAIN}
    val = []
    for vn, th in variants:
        for s in SEEDS:
            J = [objective(venvs[n], venvs[n].run(neural_policy(venvs[n], th[s]), f=f), f) for f, n in COMBOS]
            J_nom = sum(j for (f, n), j in zip(COMBOS, J) if n == 1.0)
            val.append(dict(method=vn, seed=s, J_val=float(sum(J)), J_nominal3=float(J_nom)))
            print(f"val {vn:13s} s{s}  J(12) {sum(J):.3f}  J(nominal 3) {J_nom:.3f}", flush=True)
    J_rule = [objective(venvs[n], venvs[n].run(rule_policy(venvs[n]), f=f), f) for f, n in COMBOS]
    val.append(dict(method="rule", seed=-1, J_val=float(sum(J_rule)),
                    J_nominal3=float(sum(j for (f, n), j in zip(COMBOS, J_rule) if n == 1.0))))
    json.dump(val, open(result_file("validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
