"""Evaluation of the optimized staged rule and the gate-effort variants (Sects. V-A–V-F, Tables IV–V).
1) staged rule (nominal and randomised training, 3 seeds) and the effort variants of the neural supervisor
   (gate-movement weights x5 and x15, 3 seeds) in the four test scenarios
2) roughness sweep n/n0 = 0.8-1.4 at f = 0.8 for the randomised staged rule
3) objective J of the staged policies on the 12 supply/roughness validation cases
4) Molden-Gates and Gini indicators for every run, also recomputed from results/timeseries.npz
Outputs (results/): results_extra.json, robustness_extra.json, validation_extra.json, indicators.json,
timeseries_extra.npz            Usage: python src/evaluate_extra.py      (about 6 min on one core)"""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from environment import Env
from supervisory import neural_policy
from staged_rule import staged_policy
from evaluate import evaluate
from objective import objective
from scenarios import SCENARIOS, COMBOS, N_TRAIN, tag
from indicators import delivery_indicators
from paths import policy_file, result_file

SEEDS3 = [0, 1, 2]
ROBUST_N = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


def variants():
    v = []
    for s in SEEDS3:
        v.append((f"staged_dr_s{s}", "staged", f"staged_randomised_g140_seed{s}.npy"))
        v.append((f"staged_nom_s{s}", "staged", f"staged_nominal_g100_seed{s}.npy"))
    for w in (5, 15):
        for s in SEEDS3:
            v.append((f"neural_w{w}_s{s}", "neural", f"effort_w{w}_g140_seed{s}.npy"))
    return v


def make(env, kind, name):
    th = np.load(policy_file(name))
    return staged_policy(env, th) if kind == "staged" else neural_policy(env, th)


def main():
    V = variants(); results, series, envs = [], {}, {}
    for name, kw, f_ref, nsc in SCENARIOS:
        env = envs.setdefault(nsc, Env(n_scale=nsc))
        for m, kind, fn in V:
            rec = env.run(make(env, kind, fn), **kw)
            res = evaluate(env, rec, m, len(env.majors))
            if "failed_at_h" not in rec:
                res.update(delivery_indicators(rec["t"], rec["shares"]))
            res.update(scenario=name, f_ref=f_ref); results.append(res)
            for k in ("t", "hu", "shares", "u", "a"):
                series[f"{tag(name)}__{m}__{k}"] = rec[k]
            print(name, m, res.get("cv"), res.get("failed_at_h"), flush=True)
    json.dump(results, open(result_file("results_extra.json"), "w"), indent=1)
    np.savez_compressed(result_file("timeseries_extra.npz"), **series)

    robust = []
    for n in ROBUST_N:
        env = Env(n_scale=n)
        for m, kind, fn in V:
            if m.startswith("staged_dr"):
                res = evaluate(env, env.run(make(env, kind, fn), f=0.8), m, len(env.majors)); res.update(n_scale=n)
                robust.append(res)
    json.dump(robust, open(result_file("robustness_extra.json"), "w"), indent=1)

    venvs = {n: Env(n_scale=n) for n in N_TRAIN}; val = []
    for m, kind, fn in V:
        if m.startswith("staged"):
            J = [objective(venvs[n], venvs[n].run(make(venvs[n], kind, fn), f=f), f) for f, n in COMBOS]
            val.append(dict(method=m.rsplit("_s", 1)[0], seed=int(m.rsplit("_s", 1)[1]), J_val=float(sum(J)),
                            J_nominal3=float(sum(j for (f, n), j in zip(COMBOS, J) if n == 1.0))))
    json.dump(val, open(result_file("validation_extra.json"), "w"), indent=1)

    d = np.load(result_file("timeseries.npz")); ind = []
    for k in d.files:
        if k.endswith("__shares"):
            sc, m, _ = k.split("__"); t = d[f"{sc}__{m}__t"]
            if len(t) and t[-1] >= 72 * 3600 - 1:
                r = delivery_indicators(t, d[k]); r.update(scenario=sc, method=m); ind.append(r)
    json.dump(ind, open(result_file("indicators.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
