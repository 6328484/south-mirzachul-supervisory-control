"""Training of the optimized staged rule and of the gate-effort variants (Sects. 3.4 and 4.5). Same OpenAI-ES, population (8 antithetic pairs), sigma = 0.1,
Adam step 0.05, objective J and selection rules as train_nominal.py / train_continue.py.
  python src/train_extra.py staged nominal <seed> 100      staged rule, nominal canal, from theta ~ N(0, 0.01^2)
  python src/train_extra.py staged randomised <seed> 40    staged rule, +40 generations with randomised roughness
  python src/train_extra.py effort <w> <seed> 40           neural supervisor, +40 nominal generations from
                                                           nominal_g100 with both gate-movement weights x w
Outputs (results/policies, results/training_logs):
  staged_nominal_g100_seed<s>, staged_randomised_g140_seed<s>, effort_w<w>_g140_seed<s>  (.npy / .csv)
About 50 min (staged nominal), 30 min (staged randomised) and 19 min (effort) per run on two cores."""
import sys, time, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import objective as OBJ
from environment import Env
from supervisory import neural_policy
from staged_rule import staged_policy, n_params as staged_n_params
from scenarios import F_TRAIN, N_TRAIN, COMBOS
from paths import policy_file, log_file

N_EP, VAL_EVERY = 4, 10
_envs, _pol = None, None


def _init(kind, weight):
    global _envs, _pol
    warnings.filterwarnings("ignore")
    OBJ.W_TRAVEL *= weight; OBJ.W_REG *= weight
    _envs = {n: Env(n_scale=n) for n in N_TRAIN}
    _pol = staged_policy if kind == "staged" else neural_policy


def evaluate(args):
    theta, episodes = args
    return sum(OBJ.objective(_envs[n], _envs[n].run(_pol(_envs[n], theta), f=f), f) for f, n in episodes)


def main(kind, mode, seed, gens, half_pop=8, sigma=0.1, lr=0.05):
    weight = 1.0
    if kind == "effort":
        weight, mode = float(mode), "nominal"
        name = f"effort_w{weight:g}_g140_seed{seed}"
        theta = np.load(policy_file(f"nominal_g100_seed{seed}.npy")).copy()
        rng = np.random.default_rng(seed)
    elif mode == "nominal":
        name = f"staged_nominal_g100_seed{seed}"
        rng = np.random.default_rng(seed + 77)
        theta = 0.01 * rng.standard_normal(staged_n_params(Env()))
    else:
        name = f"staged_randomised_g140_seed{seed}"
        rng = np.random.default_rng(1000 + seed + 77)
        theta = np.load(policy_file(f"staged_nominal_g100_seed{seed}.npy")).copy()
    kindp = "staged" if kind == "staged" else "neural"
    P = len(theta); m = np.zeros(P); v = np.zeros(P); b1, b2 = 0.9, 0.999
    nominal = [(f, 1.0) for f in F_TRAIN]
    log = open(log_file(name + ".csv"), "w"); log.write("gen,train_mean,train_center,select,best,seconds\n")
    with Pool(2, initializer=_init, initargs=(kindp, weight)) as pool:
        def sel_score(th):
            return float(sum(pool.map(evaluate, [(th, [e]) for e in COMBOS])))
        best = (sel_score(theta), theta.copy(), 0) if mode == "randomised" else (np.inf, theta.copy(), 0)
        for g in range(1, gens + 1):
            t0 = time.time()
            ep = [COMBOS[i] for i in rng.choice(len(COMBOS), N_EP, replace=False)] if mode == "randomised" else nominal
            eps = rng.standard_normal((half_pop, P))
            cands = [theta + sigma * e for e in eps] + [theta - sigma * e for e in eps] + [theta]
            sc = np.array(pool.map(evaluate, [(c, ep) for c in cands])); scc = sc[-1]; sc = sc[:-1]
            s_txt = ""
            if mode == "nominal":
                if scc < best[0]: best = (scc, theta.copy(), g)
                ib = int(np.argmin(sc))
                if sc[ib] < best[0]: best = (sc[ib], cands[ib].copy(), g)
            elif g % VAL_EVERY == 0:
                s = sel_score(theta); s_txt = f"{s:.4f}"
                if s < best[0]: best = (s, theta.copy(), g)
            r = np.empty(2 * half_pop); r[np.argsort(sc)] = np.arange(2 * half_pop)
            ut = -(r / (2 * half_pop - 1) - 0.5)
            grad = -((ut[:half_pop] - ut[half_pop:]) @ eps) / (half_pop * sigma)
            m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
            theta = theta - lr * (m / (1 - b1 ** g)) / (np.sqrt(v / (1 - b2 ** g)) + 1e-8)
            log.write(f"{g},{sc.mean():.4f},{scc:.4f},{s_txt},{best[0]:.4f},{time.time()-t0:.1f}\n"); log.flush()
            np.save(policy_file(name + ".npy"), best[1])
            print(f"{name} gen {g} mean {sc.mean():.3f} best {best[0]:.3f}", flush=True)
        if mode == "randomised":
            s = sel_score(theta)
            if s < best[0]: best = (s, theta.copy(), gens + 1)
        np.save(policy_file(name + ".npy"), best[1])
    log.close()


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]), int(a[4]))
