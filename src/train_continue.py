"""Continued training of the nominal policies for 40 more generations (Sects. 3.4 and 4.6).
Two variants; all other settings are identical (ES, population, sigma, learning rate, generations):
  randomised — randomised roughness: f in {0.7, 0.8, 0.9} x n/n0 in {0.9, 1.0, 1.1, 1.3};
               each generation uses 4 of the 12 combinations (common to all candidates); the policy is
               selected by its score on all 12 combinations, checked every 10 generations
               (generation 0, the starting policy, is also a candidate)
  nominal    — control: nominal canal, f in {0.7, 0.8, 0.9} (as in train_nominal.py); selection by the
               training score (generation 0, the starting policy, is also a candidate)
The test value n/n0 = 1.2 is used in neither variant.
Input:  results/policies/nominal_g100_seed<seed>.npy
Output: results/policies/<variant>_g140_seed<seed>.npy, results/training_logs/<variant>_g140_seed<seed>.csv
Usage:  python src/train_continue.py <randomised|nominal> <seed> <generations>
        (paper: 40 generations; about 22 min (randomised) and 13 min (nominal) per run on two cores)"""
import sys
import time
import warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from environment import Env
from supervisory import neural_policy
from objective import objective
from scenarios import F_TRAIN, N_TRAIN, COMBOS
from paths import policy_file, log_file

N_EP, VAL_EVERY = 4, 10
_envs = None


def _init():
    global _envs
    warnings.filterwarnings("ignore")
    _envs = {n: Env(n_scale=n) for n in N_TRAIN}


def evaluate(args):
    theta, episodes = args
    return sum(objective(_envs[n], _envs[n].run(neural_policy(_envs[n], theta), f=f), f) for f, n in episodes)


def main(mode, seed, gens=40, half_pop=8, sigma=0.1, lr=0.05):
    if mode not in ("randomised", "nominal"):
        raise SystemExit("mode must be 'randomised' or 'nominal'")
    rng = np.random.default_rng(1000 + seed)
    theta = np.load(policy_file(f"nominal_g100_seed{seed}.npy")).copy()
    P = len(theta)
    m = np.zeros(P); v = np.zeros(P); b1, b2 = 0.9, 0.999
    nominal_eps = [(f, 1.0) for f in F_TRAIN]
    out = f"{mode}_g140_seed{seed}"
    log = open(log_file(f"{out}.csv"), "w")
    log.write("gen,train_mean,train_center,select_score,best,seconds\n")
    with Pool(2, initializer=_init) as pool:
        def select_score(th):
            eps = COMBOS if mode == "randomised" else nominal_eps
            return float(sum(pool.map(evaluate, [(th, [e]) for e in eps])))
        s0 = select_score(theta)
        best = (s0, theta.copy(), 0)
        log.write(f"0,,,{s0:.4f},{s0:.4f},0\n")
        print(f"continue-{mode} seed {seed} gen   0 select {s0:.3f}", flush=True)
        for g in range(1, gens + 1):
            t0 = time.time()
            ep = ([COMBOS[i] for i in rng.choice(len(COMBOS), N_EP, replace=False)] if mode == "randomised" else nominal_eps)
            eps = rng.standard_normal((half_pop, P))
            cands = [theta + sigma * e for e in eps] + [theta - sigma * e for e in eps] + [theta]
            scores = np.array(pool.map(evaluate, [(c, ep) for c in cands]))
            sc_c = scores[-1]; scores = scores[:-1]
            sel = ""
            if mode == "nominal":
                # nominal: the training episodes are fixed, so scores are comparable between generations
                if sc_c < best[0]:
                    best = (sc_c, theta.copy(), g)
                ib = int(np.argmin(scores))
                if scores[ib] < best[0]:
                    best = (scores[ib], cands[ib].copy(), g)
            elif g % VAL_EVERY == 0:
                sel = select_score(theta)
                if sel < best[0]:
                    best = (sel, theta.copy(), g)
            ranks = np.empty(2 * half_pop); ranks[np.argsort(scores)] = np.arange(2 * half_pop)
            util = -(ranks / (2 * half_pop - 1) - 0.5)
            grad = -((util[:half_pop] - util[half_pop:]) @ eps) / (half_pop * sigma)
            m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad**2
            theta = theta - lr * (m / (1 - b1**g)) / (np.sqrt(v / (1 - b2**g)) + 1e-8)
            dt = time.time() - t0
            ss = f"{sel:.4f}" if sel != "" else ""
            log.write(f"{g},{scores.mean():.4f},{sc_c:.4f},{ss},{best[0]:.4f},{dt:.1f}\n"); log.flush()
            np.save(policy_file(f"{out}.npy"), best[1])
            print(f"continue-{mode} seed {seed} gen {g:3d} train {scores.mean():.3f} center {sc_c:.3f} "
                  f"select {ss or '-':>7} best {best[0]:.3f} (gen {best[2]}) ({dt:.0f}s)", flush=True)
        if mode == "randomised":
            sel = select_score(theta)
            if sel < best[0]:
                best = (sel, theta.copy(), gens + 1)
        np.save(policy_file(f"{out}.npy"), best[1])
        print(f"continue-{mode} seed {seed}: selected generation {best[2]}, score {best[0]:.3f} (start {s0:.3f})", flush=True)
    log.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 40)
