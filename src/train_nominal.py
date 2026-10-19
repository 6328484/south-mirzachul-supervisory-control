"""Training of the neural supervisor on the nominal canal (Sect. 3.4).
OpenAI-ES with antithetic sampling (8 pairs = 16 candidates per generation, sigma = 0.1), centred-rank
utilities and Adam (step size 0.05). Each candidate is scored by the sum of the objective J
(objective.py) over three 72-h training episodes with f = 0.7, 0.8 and 0.9. The best candidate
evaluated during training is kept. The parameters start from N(0, 0.01^2), i.e. practically the rule.
Output: results/policies/nominal_g100_seed<seed>.npy, results/training_logs/nominal_g100_seed<seed>.csv
Usage:  python src/train_nominal.py <seed> <generations>
        (paper: seeds 0-4, 100 generations, about 36 min per run on two cores)"""
import sys
import time
import warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from environment import Env
from supervisory import neural_policy, n_params
from objective import objective
from scenarios import F_TRAIN
from paths import policy_file, log_file

# The stored runs used the solver without step halving (see GatedCanal.max_halvings); keep 0 to
# reproduce them exactly. Halving only changes candidate runs in which Newton iteration fails.
SOLVER_HALVINGS = 0
_env = None


def _init():
    global _env
    warnings.filterwarnings("ignore")
    _env = Env()
    _env.c.max_halvings = SOLVER_HALVINGS


def evaluate(theta):
    return sum(objective(_env, _env.run(neural_policy(_env, theta), f=f), f) for f in F_TRAIN)


def main(seed=0, gens=100, half_pop=8, sigma=0.1, lr=0.05):
    rng = np.random.default_rng(seed)
    env = Env(); P = n_params(env)
    theta = 0.01 * rng.standard_normal(P)           # a different starting point for each seed
    m = np.zeros(P); v = np.zeros(P); b1, b2 = 0.9, 0.999
    best = (np.inf, theta.copy())
    log = open(log_file(f"nominal_g100_seed{seed}.csv"), "w"); log.write("gen,score_mean,score_center,best,seconds\n")
    with Pool(2, initializer=_init) as pool:
        for g in range(1, gens + 1):
            t0 = time.time()
            eps = rng.standard_normal((half_pop, P))
            cands = [theta + sigma * e for e in eps] + [theta - sigma * e for e in eps] + [theta]
            scores = np.array(pool.map(evaluate, cands))
            sc_c = scores[-1]; scores = scores[:-1]
            if sc_c < best[0]:
                best = (sc_c, theta.copy())
            ib = int(np.argmin(scores))
            if scores[ib] < best[0]:
                best = (scores[ib], cands[ib].copy())
            ranks = np.empty(2 * half_pop); ranks[np.argsort(scores)] = np.arange(2 * half_pop)
            util = -(ranks / (2 * half_pop - 1) - 0.5)
            grad = -((util[:half_pop] - util[half_pop:]) @ eps) / (half_pop * sigma)
            m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad**2
            theta = theta - lr * (m / (1 - b1**g)) / (np.sqrt(v / (1 - b2**g)) + 1e-8)
            dt = time.time() - t0
            log.write(f"{g},{scores.mean():.4f},{sc_c:.4f},{best[0]:.4f},{dt:.1f}\n"); log.flush()
            np.save(policy_file(f"nominal_g100_seed{seed}.npy"), best[1])
            print(f"seed {seed} gen {g:3d}  mean {scores.mean():.3f}  center {sc_c:.3f}  best {best[0]:.3f}  ({dt:.0f}s)", flush=True)
    log.close()


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0,
         gens=int(sys.argv[2]) if len(sys.argv) > 2 else 100)
