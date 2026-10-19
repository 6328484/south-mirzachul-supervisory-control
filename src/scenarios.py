"""Test scenarios (Table 3; not used in training) and training cases (Sect. 3.4)."""
import itertools


def variable_inflow(t):
    """Head discharge 300 -> 230 m3/s within 6 h, partial recovery to 260 m3/s from 36 h."""
    if t < 3600:
        return 300.0
    if t < 7 * 3600:
        return 300 - 70 * (t - 3600) / (6 * 3600)
    if t < 36 * 3600:
        return 230.0
    return min(260.0, 230 + 30 * (t - 36 * 3600) / (6 * 3600))


# (name, arguments of Env.run, reference supply ratio f, roughness multiplier n/n0)
SCENARIOS = [
    ("f=0.75", dict(f=0.75), 0.75, 1.0),
    ("f=0.85", dict(f=0.85), 0.85, 1.0),
    ("variable", dict(f=0.8, Qin_fn=variable_inflow), 230 / 300, 1.0),
    ("n x1.2", dict(f=0.8), 0.8, 1.2),
]

# Training: supply ratios f of the training episodes and roughness multipliers n/n0 of the
# randomised-roughness training (the test value n/n0 = 1.2 is never used in training).
F_TRAIN = [0.7, 0.8, 0.9]
N_TRAIN = [0.9, 1.0, 1.1, 1.3]
COMBOS = list(itertools.product(F_TRAIN, N_TRAIN))


def tag(name):
    """File-name tag of a scenario, e.g. "f=0.75" -> "f075", "n x1.2" -> "n_x12"."""
    return name.replace(" ", "_").replace("=", "").replace(",", "").replace(".", "")
