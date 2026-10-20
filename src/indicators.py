"""Additional delivery indicators (Sect. 3.2, Eqs. 17-18, Table 5): Molden and Gates (1990) adequacy, dependability and equity,
and the Gini coefficient of the time-averaged delivery ratios of the nine groups (t = 24-72 h)."""
import numpy as np


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def delivery_indicators(t, shares, t_from=24 * 3600):
    s = np.asarray(shares)[np.asarray(t) >= t_from]          # (n_t, 9) delivery ratio vs demand
    P_A = float(np.minimum(s, 1.0).mean())                     # adequacy vs full demand
    P_D = float((s.std(axis=0) / s.mean(axis=0)).mean())       # temporal CV, averaged over groups
    P_E = float((s.std(axis=1) / s.mean(axis=1)).mean())       # spatial CV, averaged over time
    sbar = s.mean(axis=0)
    return dict(P_A=P_A, P_D=P_D, P_E=P_E, gini=gini(sbar), cv=float(sbar.std() / sbar.mean()),
                worst=float(sbar.min()))
