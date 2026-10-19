"""Fig. 2: the canal in the model, calibrated steady state at Qmax = 300 m3/s.
(a) discharge along the canal (the offtakes reduce it), with the five regulated branches marked;
(b) longitudinal profile: bed, water surface and the allowed depth range at the 9 control points.
Output: figures/Fig2_canal.png/.pdf/.eps (174 mm, Liberation Sans 8 pt) and the values of
Tables 1 and 2: results/table_canal.json.
Usage:  python src/make_fig_canal.py"""
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from environment import Env
from metrics import level_bounds
from gated_model import PASSPORT, reach_table, load_structures
from make_figures import MM, INK, INK2, MUTED, GRID, AXIS, BLUE, BAND, save  # noqa: F401 (common style)
from paths import result_file

BRANCH_LABEL = {"Right Bayaut branch": "Right Bayaut", "YuR-10": "YuR-10", "Central branch": "Central",
                "Kurgantepa branch": "Kurgantepa", "YuR-18 (Akbulak branch)": "YuR-18 (Akbulak)"}
PK_LABEL = ["PK145", "PK400", "PK623+86", "PK746+62", "PK923+02", "PK1035", "PK1087+23", "PK1177+29", "PK1237"]


def main():
    env = Env()
    c = env.c
    x = c.x / 1000.0
    z = c.z - c.z[0]
    h0, Q0 = env.h0, env.Q0
    _, lo, _ = level_bounds(env)
    up = env.hu0
    xs = x[env.up_nodes]                       # control points (9)
    zs = z[env.up_nodes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(174 * MM, 118 * MM), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.12))
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
        for xk in xs:
            ax.axvline(xk, color=AXIS, lw=0.7, zorder=1)
    # (a) discharge
    ax1.plot(x, Q0, color=INK, lw=1.3, zorder=3)
    for name, km, seg, q in env.majors:
        k = int(np.argmin(abs(x - km)))
        qa = Q0[max(k - 1, 0)]
        ax1.plot([km], [qa], marker="v", ms=5, color=BLUE, zorder=4, mec="white", mew=0.6)
    lab_pos = {"Right Bayaut": (16.5, 285, "left"), "YuR-10": (49.3, 255, "left"), "Central": (64.4, 175, "left"),
               "Kurgantepa": (64.4, 135, "left"), "YuR-18 (Akbulak)": (94.0, 110, "left")}
    for name, km, seg, q in env.majors:
        lab = BRANCH_LABEL[name]
        tx, ty, ha = lab_pos[lab]
        k = int(np.argmin(abs(x - km)))
        ax1.annotate(f"{lab}, {q:.1f}", xy=(km, Q0[max(k - 1, 0)]), xytext=(tx, ty), ha=ha, va="center",
                     fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                                            shrinkA=1, shrinkB=3))
    ax1.set_ylim(0, 340)
    ax1.set_yticks([0, 100, 200, 300])
    ax1.set_ylabel("Discharge (m³/s)")
    fig.text(0.1, 0.985, "(a) Discharge along the canal at Qmax; markers: the five regulated branches (demand, m³/s)",
             ha="left", va="top", fontsize=8, color=INK)
    # structure names
    for xk, lab in zip(xs, PK_LABEL):
        ax1.text(xk, 344, lab, rotation=90, ha="center", va="bottom", fontsize=8, color=INK2)
    # reach numbers
    mid = 0.5 * (c.edges[:-1] + c.edges[1:]) / 1000.0
    for i, xm in enumerate(mid):
        ax1.text(xm, 318, str(i + 1), ha="center", va="center", fontsize=8, color=MUTED)
    ax1.text(-2.5, 318, "Reach", ha="right", va="center", fontsize=8, color=MUTED, clip_on=False)

    # (b) longitudinal profile
    ax2.fill_between(x, z, z + h0, color=BAND, lw=0, zorder=2)
    ax2.plot(x, z, color=INK2, lw=1.0, zorder=3)
    ax2.plot(x, z + h0, color=BLUE, lw=1.3, zorder=4)
    for xk, zk, a, b in zip(xs, zs, lo, up):
        ax2.plot([xk, xk], [zk + a, zk + b], color=INK, lw=2.4, solid_capstyle="butt", zorder=5)
        ax2.plot([xk - 0.9, xk + 0.9], [zk + a, zk + a], color=INK, lw=0.9, zorder=5)
    ax2.text(20, 7.0, "Water surface at Qmax", color=BLUE, fontsize=8, ha="left")
    ax2.text(6, -4.2, "Bed (continuous, passport slopes)", color=INK2, fontsize=8, ha="left")
    ax2.annotate("Allowed depth range\n(Qmin–Qmax depth)", xy=(xs[3] + 0.5, zs[3] + 0.5 * (lo[3] + up[3])),
                 xytext=(80, 6.4), fontsize=8, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=2))
    ax2.set_ylabel("Elevation relative to\nbed at the head (m)")
    ax2.set_xlabel("Chainage (km)")
    ax2.set_xlim(-1, 125)
    ax2.set_ylim(-9, 8.5)
    ax2.set_title("(b) Longitudinal profile at Qmax with the allowed depth range at the nine control points",
                  loc="left", fontsize=8, color=INK, pad=4)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.84, bottom=0.09)
    save(fig, "Fig2_canal")

    # ---- values for Tables 1 and 2
    t = reach_table()
    st, checks = load_structures()
    xm = 0.5 * (c.x[1:] + c.x[:-1])
    reaches = []
    for i in range(10):
        a_, b_ = c.edges[i], c.edges[i + 1]
        offs = [o for o in c.offtake_list if a_ < xm[o[2]] <= b_ + 1]
        majors = [(BRANCH_LABEL[o[0]], round(o[3], 1)) for o in env.majors if a_ < xm[o[2]] <= b_ + 1]
        reaches.append(dict(reach=i + 1, km=(round(a_ / 1000, 3), round(b_ / 1000, 3)), L=PASSPORT["L"][i],
                            Q=PASSPORT["Q"][i], b=PASSPORT["b"][i], T=PASSPORT["T"][i], m=round(float(t["m"][i]), 2),
                            h=PASSPORT["h"][i], h_min=PASSPORT["h_min"][i], S=PASSPORT["S"][i],
                            n=round(float(t["n"][i]), 4), n_off=len(offs),
                            cap=round(sum(float(st[st.name == o[0]].Q_cap_m3s.iloc[0]) for o in offs), 1),
                            demand=round(sum(o[3] for o in offs), 1), majors=majors,
                            seep=round(float(c.seepage[(xm > a_) & (xm <= b_ + 1)].sum()), 1)))
    structs = []
    for k in range(c.n_struct + 1):
        r = checks.iloc[int(np.argmin(abs(checks.km - c.edges[(c.struct_reach + [9])[k] + 1] / 1000)))]
        structs.append(dict(pk=PK_LABEL[k], km=float(r.km), spans=int(r.spans), gw=float(r.gate_w_m),
                            gh=float(r.gate_h_m), cap=float(r.Q_cap_m3s), W_pass=float(c.W_passport[k]),
                            W=round(float(c.W[k]), 2), a0=None if not np.isfinite(c.a[k]) else round(float(c.a[k]), 2),
                            up=round(float(up[k]), 2), lo=float(lo[k]), open=bool(c.open_struct[k])))
    json.dump(dict(reaches=reaches, structures=structs), open(result_file("table_canal.json"), "w"), indent=1,
              ensure_ascii=False)
    for r in reaches:
        print(r)
    for s in structs:
        print(s)


if __name__ == "__main__":
    main()
