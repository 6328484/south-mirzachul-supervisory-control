"""Fig. 3: scheme of the two-layer control.
Output: figures/Fig3_architecture.png/.pdf/.eps
Usage:  python src/fig_architecture.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from paths import figure_file

MM = 1 / 25.4
INK, INK2, AXIS, FILL, BLUE, BLUE_L = "#0b0b0b", "#52514e", "#c3c2b7", "#f6f5f2", "#2a78d6", "#eaf2fc"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "mathtext.fontset": "custom",
                     "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "mathtext.bf": "Liberation Sans:bold"})

fig = plt.figure(figsize=(174 * MM, 92 * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 174); ax.set_ylim(0, 92); ax.axis("off")


def box(x, y, w, h, title, body, edge=AXIS, face=FILL, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=face, ec=edge, lw=lw))
    ax.text(x + 2.5, y + h - 2.5, title, va="top", ha="left", fontweight="bold", color=INK)
    ax.text(x + 2.5, y + h - 7.5, body, va="top", ha="left", color=INK2, linespacing=1.5)


def arrow(p, q, label=None, lx=0, ly=0, ha="center", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha=ha, va="center", color=INK, fontsize=8)


# offline training
box(4, 68, 81, 20, "Offline training (evolution strategy)",
    "Episodes on the Saint-Venant model, f = 0.7, 0.8, 0.9\n"
    "J = CV + 0.5·shortfall + 2·level violation + gate travel\n"
    r"5 seeds × 100 generations, then 40 with $n/n_0$ = 0.9–1.3")
# supervisory layer
box(4, 34, 81, 29.5, "Supervisory layer (every 1 h)",
    "Neural network 11 → 8 → 12  (or fixed rule)\n"
    "Inputs: head supply ratio f, depths upstream of\n"
    "9 check structures, tail outflow\n"
    "Outputs: 7 level set-points (within allowed range),\n"
    "5 branch-regulator openings u (steps ≥ 5 %)", edge=BLUE, face=BLUE_L, lw=1.0)
# local layer
box(4, 4, 81, 26, "Local layer FF/P (every 1 h, 8 gates)",
    r"Target flow:  $Q_k = Q_{\mathrm{in},k} - \sum q_{\mathrm{pool}\,k} + A_k\,(h_k - h_{\mathrm{set},k})\,/\,T_c$" "\n"
    r"Opening (gate equation):  $a_k = Q_k \,/\, [\mu W_k (2g\,\Delta Y_k)^{1/2}]$,  $\mu$ = 0.65" "\n"
    "Dead band 1 cm, at most 0.30 m per hour\n"
    "Same layer for all supervisors and the benchmark")
# canal
box(112, 4, 58, 84, "Canal (South Mirzachul, 123.7 km)",
    "1-D Saint-Venant equations,\n"
    "Preissmann scheme (θ = 0.6)\n\n"
    "10 reaches from the 2000 passport\n"
    "9 check structures (incl. tail gate)\n"
    r"58 offtakes, $q = u\,q^0 (h / h^0)^{1/2}$" "\n"
    "5 major branches regulated:\n"
    "  Right Bayaut, YuR-10, Central,\n"
    "  Kurgantepa, YuR-18 (Akbulak)\n\n"
    r"Head inflow $Q_H(t)$, initially 300 m³/s" "\n"
    "→ shortage scenarios\n\n"
    "All controller inputs are\n"
    "measurable by SCADA")

arrow((44.5, 68), (44.5, 63.5), "policy weights θ", lx=1.5, ha="left")
arrow((44.5, 34), (44.5, 30), "level set-points", lx=1.5, ha="left")
arrow((85, 13), (112, 13), "gate openings", ly=2.2)
arrow((85, 42), (112, 42), "branch openings u", ly=2.2)
arrow((112, 55), (85, 55), "depths, f, tail flow", ly=2.2)
arrow((112, 24), (85, 24), "local depths, flows", ly=2.2)
ax.add_patch(FancyArrowPatch((112, 80), (85, 80), arrowstyle="-|>", mutation_scale=8, lw=0.8,
                             color=INK2, linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0))
ax.text(98.5, 82.2, "simulated episodes", ha="center", va="center", color=INK2, fontsize=8)
for ext in ("png", "pdf", "eps"):
    fig.savefig(figure_file(f"Fig3_architecture.{ext}"), dpi=300)
