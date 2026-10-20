"""Figures 4-9 and Table IV of the article.
Input:  results/results.json, results/results_extra.json, results/timeseries.npz, results/timeseries_extra.npz,
        results/robustness.json, results/robustness_extra.json, results/validation.json,
        results/validation_extra.json, results/training_logs/*.csv
Output: figures/Fig4_training, Fig5_shares, Fig6_levels, Fig7_gates, Fig8_effort, Fig9_robustness
        (.png 300 dpi, .pdf, .eps), results/table_results.csv, results/table_results.md (Table IV).
Width <= 174 mm, Liberation Sans (metric-compatible with Arial) 8 pt.
Usage:  python src/make_figures.py"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scenarios import SCENARIOS, tag
from paths import figure_file, log_file, result_file

MM = 1 / 25.4
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a4cc2"
BAND, BAND_NOM, LEVEL_BAND = "#cde2fb", "#e8f0fb", "#f0efec"
STYLE = {
    "none":      ("No control", MUTED, "x", "-"),
    "practice":  ("Current practice", INK2, "D", "-"),
    "rule":      ("Rule + FF/P (13 gates)", ORANGE, "s", "-"),
    "neural":       ("Neural + FF/P, nominal, 100 gen. (13 gates)", BLUE, "o", ":"),
    "neural_ftnom": ("Neural + FF/P, nominal, 140 gen. (13 gates)", BLUE, "o", "--"),
    "neural_ftdr":  ("Neural + FF/P, randomized roughness, 140 gen. (13 gates)", BLUE, "o", "-"),
    "fcb":       ("Full-control benchmark (66 gates)", AQUA, "^", "-"),
    "staged_dr": ("Optimized staged rule + FF/P (13 gates)", PURPLE, "v", "-"),
    "staged_nom": ("Optimized staged rule, nominal, 100 gen.", PURPLE, "v", ":"),
    "neural_w5":  ("Neural, nominal 140 gen., gate-effort weights x5", BLUE, "o", "-"),
    "neural_w15": ("Neural, nominal 140 gen., gate-effort weights x15", BLUE, "o", "-"),
}
MAIN = "neural_ftdr"        # neural variant shown in the main figures
MAIN_LABEL = "Neural + FF/P (13 gates)"   # the variant is described in the figure captions
SCEN_LABEL = {"f=0.75": "f = 0.75", "f=0.85": "f = 0.85", "variable": "Variable inflow",
              "n x1.2": "Roughness +20 %, f = 0.8"}
plt.rcParams.update({
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "font.family": "Liberation Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "legend.frameon": False, "lines.linewidth": 1.4, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    for ext in ("png", "pdf", "eps"):
        fig.savefig(figure_file(f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def pick(R, scen, method, key="scenario"):
    """method: none | practice | rule | fcb | neural | neural_ftnom | neural_ftdr (5 seed)"""
    if method.startswith("neural") or method.startswith("staged"):
        return [r for r in R if r[key] == scen and r["method"].startswith(method + "_s")]
    return [r for r in R if r[key] == scen and r["method"] == method]


def handle(m, band=None):
    lab, col, mk, ls = STYLE[m]
    line = Line2D([], [], color=col, marker=mk, ms=4, lw=1.6 if m.startswith("neural") else 1.0, ls=ls,
                  mfc="white" if m in ("neural", "neural_ftnom") else col)
    return (Patch(color=band, lw=0), line) if band else line


# ---------------------------------------------------------------- training curves (Fig. 4)
def fig_training():
    """(a) nominal training: neural supervisor 100 generations + 40 continued nominal generations (control),
    5 seeds; staged rule 100 generations, 3 seeds; (b) objective J on the 12 randomisation cases."""
    from environment import Env
    from supervisory import rule_policy
    from objective import objective
    from scenarios import F_TRAIN
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM))
    ax = axes[0]
    for s_ in range(5):
        d = np.genfromtxt(log_file(f"nominal_g100_seed{s_}.csv"), delimiter=",", names=True)
        ax.plot(d["gen"], d["best"], color=BLUE, lw=1.0, ls=":")
        try:
            e = np.genfromtxt(log_file(f"nominal_g140_seed{s_}.csv"), delimiter=",", names=True)
            ax.plot(100 + e["gen"], e["best"], color=BLUE, lw=1.0, ls="--")
        except OSError:
            pass
    for s_ in range(5):
        try:
            d = np.genfromtxt(log_file(f"staged_nominal_g100_seed{s_}.csv"), delimiter=",", names=True)
            ax.plot(d["gen"], d["best"], color=PURPLE, lw=1.0, ls=":")
        except OSError:
            pass
    env = Env()
    j_rule = sum(objective(env, env.run(rule_policy(env), f=f), f) for f in F_TRAIN)
    ax.axhline(j_rule, color=ORANGE, lw=1.2)
    ax.axvline(100, color=AXIS, lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("Generation"); ax.set_ylabel("Training objective J (nominal, 3 episodes)")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [1, 2, 5, 10, 20]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.yaxis.set_minor_locator(NullLocator()); ax.set_ylim(0.8, 30); ax.set_xlim(0, 140)
    ax.legend([Line2D([], [], color=BLUE, ls=":", lw=1.0), Line2D([], [], color=BLUE, ls="--", lw=1.0),
               Line2D([], [], color=PURPLE, ls=":", lw=1.0), Line2D([], [], color=ORANGE, lw=1.2)],
              ["Neural, nominal training, 5 seeds", "Neural, +40 generations, nominal (control)",
               "Staged rule, nominal training, 3 seeds", f"Fixed rule + FF/P ({j_rule:.1f})"],
              loc="upper right", bbox_to_anchor=(1, 0.86))
    ax.set_title("(a)", loc="left", fontsize=8, color=INK, pad=4)
    ax = axes[1]
    V = json.load(open(result_file("validation.json"))) + json.load(open(result_file("validation_extra.json")))
    groups = [("neural", "Neural\n100 gen."), ("neural_ftnom", "Neural\n+40 nom."),
              ("neural_ftdr", "Neural\n+40 rand."), ("staged_nom", "Staged\n100 gen."),
              ("staged_dr", "Staged\n+40 rand.")]
    for i, (m, lab) in enumerate(groups):
        y = [v["J_val"] for v in V if v["method"] == m]
        col = PURPLE if m.startswith("staged") else BLUE
        ax.scatter(np.full(len(y), i) + np.linspace(-0.12, 0.12, len(y)), y, s=14, color=col,
                   facecolors=col if m.endswith("dr") else "white", zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(y)] * 2, color=INK, lw=1.2, zorder=4)
    jr = [v["J_val"] for v in V if v["method"] == "rule"][0]
    ax.text(0.02, 0.03, f"Fixed rule + FF/P: J = {jr:.1f} (off scale)", transform=ax.transAxes, ha="left", va="bottom",
            color=INK2, fontsize=7)
    ax.set_ylim(0, max(18, max(v["J_val"] for v in V if v["method"] != "rule") * 1.1))
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[1] for g in groups], fontsize=7)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("Objective J on 12 roughness/supply cases"); ax.grid(axis="x", visible=False)
    ax.set_title("(b)", loc="left", fontsize=8, color=INK, pad=4)
    fig.tight_layout()
    save(fig, "Fig4_training")


# ---------------------------------------------------------------- delivery shares (Fig. 5)
def fig_shares(R):
    fig, axes = plt.subplots(2, 2, figsize=(174 * MM, 118 * MM), sharey=True, sharex=True)
    x = np.arange(2, 11)
    short = {"none": "No control", "practice": "Current practice"}
    for k, (ax, (scen, _, f_ref, _)) in enumerate(zip(axes.ravel(), SCENARIOS)):
        notes = []
        ax.axhline(f_ref * 100, color=AXIS, lw=0.8, zorder=1)
        for m in ("none", "practice", "rule", "fcb"):
            r = pick(R, scen, m)[0]
            lab, col, mk, ls = STYLE[m]
            if "failed_at_h" in r:
                notes.append(f"{short.get(m, lab)}: tail dry after {r['failed_at_h']:.1f} h")
                continue
            ax.plot(x, np.array(r["shares"]) * 100, color=col, marker=mk, ms=4, lw=1.0, zorder=3)
        rs = [r for r in pick(R, scen, MAIN) if "failed_at_h" not in r]
        S = np.array([r["shares"] for r in rs]) * 100
        if len(S):
            ax.fill_between(x, S.min(0), S.max(0), color=BAND, lw=0, zorder=2)
            ax.plot(x, S.mean(0), color=BLUE, marker="o", ms=4, lw=1.6, zorder=4)
        rs2 = [r for r in pick(R, scen, "staged_dr") if "failed_at_h" not in r]
        if rs2:
            S2 = np.array([r["shares"] for r in rs2]) * 100
            ax.plot(x, S2.mean(0), color=PURPLE, marker="v", ms=4, lw=1.2, zorder=3.5)
        nf = sum("failed_at_h" in r for r in pick(R, scen, MAIN))
        if nf:
            notes.append(f"Neural: {nf} of 5 seeds dry the tail")
        ax.set_title(f"({'abcd'[k]}) {SCEN_LABEL[scen]}", loc="left", fontsize=8, color=INK, pad=4)
        for j, t in enumerate(notes):
            ax.text(0.02, 0.04 + 0.075 * j, t, transform=ax.transAxes, color=INK2, fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([str(i) for i in range(2, 10)] + ["T"]); ax.set_ylim(20, 106)
    for ax in axes[1]:
        ax.set_xlabel("Offtake group (reach 2–9; T = tail outflow)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Delivered / demand (%)")
    h = [handle("none"), handle("rule"), handle("staged_dr"), handle(MAIN, BAND), handle("fcb"),
         Line2D([], [], color=AXIS, lw=0.8)]
    labs = [STYLE["none"][0], "Fixed rule + FF/P (13 gates)", STYLE["staged_dr"][0] + ", mean of 3 seeds",
            MAIN_LABEL + ", mean and range of 5 seeds", STYLE["fcb"][0], "Head supply ratio f"]
    fig.legend(h, labs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "Fig5_shares")


# ---------------------------------------------------------------- water levels (Fig. 6)
def fig_levels(R, T, scen="f=0.75"):
    B = T["bounds__1.0"]; up, lo = B[0], B[1]
    names = ["PK145", "PK400", "PK623+86", "PK746+62", "PK923+02", "PK1035", "PK1087+23", "PK1177+29", "PK1237 (tail)"]
    hr = T[f"{tag(scen)}__rule__hu"]
    viol = np.maximum(hr - up, 0).max(0) + np.maximum(lo - hr, 0).max(0)
    pts = sorted(np.argsort(viol)[-2:])
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 72 * MM))
    t = T[f"{tag(scen)}__rule__t"] / 3600
    for k, (ax, p) in enumerate(zip(axes, pts)):
        ax.axhspan(lo[p], up[p], color=LEVEL_BAND, lw=0, zorder=0)
        ax.axhline(up[p], color=AXIS, lw=0.8); ax.axhline(lo[p], color=AXIS, lw=0.8)
        for m in ("rule", "fcb"):
            ax.plot(T[f"{tag(scen)}__{m}__t"] / 3600, T[f"{tag(scen)}__{m}__hu"][:, p], color=STYLE[m][1],
                    lw=1.2, marker=STYLE[m][2], ms=3.5, markevery=12)
        hs = [T[f"{tag(scen)}__{MAIN}_s{s}__hu"][:, p] for s in range(5)
              if T[f"{tag(scen)}__{MAIN}_s{s}__hu"].shape[0] == len(t)]
        if hs:
            H = np.array(hs)
            ax.fill_between(t, H.min(0), H.max(0), color=BAND, lw=0, zorder=1)
            ax.plot(t, H.mean(0), color=BLUE, lw=1.6, marker="o", ms=3.5, markevery=12, zorder=4)
        hs2 = [T[f"{tag(scen)}__staged_dr_s{s}__hu"][:, p] for s in range(5)
               if f"{tag(scen)}__staged_dr_s{s}__hu" in T and T[f"{tag(scen)}__staged_dr_s{s}__hu"].shape[0] == len(t)]
        if hs2:
            ax.plot(t, np.mean(hs2, 0), color=PURPLE, lw=1.2, marker="v", ms=3.5, markevery=12, zorder=3.5)
        ax.set_title(f"({'ab'[k]}) Upstream of {names[p]}", loc="left", fontsize=8, color=INK, pad=4)
        ax.text(71, up[p], "upper limit (Qmax depth)", va="bottom", ha="right", color=INK2, fontsize=7)
        ax.text(71, lo[p], "lower limit (Qmin depth)", va="bottom", ha="right", color=INK2, fontsize=7)
        ax.set_xlim(0, 72); ax.set_xticks(range(0, 73, 12)); ax.set_xlabel("Time (h)"); ax.set_ylabel("Water depth (m)")
    fig.legend([handle("rule"), handle("staged_dr"), handle(MAIN, BAND), handle("fcb"), Patch(color=LEVEL_BAND, lw=0)],
               ["Fixed rule + FF/P (13 gates)", STYLE["staged_dr"][0] + ", mean of 3 seeds",
                MAIN_LABEL + ", mean and range of 5 seeds",
                STYLE["fcb"][0], "Allowed depth range"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save(fig, "Fig6_levels")
    return [names[p] for p in pts]


# ---------------------------------------------------------------- gate effort (Fig. 7)
def fig_gates(R):
    methods = ["rule", "staged_dr", MAIN, "fcb"]
    xl = {"f=0.75": "f = 0.75", "f=0.85": "f = 0.85", "variable": "Variable\ninflow", "n x1.2": "Roughness\n+20 %"}
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM))
    xs = np.arange(len(SCENARIOS)); w = 0.2
    for j, m in enumerate(methods):
        trav, trav_e, ops, ops_e = [], [], [], []
        for scen, *_ in SCENARIOS:
            rs = [r for r in pick(R, scen, m) if "failed_at_h" not in r]
            a = np.array([r["check_travel_m"] for r in rs]); o = np.array([r["check_moves"] + r["reg_moves"] for r in rs])
            trav.append(a.mean()); trav_e.append(a.std()); ops.append(o.mean()); ops_e.append(o.std())
        lab, col, mk, ls = STYLE[m]
        seeded = m in (MAIN, "staged_dr")
        kw = dict(color=col, width=w, label=MAIN_LABEL if m == MAIN else ("Fixed rule + FF/P (13 gates)" if m == "rule" else lab),
                  edgecolor="white", linewidth=0.8)
        err = dict(ecolor=INK, capsize=2, error_kw=dict(lw=0.8)) if seeded else {}
        axes[0].bar(xs + (j - 1.5) * w, trav, yerr=trav_e if seeded else None, **kw, **err)
        axes[1].bar(xs + (j - 1.5) * w, ops, yerr=ops_e if seeded else None, **kw, **err)
    for k, (ax, yl) in enumerate(zip(axes, ["Check-gate travel, 0–72 h (m)", "Gate operations, 0–72 h (count)"])):
        ax.set_xticks(xs); ax.set_xticklabels([xl[s[0]] for s in SCENARIOS], fontsize=7)
        ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
        ax.set_title(f"({'ab'[k]})", loc="left", fontsize=8, color=INK, pad=4)
    h, l = axes[0].get_legend_handles_labels()
    l = [x + ", mean ± SD of 5 seeds" if x == MAIN_LABEL else (x + ", mean ± SD of 3 seeds" if x == STYLE["staged_dr"][0] else x) for x in l]
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, "Fig7_gates")


# ---------------------------------------------------------------- roughness robustness (Fig. 9)
def fig_robustness(RB):
    ns = sorted({r["n_scale"] for r in RB})
    metrics = [("worst", 100, "Worst offtake group (%)"), ("cv", 1, "CV of delivery"),
               ("violation_hours", 1, "Level violation (point-h)")]
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 62 * MM))
    for k, (ax, (key, sc, yl)) in enumerate(zip(axes, metrics)):
        for x_ in (0.9, 1.0, 1.1, 1.3):
            ax.axvline(x_, color=GRID, lw=0.8, zorder=0)
        ax.axvline(1.2, color=AXIS, lw=0.8, ls="--", zorder=0)
        r = [pick(RB, n, "rule", key="n_scale")[0] for n in ns]
        y = [rr[key] * sc if "failed_at_h" not in rr else np.nan for rr in r]
        ax.plot(ns, y, color=ORANGE, marker="s", ms=4, lw=1.0, zorder=3)
        M2 = np.array([[rr[key] * sc if "failed_at_h" not in rr else np.nan for rr in pick(RB, n, "staged_dr", key="n_scale")]
                       for n in ns])
        if M2.size:
            ax.plot(ns, np.nanmean(M2, 1), color=PURPLE, marker="v", ms=4, lw=1.2, zorder=3.5)
        for m, band in (("neural_ftnom", BAND_NOM), ("neural_ftdr", BAND)):
            M = np.array([[rr[key] * sc if "failed_at_h" not in rr else np.nan for rr in pick(RB, n, m, key="n_scale")]
                          for n in ns])
            ax.fill_between(ns, np.nanmin(M, 1), np.nanmax(M, 1), color=band, lw=0, zorder=1)
            ax.plot(ns, np.nanmean(M, 1), color=BLUE, ls=STYLE[m][3], marker="o", ms=4,
                    mfc="white" if m == "neural_ftnom" else BLUE, lw=1.6, zorder=4)
        ax.set_xlabel(r"Roughness multiplier $n/n_0$"); ax.set_ylabel(yl); ax.set_xticks(ns)
        ax.set_title(f"({'abc'[k]})", loc="left", fontsize=8, color=INK, pad=4)
        ax.grid(axis="x", visible=False)
    h = [handle("rule"), handle("staged_dr"), handle("neural_ftnom", BAND_NOM), handle("neural_ftdr", BAND),
         Line2D([], [], color=AXIS, lw=0.8, ls="--"), Line2D([], [], color=GRID, lw=0.8)]
    labs = ["Fixed rule + FF/P (13 gates)", "Staged rule, randomized training, mean of 3 seeds", r"Neural, nominal training ($n/n_0$ = 1), 140 gen.",
            r"Neural, randomized roughness ($n/n_0$ = 0.9, 1.0, 1.1, 1.3), 140 gen.", "Test value (not used in training)",
            "Randomized training values"]
    fig.legend(h, labs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "Fig9_robustness")


# ---------------------------------------------------------------- Table IV
def table(R):
    fmt = {"cv": ("{:.3f}", 1), "worst": ("{:.0f}", 100), "violation_hours": ("{:.0f}", 1),
           "max_over_m": ("{:.2f}", 1), "max_under_m": ("{:.2f}", 1), "max_drawdown_rate": ("{:.2f}", 1),
           "check_travel_m": ("{:.1f}", 1), "ops": ("{:.0f}", 1), "n_gates": ("{:.0f}", 1)}
    rows = []
    for scen, *_ in SCENARIOS:
        for m in ("none", "practice", "rule", "staged_nom", "staged_dr", "neural", "neural_ftnom", "neural_ftdr", "neural_w5", "neural_w15", "fcb"):
            rs = pick(R, scen, m)
            ok = [dict(r, ops=r["check_moves"] + r["reg_moves"]) for r in rs if "failed_at_h" not in r]
            row = dict(scenario=SCEN_LABEL[scen], method=STYLE[m][0], n_runs=len(rs), n_failed=len(rs) - len(ok),
                       failed_at_h=";".join(f"{r['failed_at_h']:.1f}" for r in rs if "failed_at_h" in r))
            for c, (f, k) in fmt.items():
                v = np.array([r[c] for r in ok], float) * k if ok else np.array([])
                row[c] = "" if not len(v) else (f.format(v.mean()) + " ± " + f.format(v.std())
                                                if (m.startswith("neural") or m.startswith("staged")) and c != "n_gates" else f.format(v.mean()))
            rows.append(row)
    keys = ["scenario", "method", "n_runs", "n_failed", "failed_at_h"] + list(fmt)
    with open(result_file("table_results.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f'"{r.get(k, "")}"' for k in keys) + "\n")
    hdr = ["Scenario", "Method", "Gates", "CV of delivery", "Worst group (%)", "Level violation (point-h)",
           "Max rise above limit (m)", "Max drop below limit (m)", "Check-gate travel (m)", "Gate operations"]
    with open(result_file("table_results.md"), "w") as fh:
        fh.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            if r["n_failed"] == r["n_runs"]:
                fh.write(f"| {r['scenario']} | {r['method']} | | tail dries after {r['failed_at_h']} h | | | | | | |\n")
                continue
            fh.write(f"| {r['scenario']} | {r['method']} | {r['n_gates']} | {r['cv']} | {r['worst']} | "
                     f"{r['violation_hours']} | {r['max_over_m']} | {r['max_under_m']} | {r['check_travel_m']} | {r['ops']} |\n")
    return rows


def fig_effort(R):
    """Trade-off between gate operations and performance, mean over the three nominal test scenarios."""
    nominal = [sc for sc, _, _, n in SCENARIOS if n == 1.0]
    pts = [("rule", None, "Fixed rule"), ("staged_dr", None, "Staged rule"),
           ("neural_ftnom", [0, 1, 2], "Neural, weights x1"), ("neural_w5", None, "Neural, weights x5"),
           ("neural_w15", None, "Neural, weights x15"), ("neural_ftdr", None, "Neural, randomized")]
    keys = [("cv", 1, "CV of delivery"), ("violation_hours", 1, "Level violation (point-h)")]
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 66 * MM))
    out = {}
    for m, seeds, lab in pts:
        per = {}
        for sc in nominal:
            for r in pick(R, sc, m):
                if "failed_at_h" in r:
                    continue
                s_ = r["method"].rsplit("_s", 1)[-1] if "_s" in r["method"] else "0"
                if seeds is not None and int(s_) not in seeds:
                    continue
                per.setdefault(s_, []).append(r)
        if not per:
            continue
        ops = np.array([np.mean([r["check_moves"] + r["reg_moves"] for r in v]) for v in per.values()])
        out[m] = dict(label=lab, n=len(per), ops=(ops.mean(), ops.std()))
        for k, (ax, (key, sc_, yl)) in enumerate(zip(axes, keys)):
            y = np.array([np.mean([r[key] * sc_ for r in v]) for v in per.values()])
            out[m][key] = (y.mean(), y.std())
            col = ORANGE if m == "rule" else PURPLE if m.startswith("staged") else BLUE
            mk = "s" if m == "rule" else "v" if m.startswith("staged") else "o"
            mfc = col if m in ("rule", "staged_dr", "neural_ftdr") else "white"
            ax.errorbar(ops.mean(), y.mean(), xerr=ops.std() if len(per) > 1 else None,
                        yerr=y.std() if len(per) > 1 else None, color=col, marker=mk, ms=5, mfc=mfc,
                        lw=0.8, capsize=2, ls="none", zorder=3)
            off = {("staged_dr", 1): (6, -14), ("rule", 1): (6, 5), ("neural_w5", 0): (6, -12),
                   ("neural_ftnom", 1): (6, -12), ("staged_dr", 0): (6, -12)}.get((m, k), (5, 4))
            ax.annotate(lab, (ops.mean(), y.mean()), textcoords="offset points", xytext=off, fontsize=6.5,
                        color=INK2)
    for k, (ax, (key, sc_, yl)) in enumerate(zip(axes, keys)):
        xs = [out[m]["ops"][0] for m in ("neural_w15", "neural_w5", "neural_ftnom") if m in out]
        ys = [out[m][key][0] for m in ("neural_w15", "neural_w5", "neural_ftnom") if m in out]
        ax.plot(xs, ys, color=BLUE, lw=0.8, ls="--", zorder=2)
        ax.set_xlabel("Gate operations, 0–72 h (count, mean of 3 nominal scenarios)"); ax.set_ylabel(yl)
        if key == "violation_hours":
            ax.set_ylim(12, 105)
        ax.set_title(f"({'ab'[k]})", loc="left", fontsize=8, color=INK, pad=4)
    fig.tight_layout()
    save(fig, "Fig8_effort")
    return out


class Merged(dict):
    def __init__(self, *npz):
        super().__init__()
        for z in npz:
            for k in z.files:
                self[k] = z[k]


if __name__ == "__main__":
    R = json.load(open(result_file("results.json"))) + json.load(open(result_file("results_extra.json")))
    T = Merged(np.load(result_file("timeseries.npz")), np.load(result_file("timeseries_extra.npz")))
    RB = json.load(open(result_file("robustness.json"))) + json.load(open(result_file("robustness_extra.json")))
    fig_training()
    fig_shares(R)
    print("level panels:", fig_levels(R, T))
    fig_gates(R)
    fig_robustness(RB)
    print(json.dumps(fig_effort(R), indent=1, default=float))
    table(R)
