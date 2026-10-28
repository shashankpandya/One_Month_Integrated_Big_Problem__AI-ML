"""Figures for the synthetic data, AI/ML and case-study chapters (same style as make_figures.py)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from make_figures import save, W, INK, INK2, MUTED, GRID, CAT, MECH_C, BLUES, RED
from psc import inputs as P, twin as TW, scenarios as SC, features as FE

MECHS = SC.MECHS
SHORT = {"PL": "Prestress loss", "CD": "Concrete det.", "CR": "Cracking", "CO": "Corrosion", "BR": "Bearing",
         "LS": "Local stiffness"}
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8312f"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
SRC_SHORT = {"Vibration only": "Vibration only", "Static + geometry + bearings": "Static + geometry + bearings",
             "Inspection + NDT": "Inspection + NDT", "Continuous monitoring (no closure)": "Continuous monitoring",
             "All sources fused": "All sources fused"}


# ---------------------------------------------------------------- physics: what each mechanism does
def fig_sensitivity():
    b = json.load(open("out/baseline.json"))["twin"]
    tw = TW.Twin(E_scale=b["E_scale"], barrier=b["barrier"], E_dyn=b["E_dyn"])
    f0, _ = tw.modal(TW.healthy_state()); d0, e0 = tw.load_test(TW.healthy_state())
    s0 = tw.bearings(TW.healthy_state())
    lev_s = np.linspace(0, 1, 7)

    def state(m, s):
        st = TW.healthy_state()
        if m == "PL": st["eta"][:] = 1 - 0.35 * s
        if m == "CD": st["conc"].append((0, 0, P.L, 0.30 * s))
        if m == "CR": st["stiff"].append((0, 13.5, 19.5, 0.5 * s, 0.175 * s))
        if m == "CO":
            st["strand"].append((0, 13.5, 19.5, 0.25 * s)); st["stiff"].append((0, 13.5, 19.5, 0.0625 * s, 0.0625 * s))
        if m == "BR": st["kh_mult"][:, :] = 10 ** (2.7 * s)
        if m == "LS": st["stiff"].append((0, 7.25, 9.25, 0.6 * s, 0.6 * s))
        return st
    R = {m: dict(f1=[], f2=[], d=[], lev=[], vw=[], brg=[]) for m in MECHS}
    for m in MECHS:
        for s in lev_s:
            st = state(m, s)
            f, _ = tw.modal(st); d, e = tw.load_test(st); lev, vw = tw.long_term(st); br = tw.bearings(st)
            R[m]["f1"].append(100 * (f[0] / f0[0] - 1)); R[m]["f2"].append(100 * (f[1] / f0[1] - 1))
            R[m]["d"].append(100 * (d.mean() / d0.mean() - 1)); R[m]["lev"].append(lev.mean())
            R[m]["vw"].append(vw.mean()); R[m]["brg"].append(br[0].min() / s0[0, 0])
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.9), sharex=True)
    panels = [("f1", "(a) change of $f_1$ (%)"), ("f2", "(b) change of $f_2$ (%)"), ("d", "(c) load-test deflection (%)"),
              ("lev", "(d) mid-span level (mm)"), ("vw", r"(e) VW strain ($\mu\varepsilon$)"), ("brg", "(f) bearing movement ratio")]
    for ax, (k, t) in zip(axs.flat, panels):
        for m in MECHS:
            ax.plot(lev_s, R[m][k], color=MECH_C[m], lw=1.6, label=SC.MECH_NAMES[m])
        ax.set_title(t, fontsize=10)
    for ax in axs[1]:
        ax.set_xlabel("relative severity")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "f13_sensitivity")
    json.dump({m: {k: list(map(float, v)) for k, v in R[m].items()} for m in MECHS}, open("out/sensitivity.json", "w"), indent=1)


def fig_effect_size(res):
    names = res["effect_size"]["names"]
    M = np.array([res["effect_size"]["values"][m] for m in MECHS]).T
    M = np.clip(M, -8, 8)
    fig, ax = plt.subplots(figsize=(5.6, 8.0))
    ax.imshow(M, cmap=DIV, vmin=-8, vmax=8, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(MECHS)), [SHORT[m] for m in MECHS], rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(names)), [FE.LABELS[n] for n in names], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= 1.5:
                ax.text(j, i, f"{M[i,j]:.0f}" if abs(M[i, j]) < 7.99 else (">8" if M[i, j] > 0 else "<-8"), ha="center",
                        va="center", fontsize=8.5, color="white" if abs(M[i, j]) > 5 else INK)
    save(fig, "f14_effect_size")


def fig_distributions(rows, F, keys):
    feats = [("r_f1", "(a) change of $f_1$ (%)"), ("r_d_mean", "(b) load-test deflection (%)"), ("lev_mean", "(c) level change (mm)"),
             ("brg_min", "(d) bearing movement ratio"), ("HCP_min", "(e) most negative HCP (mV)"), ("UPV_min", "(f) lowest UPV (km/s)")]
    cls = ["H"] + list(MECHS)
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.8))
    for ax, (k, t) in zip(axs.flat, feats):
        data = [[f[k] for f, c in zip(F, keys) if c == cl] for cl in cls]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
        for patch, cl in zip(bp["boxes"], cls):
            patch.set_facecolor(MECH_C[cl]); patch.set_edgecolor(INK2); patch.set_linewidth(0.6)
        ax.set_xticks(range(1, len(cls) + 1), cls, fontsize=9)
        ax.set_title(t, fontsize=10)
    fig.tight_layout()
    save(fig, "f15_distributions")


def fig_novelty(res):
    nov = res["novelty"]
    srcs = list(nov.keys())
    cats = ["false alarm"] + list(MECHS)
    fig, axs = plt.subplots(2, 1, figsize=(W, 6.2), gridspec_kw=dict(height_ratios=[1.15, 1]))
    ax = axs[0]
    x = np.arange(len(cats)); w = 0.16
    for i, s in enumerate(srcs):
        ax.bar(x + (i - 2) * w, [nov[s][c] for c in cats], w * 0.92, color=BLUES[i], label=SRC_SHORT[s])
    ax.set_xticks(x, ["Healthy\n(false alarm)"] + [SHORT[m].replace(" ", "\n", 1) for m in MECHS], fontsize=9.5)
    ax.set_ylabel("share of test cases flagged")
    ax.set_title("(a) Detection rate by data source")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax = axs[1]
    cv = res["novelty_curves"]
    for m in ("PL", "BR", "CR", "CO"):
        xs, ys = cv["All sources fused"][m]
        xv, yv = cv["Vibration only"][m]
        r = np.array(xs); r = (r - r.min()) / (r.max() - r.min() + 1e-12)
        rv = np.array(xv); rv = (rv - rv.min()) / (rv.max() - rv.min() + 1e-12)
        ax.plot(r, ys, "-o", color=MECH_C[m], ms=4.5, lw=1.5, label=SC.MECH_NAMES[m])
        ax.plot(rv, yv, ":", color=MECH_C[m], lw=1.5)
    ax.set_xlabel("relative severity within the range of the mechanism"); ax.set_ylabel("probability of detection")
    ax.set_title("(b) All sources fused (solid lines) and vibration only (dotted lines)")
    ax.legend(loc="center right", fontsize=9); ax.set_ylim(-0.03, 1.05)
    fig.tight_layout(h_pad=1.0)
    save(fig, "f16_novelty")


def fig_supervised(res):
    sup = res["supervised"]
    srcs = list(sup.keys())
    A = np.array([[sup[s][m]["AUC"] for m in MECHS] for s in srcs])
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw=dict(width_ratios=[1.75, 1]))
    ax = axs[0]
    ax.imshow(A, cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(MECHS)), [SHORT[m] for m in MECHS], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(srcs)), [SRC_SHORT[s] for s in srcs], fontsize=9)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=9, color="white" if A[i, j] > 0.85 else INK)
    ax.set_title("(a) ROC-AUC")
    ax = axs[1]
    f1 = [sup[s]["macro_F1"] for s in srcs]
    ax.barh(range(len(srcs)), f1, color=BLUES[:len(srcs)], height=0.6)
    for i, v in enumerate(f1):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(srcs)), [""] * len(srcs)); ax.invert_yaxis(); ax.set_xlim(0, 1.15)
    ax.set_title("(b) Macro-F1")
    fig.tight_layout(w_pad=0.8)
    save(fig, "f17_supervised")


def fig_confusion(res):
    labs = res["confusion_labels"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.4))
    for ax, s, tag in zip(axs, ("Vibration only", "All sources fused"), ("(a)", "(b)")):
        M = np.array(res["confusion"][s], float)
        Mn = M / M.sum(axis=1, keepdims=True)
        ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=1); ax.grid(False)
        ax.set_xticks(range(len(labs)), labs, fontsize=9.5); ax.set_yticks(range(len(labs)), labs, fontsize=9.5)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if M[i, j] > 0:
                    ax.text(j, i, f"{int(M[i,j])}", ha="center", va="center", fontsize=8.5, color="white" if Mn[i, j] > 0.6 else INK)
        acc = np.trace(M) / M.sum()
        ax.set_title(f"{tag} {s}, accuracy {acc:.2f}", fontsize=10)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(w_pad=1.5)
    save(fig, "f18_confusion")


def fig_importance(res):
    names = res["importance"]["names"]
    fig, axs = plt.subplots(3, 2, figsize=(W, 6.6))
    for k, (ax, m) in enumerate(zip(axs.flat, MECHS)):
        v = np.array(res["importance"]["values"][m])
        o = np.argsort(-v)[:5][::-1]
        ax.barh(range(5), v[o], color=MECH_C[m], height=0.6)
        ax.set_yticks(range(5), [FE.LABELS[names[i]] for i in o], fontsize=9)
        ax.set_title(f"({'abcdef'[k]}) {SC.MECH_NAMES[m]}", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.xaxis.get_major_formatter().set_useOffset(False)
    for ax in axs[2]:
        ax.set_xlabel("drop in AUC when the feature is shuffled", fontsize=9.5)
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    save(fig, "f19_importance")


def fig_pl_regression(res):
    a = np.load("out/ml_arrays.npz")
    y, mu, sd, ph = a["pl_y"], a["pl_mu"], a["pl_sd"], a["pl_phys"]
    g = res["pl_regression"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True)
    ax = axs[0]
    ax.errorbar(y, mu, yerr=1.96 * sd, fmt="o", ms=3, color=CAT[0], ecolor="#b7d3f6", elinewidth=0.7, alpha=0.9,
                label="mean ± 1.96 s.d.")
    ax.plot([0, 37], [0, 37], color=INK2, lw=0.8)
    ax.set_xlabel("true extra prestress loss (%)"); ax.set_ylabel("estimated loss (%)")
    ax.set_title(f"(a) Gaussian process (MAE {g['gpr']['MAE']:.1f} %)")
    ax.legend(loc="upper left", fontsize=9)
    ax = axs[1]
    ax.plot(y, ph, "o", ms=3, color=MUTED, alpha=0.8, mec="none")
    ax.plot([0, 37], [0, 37], color=INK2, lw=0.8)
    ax.set_xlabel("true extra prestress loss (%)")
    ax.set_title(f"(b) Physical inversion (MAE {g['levelling_only']['MAE']:.1f} %)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "f20_pl_regression")


def run():
    res = json.load(open("out/ml_results.json"))
    rows = FE.read_csv("out/data/library.csv")
    from run_04_ml import class_key
    keys = [class_key(r) for r in rows]
    norm = FE.Normaliser().fit([r for r, k in zip(rows, keys) if k == "H"])
    F = [norm.transform(r) for r in rows]
    fig_sensitivity(); fig_effect_size(res); fig_distributions(rows, F, keys); fig_novelty(res)
    fig_supervised(res); fig_confusion(res); fig_importance(res); fig_pl_regression(res)


# ---------------------------------------------------------------- case study
def fig_case_trends(cs):
    mon = FE.read_csv("out/data/case_monitoring.csv")
    a = np.load("out/case_arrays.npz")
    t = a["tm"]
    on = cs["onset"]
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.0), sharex=True)

    def ew(x, base):
        mu, sd = x[base].mean(), x[base].std()
        z = np.zeros_like(x); z[0] = x[0]
        for i in range(1, len(x)):
            z[i] = 0.2 * x[i] + 0.8 * z[i - 1]
        return z, mu, 3 * sd * np.sqrt(0.2 / 1.8)
    base = t < 2
    panels = [(axs[0, 0], a["rf1"], "(a) $f_1$ residual (%)", "f1 residual"),
              (axs[0, 1], np.array([r["brg_B"] for r in mon]), "(b) bearing movement (mm/°C)", "bearing B movement"),
              (axs[1, 0], np.array([r["vw_G1"] for r in mon]), r"(c) VW strain change, G1 ($\mu\varepsilon$)", "VW strain G1"),
              (axs[1, 1], np.array([r["lev_G1"] for r in mon]), "(d) level change, G1 mid-span (mm)", "level G1")]
    for ax, x, title, key in panels:
        for v in on.values():
            ax.axvline(v, color="#e3e3e3", lw=3, zorder=0)
        z, mu, lim = ew(x, base)
        ax.plot(t, x, "o", ms=2.3, color=CAT[0], alpha=0.45, mec="none")
        ax.plot(t, z, color=CAT[0], lw=1.6)
        ax.axhline(mu + lim, color=RED, lw=0.8, ls="--"); ax.axhline(mu - lim, color=RED, lw=0.8, ls="--")
        fa = cs["trends"][key]["first_alarm_year"]
        if fa is not None:
            ax.axvline(fa, color=INK2, lw=0.9, ls="-.")
        ax.set_title(title, fontsize=10)
    x = np.array([r["brg_A"] for r in mon])
    axs[0, 1].plot(t, x, "o", ms=2.3, color=CAT[1], alpha=0.45, mec="none")
    axs[0, 1].plot(t, ew(x, base)[0], color=CAT[1], lw=1.6)
    axs[0, 1].text(t[-1] - 0.3, 0.29, "end A", fontsize=9.5, ha="right", va="top")
    axs[0, 1].text(t[-1] - 0.3, 0.06, "end B", fontsize=9.5, ha="right")
    axs[1, 0].set_xlabel("age of the bridge (years)"); axs[1, 1].set_xlabel("age of the bridge (years)")
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    save(fig, "f21_case_trends")


def fig_case_ndt(cs):
    import csv as _csv
    rows = list(_csv.DictReader(open("out/data/case_ndt_survey.csv")))
    cl = sorted(cs["ndt_clusters"]["clusters"], key=lambda c: -c["hcp"])
    names = [c["name"] for c in cl]
    col = {n: c for n, c in zip(names, [CAT[0], CAT[1], CAT[2], CAT[6], CAT[4]])}
    fig, axs = plt.subplots(2, 1, figsize=(W, 6.0), gridspec_kw=dict(height_ratios=[1, 1.1]))
    ax = axs[0]; ax.grid(False)
    for si, sp in enumerate(("S1", "S2", "S3")):
        for r in rows:
            if r["span"] != sp:
                continue
            gi = int(r["girder"][1:]) - 1; zi = int(r["zone"][1:]) - 1
            ax.add_patch(plt.Rectangle((si * 5.6 + zi, 5 - gi), 0.92, 0.85, color=col[r["cluster"]]))
            if float(r["hcp"]) < -350:
                ax.add_patch(plt.Rectangle((si * 5.6 + zi + 0.07, 5 - gi + 0.07), 0.78, 0.71, fill=False, ec=INK, lw=1.4))
        ax.text(si * 5.6 + 2.45, 6.2, f"Span {sp}", ha="center", fontsize=10)
        for zi in range(5):
            ax.text(si * 5.6 + zi + 0.46, -0.3, f"Z{zi+1}", ha="center", fontsize=8.5)
    for gi in range(6):
        ax.text(-0.2, 5 - gi + 0.42, f"G{gi+1}", ha="right", va="center", fontsize=9.5)
    ax.set_xlim(-1.0, 16.6); ax.set_ylim(-0.6, 6.6); ax.set_axis_off()
    ax.set_title("(a) Clusters of the 90 test areas (black outline: HCP below $-$350 mV)", fontsize=10)
    hand = [plt.Rectangle((0, 0), 1, 1, color=col[n]) for n in names]
    ax.legend(hand, names, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9.5)
    ax2 = axs[1]
    for n in names:
        k = [r for r in rows if r["cluster"] == n]
        ax2.plot([float(r["resistivity"]) for r in k], [float(r["hcp"]) for r in k], "o", ms=4.5, color=col[n], mec="white",
                 mew=0.4, label=n)
    ax2.set_xscale("log")
    ax2.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"]); ax2.minorticks_off()
    ax2.axhline(-350, color=RED, lw=0.8, ls="--"); ax2.axhline(-200, color=MUTED, lw=0.8, ls="--")
    ax2.text(9.5, -340, "$-$350 mV", fontsize=9); ax2.text(9.5, -190, "$-$200 mV", fontsize=9)
    ax2.set_xlabel("resistivity (k$\\Omega$cm)"); ax2.set_ylabel("half-cell potential (mV, CSE)")
    ax2.set_title("(b) Half-cell potential against resistivity", fontsize=10)
    ax2.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout(h_pad=1.2)
    save(fig, "f22_case_ndt")


def fig_case_ml(cs):
    ml = cs["ml"]
    truth = {m["type"] for m in cs["truth"]}
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    x = np.arange(len(MECHS))
    ax.bar(x - 0.19, [ml["p_vibration"][m] for m in MECHS], 0.36, color=BLUES[0], label="vibration only")
    ax.bar(x + 0.19, [ml["p_fused"][m] for m in MECHS], 0.36, color=BLUES[3], label="all sources fused")
    ax.axhline(0.5, color=INK2, lw=0.7, ls="--")
    for i, m in enumerate(MECHS):
        if m in truth:
            ax.text(i, 1.05, "*", ha="center", fontsize=13)
    ax.set_xticks(x, [SHORT[m] for m in MECHS], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.16); ax.set_ylabel("probability")
    ax.set_title("(a) Probability of each mechanism")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=9)      # below the * markers
    ax = axs[1]
    d = cs["campaign_girders"]["d"]; e = cs["campaign_girders"]["e"]
    xx = np.arange(6)
    ax.bar(xx - 0.19, d, 0.36, color=CAT[0], label="deflection")
    ax.bar(xx + 0.19, e, 0.36, color=CAT[1], label="soffit strain")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(xx, [f"G{i+1}" for i in range(6)])
    ax.set_ylabel("change from year 0 (%)")
    ax.set_title("(b) Load test, year 18")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(-9, 11)
    fig.tight_layout(w_pad=1.0)
    save(fig, "f23_case_ml")


def run_case():
    cs = json.load(open("out/case_study.json"))
    fig_case_trends(cs); fig_case_ndt(cs); fig_case_ml(cs)
