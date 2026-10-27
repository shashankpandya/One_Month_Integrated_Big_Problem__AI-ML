"""Figures for the synthetic data, AI/ML and case-study sections."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from make_figures import save, W, INK, INK2, MUTED, GRID, CAT, MECH_C, BLUES, STATUS
from psc import inputs as P, twin as TW, scenarios as SC, features as FE

MECHS = SC.MECHS
SHORT = {"PL": "Prestress loss", "CD": "Concrete det.", "CR": "Cracking", "CO": "Corrosion", "BR": "Bearing", "LS": "Local stiffness"}
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8312f"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])


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
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.2), sharex=True)
    panels = [("f1", "f1 change (%)"), ("f2", "f2 change (%)"), ("d", "load-test deflection (%)"),
              ("lev", "mid-span level (mm)"), ("vw", "VW strain (microstrain)"), ("brg", "bearing movement / healthy")]
    for ax, (k, t) in zip(axs.flat, panels):
        for m in MECHS:
            ax.plot(lev_s, R[m][k], color=MECH_C[m], lw=1.6, label=SHORT[m])
        ax.set_title(t, fontsize=8.5)
    axs[0, 0].legend(fontsize=6.5, ncol=1, loc="lower left")
    fig.supxlabel("severity (0 = healthy, 1 = top of its range)", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "f13_sensitivity")
    json.dump({m: {k: list(map(float, v)) for k, v in R[m].items()} for m in MECHS}, open("out/sensitivity.json", "w"), indent=1)


def fig_effect_size(res):
    names = res["effect_size"]["names"]
    M = np.array([res["effect_size"]["values"][m] for m in MECHS]).T
    M = np.clip(M, -8, 8)
    fig, ax = plt.subplots(figsize=(W * 0.72, 6.0))
    ax.imshow(M, cmap=DIV, vmin=-8, vmax=8, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(MECHS)), [SHORT[m] for m in MECHS], rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(names)), [FE.LABELS[n] for n in names], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= 1.5:
                ax.text(j, i, f"{M[i,j]:.0f}" if abs(M[i, j]) < 7.99 else (">8" if M[i, j] > 0 else "<-8"), ha="center",
                        va="center", fontsize=6.3, color="white" if abs(M[i, j]) > 5 else INK)
    ax.set_title("Effect size of each mechanism on each feature\n(difference from healthy mean / healthy s.d., upper-half severities)", fontsize=8.5)
    save(fig, "f14_effect_size")


def fig_distributions(rows, F, keys):
    feats = [("r_f1", "f1 change (%)"), ("r_d_mean", "load-test deflection change (%)"), ("lev_mean", "level change (mm)"),
             ("brg_min", "min bearing movement ratio"), ("HCP_min", "most negative HCP (mV)"), ("UPV_min", "min UPV (km/s)")]
    cls = ["H"] + list(MECHS)
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.3))
    for ax, (k, t) in zip(axs.flat, feats):
        data = [[f[k] for f, c in zip(F, keys) if c == cl] for cl in cls]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
        for patch, cl in zip(bp["boxes"], cls):
            patch.set_facecolor(MECH_C[cl]); patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(cls) + 1), cls, fontsize=7)
        ax.set_title(t, fontsize=8.5)
    fig.tight_layout()
    save(fig, "f15_distributions")


def fig_novelty(res):
    nov = res["novelty"]
    srcs = list(nov.keys())
    cats = ["false alarm"] + list(MECHS)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    x = np.arange(len(cats)); w = 0.16
    for i, s in enumerate(srcs):
        ax.bar(x + (i - 2) * w, [nov[s][c] for c in cats], w * 0.92, color=BLUES[i], label=s)
    ax.set_xticks(x, ["healthy\n(false alarm)"] + [SHORT[m] for m in MECHS], fontsize=6.6, rotation=30, ha="right")
    ax.set_ylabel("share of test cases flagged")
    ax.set_title("Novelty detection (trained on healthy data only)", fontsize=8.5)
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.set_ylim(0, 1.05)
    ax = axs[1]
    cv = res["novelty_curves"]
    for m in ("PL", "BR", "CR", "CO"):
        xs, ys = cv["All sources fused"][m]
        xv, yv = cv["Vibration only"][m]
        r = np.array(xs); r = (r - r.min()) / (r.max() - r.min() + 1e-12)
        rv = np.array(xv); rv = (rv - rv.min()) / (rv.max() - rv.min() + 1e-12)
        ax.plot(r, ys, "-o", color=MECH_C[m], ms=4, lw=1.5, label=f"{SHORT[m]}")
        ax.plot(rv, yv, ":", color=MECH_C[m], lw=1.3)
    ax.set_xlabel("relative severity within its range"); ax.set_ylabel("probability of detection")
    ax.set_title("Fused (solid) vs vibration only (dotted)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_ylim(-0.03, 1.05)
    fig.tight_layout()
    save(fig, "f16_novelty")


def fig_supervised(res):
    sup = res["supervised"]
    srcs = list(sup.keys())
    A = np.array([[sup[s][m]["AUC"] for m in MECHS] for s in srcs])
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    ax.imshow(A, cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(MECHS)), [SHORT[m] for m in MECHS], rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(srcs)), srcs, fontsize=7)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=6.8, color="white" if A[i, j] > 0.85 else INK)
    ax.set_title("ROC-AUC, random forest per mechanism", fontsize=8.5)
    ax = axs[1]
    f1 = [sup[s]["macro_F1"] for s in srcs]
    ax.barh(range(len(srcs)), f1, color=BLUES[:len(srcs)], height=0.6)
    for i, v in enumerate(f1):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(srcs)), [""] * len(srcs)); ax.invert_yaxis(); ax.set_xlim(0, 1.1)
    ax.set_title("macro-F1 (threshold 0.5)", fontsize=8.5)
    axs[0].set_ylim(len(srcs) - 0.5, -0.5)
    fig.tight_layout()
    save(fig, "f17_supervised")


def fig_confusion(res):
    labs = res["confusion_labels"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.0))
    for ax, s in zip(axs, ("Vibration only", "All sources fused")):
        M = np.array(res["confusion"][s], float)
        Mn = M / M.sum(axis=1, keepdims=True)
        ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=1); ax.grid(False)
        ax.set_xticks(range(len(labs)), labs, fontsize=7); ax.set_yticks(range(len(labs)), labs, fontsize=7)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if M[i, j] > 0:
                    ax.text(j, i, f"{int(M[i,j])}", ha="center", va="center", fontsize=6.5, color="white" if Mn[i, j] > 0.6 else INK)
        acc = np.trace(M) / M.sum()
        ax.set_title(f"{s} (accuracy {acc:.2f})", fontsize=8.5, loc="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout()
    save(fig, "f18_confusion")


def fig_importance(res):
    names = res["importance"]["names"]
    fig, axs = plt.subplots(2, 3, figsize=(W, 4.6))
    for ax, m in zip(axs.flat, MECHS):
        v = np.array(res["importance"]["values"][m])
        o = np.argsort(-v)[:6][::-1]
        ax.barh(range(6), v[o], color=MECH_C[m], height=0.6)
        ax.set_yticks(range(6), [FE.LABELS[names[i]] for i in o], fontsize=6.3)
        ax.set_title(SHORT[m], fontsize=8.5)
        ax.tick_params(axis="x", labelsize=6.5)
    for ax in axs[1]:
        ax.set_xlabel("AUC drop when shuffled", fontsize=7)
    fig.tight_layout()
    save(fig, "f19_importance")


def fig_pl_regression(res):
    a = np.load("out/ml_arrays.npz")
    y, mu, sd, ph = a["pl_y"], a["pl_mu"], a["pl_sd"], a["pl_phys"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9))
    ax = axs[0]
    ax.errorbar(y, mu, yerr=1.96 * sd, fmt="o", ms=2.8, color=CAT[0], ecolor="#b7d3f6", elinewidth=0.7, alpha=0.85, label="GPR mean +- 1.96 sd")
    ax.plot([0, 37], [0, 37], color=INK2, lw=0.8)
    ax.set_xlabel("true extra prestress loss (%)"); ax.set_ylabel("estimated (%)")
    g = res["pl_regression"]
    ax.set_title(f"GPR: MAE {g['gpr']['MAE']:.1f} %, R$^2$ {g['gpr']['R2']:.2f}, 95 % band covers {100*g['coverage95']:.0f} %", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax = axs[1]
    ax.plot(y, ph, "o", ms=2.8, color=MUTED, alpha=0.7, mec="none", label="levels + VW strain, worst gauged girder")
    ax.plot([0, 37], [0, 37], color=INK2, lw=0.8)
    ax.set_xlabel("true extra prestress loss (%)")
    ax.set_title(f"Physics inverse: MAE {g['levelling_only']['MAE']:.1f} %, R$^2$ {g['levelling_only']['R2']:.2f}", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
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
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.4), sharex=True)
    def ew(x, base):
        mu, sd = x[base].mean(), x[base].std()
        z = np.zeros_like(x); z[0] = x[0]
        for i in range(1, len(x)):
            z[i] = 0.2 * x[i] + 0.8 * z[i - 1]
        lim = 3 * sd * np.sqrt(0.2 / 1.8)
        return z, mu, lim
    base = t < 2
    panels = [(axs[0, 0], a["rf1"], "f1 residual after T / traffic normalisation (%)", "f1 residual"),
              (axs[0, 1], np.array([r["brg_B"] for r in mon]), "bearing movement, end B (mm/degC)", "bearing B movement"),
              (axs[1, 0], np.array([r["vw_G1"] for r in mon]), "VW strain change, G1 soffit (microstrain)", "VW strain G1"),
              (axs[1, 1], np.array([r["lev_G1"] for r in mon]), "level change, G1 mid-span (mm)", "level G1")]
    for ax, x, title, key in panels:
        z, mu, lim = ew(x, base)
        ax.plot(t, x, "o", ms=2.2, color=CAT[0], alpha=0.45, mec="none")
        ax.plot(t, z, color=CAT[0], lw=1.6)
        ax.axhline(mu + lim, color=STATUS["critical"], lw=0.7, ls="--"); ax.axhline(mu - lim, color=STATUS["critical"], lw=0.7, ls="--")
        fa = cs["trends"][key]["first_alarm_year"]
        if fa is not None:
            ax.axvline(fa, color=STATUS["serious"], lw=1.0)
            ax.text(fa + 0.2, ax.get_ylim()[1], f"alarm yr {fa:.1f}", fontsize=6.8, va="top", color=INK)
        ax.set_title(title, fontsize=8)
    if True:
        x = np.array([r["brg_A"] for r in mon])
        axs[0, 1].plot(t, x, "o", ms=2.2, color=CAT[1], alpha=0.45, mec="none")
        axs[0, 1].plot(t, ew(x, base)[0], color=CAT[1], lw=1.6)
        axs[0, 1].text(t[-1] - 0.3, 0.335, "end A", color=INK, fontsize=7, ha="right")
        axs[0, 1].text(t[-1] - 0.3, 0.05, "end B", color=INK, fontsize=7, ha="right")
    for ax in axs.flat:
        for k, v in on.items():
            ax.axvline(v, color=GRID, lw=2.5, zorder=0)
    axs[1, 0].set_xlabel("age (years)"); axs[1, 1].set_xlabel("age (years)")
    axs[0, 0].text(on["joint"] + 0.1, axs[0, 0].get_ylim()[0], "joint fails", fontsize=6.5, color=MUTED, va="bottom")
    fig.tight_layout()
    save(fig, "f21_case_trends")


def fig_case_ndt(cs):
    import csv as _csv
    rows = list(_csv.DictReader(open("out/data/case_ndt_survey.csv")))
    cl = sorted(cs["ndt_clusters"]["clusters"], key=lambda c: -c["hcp"])
    names = [c["name"] for c in cl]
    col = {n: c for n, c in zip(names, [CAT[0], CAT[1], CAT[2], CAT[6], CAT[4]])}
    fig = plt.figure(figsize=(W, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0]); ax.grid(False)
    for si, sp in enumerate(("S1", "S2", "S3")):
        for r in rows:
            if r["span"] != sp:
                continue
            gi = int(r["girder"][1:]) - 1; zi = int(r["zone"][1:]) - 1
            ax.add_patch(plt.Rectangle((si * 5.6 + zi, 5 - gi), 0.92, 0.85, color=col[r["cluster"]]))
        ax.text(si * 5.6 + 2.5, 6.25, sp, ha="center", fontsize=8, color=INK)
    for gi in range(6):
        ax.text(-0.25, 5 - gi + 0.42, f"G{gi+1}", ha="right", va="center", fontsize=7)
    ax.set_xlim(-0.9, 16.8); ax.set_ylim(-0.3, 6.7); ax.set_axis_off()
    ax.set_title("Year-18 NDT: Gaussian-mixture clusters", fontsize=8.5)
    hand = [plt.Rectangle((0, 0), 1, 1, color=col[n]) for n in names]
    ax.legend(hand, names, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    for r in rows:   # ASTM C876 band on top of the clusters: ring the areas with HCP < -350 mV
        if float(r["hcp"]) < -350:
            si = ("S1", "S2", "S3").index(r["span"]); gi = int(r["girder"][1:]) - 1; zi = int(r["zone"][1:]) - 1
            ax.add_patch(plt.Rectangle((si * 5.6 + zi + 0.06, 5 - gi + 0.06), 0.8, 0.73, fill=False, ec=INK, lw=1.2))
    ax.text(8.4, -0.35, "black outline: HCP < -350 mV (corrosion likely, ASTM C876)", fontsize=6.5, ha="center", color=INK2)
    ax2 = fig.add_subplot(gs[1])
    for n in names:
        k = [r for r in rows if r["cluster"] == n]
        ax2.plot([float(r["resistivity"]) for r in k], [float(r["hcp"]) for r in k], "o", ms=3.5, color=col[n], mec="white", mew=0.4, label=n)
    ax2.set_xscale("log")
    ax2.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"]); ax2.minorticks_off()
    ax2.axhline(-350, color=STATUS["critical"], lw=0.7, ls="--"); ax2.axhline(-200, color=MUTED, lw=0.7, ls="--")
    ax2.text(80, -345, "-350 mV", fontsize=6.5, color=INK2, ha="right"); ax2.text(80, -195, "-200 mV", fontsize=6.5, color=INK2, ha="right")
    ax2.set_xlabel("resistivity (kOhm cm)"); ax2.set_ylabel("half-cell potential (mV CSE)")
    ax2.set_title("HCP against resistivity", fontsize=8.5)
    fig.tight_layout()
    save(fig, "f22_case_ndt")


def fig_case_ml(cs):
    ml = cs["ml"]
    truth = {m["type"] for m in cs["truth"]}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]
    x = np.arange(len(MECHS))
    ax.bar(x - 0.18, [ml["p_vibration"][m] for m in MECHS], 0.34, color=BLUES[0], label="vibration only")
    ax.bar(x + 0.18, [ml["p_fused"][m] for m in MECHS], 0.34, color=BLUES[3], label="all sources fused")
    ax.axhline(0.5, color=INK2, lw=0.7, ls="--")
    for i, m in enumerate(MECHS):
        if m in truth:
            ax.text(i, 1.06, "true", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, [SHORT[m] for m in MECHS], rotation=25, ha="right", fontsize=7)
    ax.set_ylim(0, 1.15); ax.set_ylabel("probability")
    ax.set_title("Year-18 campaign: mechanism probabilities", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left")
    ax = axs[1]
    d = cs["campaign_girders"]["d"]; e = cs["campaign_girders"]["e"]
    xx = np.arange(6)
    ax.bar(xx - 0.18, d, 0.34, color=CAT[0], label="deflection")
    ax.bar(xx + 0.18, e, 0.34, color=CAT[1], label="soffit strain")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(xx, [f"G{i+1}" for i in range(6)])
    ax.set_ylabel("change from commissioning (%)")
    ax.set_title("Load test, year 18 vs year 0", fontsize=8.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "f23_case_ml")


def run_case():
    cs = json.load(open("out/case_study.json"))
    fig_case_trends(cs); fig_case_ndt(cs); fig_case_ml(cs)
