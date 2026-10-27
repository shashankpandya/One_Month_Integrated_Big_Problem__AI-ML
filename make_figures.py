"""
Figures for the report (out/figs/*.png). Run after steps 1-5.
    python make_figures.py [design|baseline|ml|case|all]
Figures are drawn at the size they are printed (text width 6.4 in), so the
font sizes below are the printed sizes.
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from psc import inputs as P, section as S, prestress as PS, dynamics as DY
from psc import features as FE, scenarios as SC

INK, INK2, MUTED, GRID = "#000000", "#333333", "#777777", "#d4d4d4"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MECH_C = {"H": "#8a8a8a", "PL": CAT[0], "CD": CAT[1], "CR": CAT[2], "CO": CAT[3], "BR": CAT[4], "LS": CAT[5]}
BLUES = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
RED = "#c62828"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "axes.titleweight": "normal",
    "axes.grid": True, "grid.color": GRID, "grid.linestyle": ":", "grid.linewidth": 0.7,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8, "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": "#aaaaaa", "legend.fancybox": False, "lines.linewidth": 1.5, "savefig.dpi": 200,
    "xtick.direction": "in", "ytick.direction": "in", "axes.titlepad": 5,
})
W = 6.4


def save(fig, name):
    fig.savefig(f"out/figs/{name}.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("  ", name)


# ================================================================ design
def fig_cross_section():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.grid(False); ax.set_axis_off()
    D, t = P.GIRDER_DEPTH, P.SLAB_T
    out = np.array(S.girder_outline())
    for y0 in P.GIRDER_Y:
        ax.add_patch(Polygon(out + [y0, 0], closed=True, fc="#e8e8e8", ec=INK2, lw=0.8))
    ax.add_patch(Rectangle((-8, D), 16, t, fc="#cfcfcf", ec=INK2, lw=0.8))
    for i in range(len(P.GIRDER_Y) - 1):
        a, b = P.GIRDER_Y[i] + P.WEB_T / 2, P.GIRDER_Y[i + 1] - P.WEB_T / 2
        ax.add_patch(Rectangle((a, D + t - 0.225 - P.CROSS_D), b - a, P.CROSS_D, fc="none", ec=MUTED, lw=0.6, ls="--"))
    ax.add_patch(Rectangle((-P.KERB_Y, D + t), 2 * P.KERB_Y, P.WEARING_T * 1.6, fc="#555555", ec="none"))
    for s in (-1, 1):
        y0 = s * P.KERB_Y
        bar = np.array([[0, 0], [0.5, 0], [0.5, 0.075], [0.325, 0.25], [0.325, 0.9], [0.15, 0.9], [0.0, 0.0]])
        if s < 0:
            bar[:, 0] = -bar[:, 0]
        ax.add_patch(Polygon(bar + [y0, D + t], closed=True, fc="#bdbdbd", ec=INK2, lw=0.7))
        fp0 = s * (P.KERB_Y + P.CRASH_BARRIER_W)
        ax.add_patch(Rectangle((min(fp0, fp0 + s * P.FOOTPATH_W), D + t), P.FOOTPATH_W, 0.25, fc="#e0e0e0", ec=INK2, lw=0.7))
        yr = s * (P.DECK_WIDTH / 2 - 0.125)
        ax.plot([yr, yr], [D + t + 0.25, D + t + 1.35], color=INK2, lw=1.2)
        ax.plot([yr - 0.12, yr + 0.12], [D + t + 1.35] * 2, color=INK2, lw=1.2)
        ax.text(s * (P.KERB_Y + 0.5 + P.FOOTPATH_W / 2), D + t + 0.38, "footpath", ha="center", fontsize=9)
    for k in range(3):
        ax.text(-P.KERB_Y + 1.833 + 3.667 * k, D + t + 0.42, f"Lane {k+1}", ha="center", fontsize=9.5)

    def dim(y1, y2, z, txt):
        ax.annotate("", (y1, z), (y2, z), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
        ax.text((y1 + y2) / 2, z + 0.08, txt, ha="center", va="bottom", fontsize=9.5)
    dim(-8, 8, 4.15, "16.000 m")
    dim(-P.KERB_Y, P.KERB_Y, 3.45, "carriageway 11.000 m")
    dim(P.KERB_Y + 0.5, P.KERB_Y + 0.5 + P.FOOTPATH_W, 3.45, "1.750")
    dim(-P.KERB_Y - 0.5 - P.FOOTPATH_W, -P.KERB_Y - 0.5, 3.45, "1.750")
    dim(P.GIRDER_Y[0], P.GIRDER_Y[-1], -0.5, "5 @ 2.700 = 13.500 m")
    dim(P.GIRDER_Y[-1], 8.0, -0.5, "1.250")
    dim(-8.0, P.GIRDER_Y[0], -0.5, "1.250")
    for i, y0 in enumerate(P.GIRDER_Y):
        ax.text(y0, 0.75, f"G{i+1}", ha="center", fontsize=10, weight="bold")
    ax.set_xlim(-8.5, 8.5); ax.set_ylim(-0.75, 4.6); ax.set_aspect("equal")
    save(fig, "f01_cross_section")


def fig_girder_section():
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.6))
    cab = PS.make_cables()
    for ax, eb, title in ((axs[0], False, "(a) Mid-span section"), (axs[1], True, "(b) End block at the anchorage face")):
        ax.grid(False); ax.set_aspect("equal"); ax.set_axis_off()
        out = np.array(S.girder_outline(eb))
        ax.add_patch(Polygon(out, closed=True, fc="#e8e8e8", ec=INK2, lw=0.9))
        ax.add_patch(Rectangle((-0.75, P.GIRDER_DEPTH), 1.5, P.SLAB_T, fc="#cfcfcf", ec=INK2, lw=0.8))
        for c in cab:
            if eb:
                y, z = 0.0, c.z_a
                ax.add_patch(Rectangle((y - 0.135, z - 0.135), 0.27, 0.27, fc="none", ec=CAT[0], lw=0.9))
            else:
                y, z = c.y_m, c.z_m
            ax.add_patch(plt.Circle((y, z), P.DUCT_OD / 2, fc="white", ec=CAT[0], lw=1.2))
            ax.text(y + (0.17 if eb else 0.0), z + (0 if eb else 0.085), c.name, ha="left" if eb else "center",
                    va="center", fontsize=9)
        ax.set_xlim(-0.95, 0.95); ax.set_ylim(-0.32, 2.12)
        ax.set_title(title, fontsize=10.5)
        sec = S.precast(eb)
        ax.plot([-0.45, 0.7], [sec["zc"]] * 2, color=MUTED, lw=0.6, ls="--")
        ax.text(0.93, sec["zc"] + 0.03, f"c.g. {sec['zc']*1e3:.0f}", fontsize=9, ha="right")
        ax.annotate("", (-0.55, 0), (-0.55, P.GIRDER_DEPTH), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
        ax.text(-0.59, 0.45, "1800", rotation=90, va="center", ha="right", fontsize=9)
        ax.text(0, -0.28, f"A = {sec['A']:.3f} m$^2$,  I = {sec['I']:.4f} m$^4$", ha="center", fontsize=9)
    axs[0].text(0, 1.87, "flange 1000 x 175", ha="center", fontsize=9)
    axs[0].text(0.19, 1.2, "web 300", fontsize=9)
    axs[0].text(0, -0.12, "bulb 750 x 250", ha="center", fontsize=9)
    axs[1].text(0, -0.12, "anchor plates 270 x 270 at 400 c/c", ha="center", fontsize=9)
    fig.tight_layout(w_pad=0.5)
    save(fig, "f02_girder_section")


def fig_cable_profile():
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.0), gridspec_kw=dict(height_ratios=[2, 1]), sharex=True)
    x = np.linspace(0, P.L_GIRDER, 600)
    cab = PS.make_cables()
    ax = axs[0]
    ax.axvspan(0, P.END_BLOCK_LEN, color="#ececec", zorder=0); ax.axvspan(P.L_GIRDER - P.END_BLOCK_LEN, P.L_GIRDER, color="#ececec", zorder=0)
    for i, c in enumerate(cab):
        ax.plot(x, c.z(x), color=CAT[i], lw=1.6, label=f"{c.name} (stage {c.stage})")
    for xb in (P.END_OVERHANG, P.L_GIRDER - P.END_OVERHANG):
        ax.plot([xb], [-0.04], marker="^", color=INK2, ms=7, clip_on=False)
    ax.set_ylim(-0.08, 1.85); ax.set_ylabel("height above soffit (m)")
    ax.set_title("(a) Elevation (end blocks shaded, bearings marked)")
    ax.legend(ncol=4, loc="upper center", fontsize=9)
    ax = axs[1]
    for i, c in enumerate(cab):
        ax.plot(x, c.y(x), color=CAT[i], lw=1.6)
    ax.axhline(P.WEB_T / 2, color=MUTED, lw=0.7, ls="--"); ax.axhline(-P.WEB_T / 2, color=MUTED, lw=0.7, ls="--")
    ax.text(0.5, P.WEB_T / 2 + 0.025, "web face", fontsize=9, color=INK2)
    ax.set_ylabel("lateral (m)"); ax.set_xlabel("distance from girder end (m)")
    ax.set_title("(b) Plan (lateral swing inside the bottom bulb)")
    ax.set_ylim(-0.3, 0.3)
    fig.tight_layout(h_pad=0.6)
    save(fig, "f03_cable_profile")


def fig_prestress_force():
    a = np.load("out/design_arrays.npz")
    x = a["xg_full"]
    names = [c.name for c in PS.make_cables()]
    tot = lambda pre: sum(a[f"{pre}_{n}"] for n in names) / 1e3
    fig, ax = plt.subplots(figsize=(W, 3.0))
    series = [("Pj", "jacking", INK2, "--"), ("Pf", "after friction", CAT[1], "-"), ("Pfs", "after draw-in", CAT[3], "-"),
              ("P0", "after elastic shortening, $P_0$", CAT[0], "-"), ("Pe", "after all losses, $P_e$", CAT[6], "-")]
    for k, lab, col, ls in series:
        ax.plot(x, tot(k), color=col, ls=ls, label=lab, lw=1.8 if k in ("P0", "Pe") else 1.4)
    d = json.load(open("out/design.json"))["prestress"]
    ax.annotate(f"$P_0$ = {d['P0_mid']/1e3:.2f} MN", (P.L_GIRDER / 2, d["P0_mid"] / 1e3), (P.L_GIRDER / 2 + 2.5, d["P0_mid"] / 1e3 + 0.45),
                fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate(f"$P_e$ = {d['Pe_mid']/1e3:.2f} MN", (P.L_GIRDER / 2, d["Pe_mid"] / 1e3), (P.L_GIRDER / 2 + 2.5, d["Pe_mid"] / 1e3 + 0.45),
                fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("distance from girder end (m)"); ax.set_ylabel("force of the four cables (MN)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9.5)
    ax.set_xlim(0, P.L_GIRDER); ax.set_ylim(6.6, 9.6)
    save(fig, "f04_prestress_force")


def fig_envelopes():
    a = np.load("out/design_arrays.npz")
    x = a["XG"]
    full = lambda h: np.concatenate([h, h[-2::-1]])
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.1))
    iy = 0
    Mdl = a["Mg"][iy] + a["Ms"][iy]
    Msidl = a["Medge"][iy] + a["Mwc"][iy]
    Mll = full(a["MLL"][iy]) + np.maximum(a["MfpL"][iy], 0) + np.maximum(a["MfpR"][iy], 0)
    Mu = P.ULS["g"] * Mdl + P.ULS["sidl"] * a["Medge"][iy] + P.ULS["wc"] * a["Mwc"][iy] + P.ULS["q"] * Mll
    labels = ("girder + slab", "SIDL", "live load", "ULS")
    for y, lab, col in zip((Mdl, Msidl, Mll, Mu), labels, (CAT[0], CAT[1], CAT[2], INK)):
        axs[0].plot(x, y / 1e3, color=col, label=lab, lw=1.9 if lab == "ULS" else 1.5)
    axs[0].set_title("(a) Bending moment"); axs[0].set_xlabel("distance from bearing (m)"); axs[0].set_ylabel("moment (MN m)")
    Vdl = a["Vg"][iy] + a["Vs"][iy]
    Vll = np.concatenate([a["VLLp"][iy], -a["VLLp"][iy][-2::-1]])
    Vsidl = a["Vedge"][iy] + a["Vwc"][iy]
    Vu = P.ULS["g"] * Vdl + P.ULS["sidl"] * Vsidl + P.ULS["q"] * Vll
    for y, lab, col in zip((Vdl, Vsidl, Vll, Vu), labels, (CAT[0], CAT[1], CAT[2], INK)):
        axs[1].plot(x, y, color=col, lw=1.9 if lab == "ULS" else 1.5)
    axs[1].set_title("(b) Shear force"); axs[1].set_xlabel("distance from bearing (m)"); axs[1].set_ylabel("shear (kN)")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "f05_envelopes")


def fig_stage_stresses():
    d = json.load(open("out/design.json"))
    rows = d["stages"]["G1"]["16.500"]["rows"]
    lim = d["limits"]
    order = ["I transfer (stage-1 cables, 10 d)", "II all cables stressed (28 d)", "III deck slab cast (60 d)",
             "IV open to traffic (90 d, rare)", "V quasi, inf", "V frequent, inf", "V rare, inf"]
    short = ["Transfer\n(2 cables)", "All cables\n(28 d)", "Deck\ncast", "Opening\n(rare)", "Quasi-\npermanent", "Frequent", "Rare"]
    top = [rows[k]["top"] for k in order]; bot = [rows[k]["bot"] for k in order]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    xx = np.arange(len(order))
    ax.bar(xx - 0.19, top, 0.36, color=CAT[0], label="top of girder")
    ax.bar(xx + 0.19, bot, 0.36, color=CAT[1], label="soffit")
    for i, (a_, b_) in enumerate(zip(top, bot)):
        ax.text(i - 0.19, a_ + (0.35 if a_ >= 0 else -1.1), f"{a_:.1f}", ha="center", fontsize=8.5)
        ax.text(i + 0.19, b_ + (0.35 if b_ >= 0 else -1.4), f"{b_:.1f}", ha="center", fontsize=8.5)
    ax.axhline(lim["rare_g"], color=RED, lw=0.8, ls="--")
    ax.text(6.55, lim["rare_g"] - 1.3, r"0.48$f_{ck}$ = 21.6 MPa", fontsize=9, ha="right")
    ax.axhline(-lim["fctm"], color=RED, lw=0.8, ls="--")
    ax.text(-0.45, -lim["fctm"] - 1.45, r"$-f_{ctm}$ = $-$3.3 MPa (cracking)", fontsize=9, ha="left")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(xx, short, fontsize=9)
    ax.set_ylabel("stress, compression positive (MPa)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(-6.0, 24.5)
    save(fig, "f06_stage_stresses")


def fig_ll_distribution():
    d = json.load(open("out/design.json"))
    a = np.load("out/design_arrays.npz")
    g = a["MLL"][:, -1]; c = np.array(d["courbon"]["M"])
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    xx = np.arange(6)
    ax.bar(xx - 0.19, g, 0.36, color=CAT[0], label="grillage, worst arrangement for each girder")
    ax.bar(xx + 0.19, c, 0.36, color=CAT[1], label="Courbon, 70R + Class A at the kerb")
    ax.set_xticks(xx, [f"G{i+1}" for i in range(6)])
    ax.set_ylabel("live-load moment (kN m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9)
    save(fig, "f07_ll_distribution")


def fig_modes():
    r = np.load("out/twin_reference.npz")
    f, w = r["f"], r["w"]
    from psc.grillage import Grillage
    g = Grillage()
    names = ["(a) 1st bending", "(b) 1st torsion", "(c) 1st transverse", "(d) 2nd bending"]
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.9))
    X, Y = np.meshgrid(g.x, g.y)
    for ax, k, nm in zip(axs, SC.TRACK, names):
        ax.grid(False); ax.set_axis_off()
        sh = w[:, :, k] / np.abs(w[:, :, k]).max()
        px = X + 0.55 * Y; py = 0.35 * Y + 3.0 * sh
        for i in range(g.ny):
            ax.plot(px[i], py[i], color=CAT[0], lw=1.0)
        for j in range(0, g.nx + 1, 2):
            ax.plot(px[:, j], py[:, j], color=CAT[0], lw=0.5, alpha=0.7)
        px0 = X + 0.55 * Y; py0 = 0.35 * Y
        ax.plot(px0[0], py0[0], color="#bbbbbb", lw=0.6); ax.plot(px0[-1], py0[-1], color="#bbbbbb", lw=0.6)
        ax.set_title(f"{nm}\n{f[k]:.2f} Hz", fontsize=9.5)
    save(fig, "f08_mode_shapes")


def fig_sensor_layout():
    fig, ax = plt.subplots(figsize=(W, 3.5))
    ax.grid(False); ax.set_aspect("equal")
    for y0 in P.GIRDER_Y:
        ax.plot([0, P.L], [y0, y0], color=MUTED, lw=1.0)
    for xc in P.CROSS_X:
        ax.plot([xc, xc], [P.GIRDER_Y[0], P.GIRDER_Y[-1]], color="#dddddd", lw=4, zorder=0)
    ax.add_patch(Rectangle((0, -8), P.L, 16, fc="none", ec=INK2, lw=0.6))
    from psc.grillage import Grillage
    gx = Grillage().x
    for name, iy, ix in DY.SENSORS:
        ax.plot(gx[ix], P.GIRDER_Y[iy], marker="o", ms=7, color=CAT[0], mec="white", mew=1.0, zorder=5)
        ax.text(gx[ix] + 0.4, P.GIRDER_Y[iy] + 0.45, name, fontsize=8.5)
    for iy in (0, 2, 3, 5):
        ax.plot(P.L / 2 - 1.0, P.GIRDER_Y[iy], marker="s", ms=5.5, color=CAT[1], mec="white", zorder=5)
    for iy in (0, 5):
        for xb in (0.0, P.L):
            ax.plot(xb, P.GIRDER_Y[iy], marker="D", ms=6, color=CAT[4], mec="white", zorder=6)
    for y in (-7.2, 0.0, 7.2):
        ax.plot(P.L / 2 + 1.5, y, marker="^", ms=6.5, color=CAT[2], mec="white", zorder=5)
    for iy in (0, 3):
        ax.plot(P.L / 2 + 1.0, P.GIRDER_Y[iy] - 0.6, marker="P", ms=6.5, color=CAT[6], mec="white", zorder=5)
    for i, y0 in enumerate(P.GIRDER_Y):
        ax.text(-0.7, y0, f"G{i+1}", ha="right", va="center", fontsize=9.5)
    ax.text(0, 8.6, "end A (pier P1)", fontsize=9); ax.text(P.L, 8.6, "end B (pier P2)", fontsize=9, ha="right")
    hand = [plt.Line2D([], [], marker=m, ls="", color=c, ms=6.5, label=l) for m, c, l in
            (("o", CAT[0], "accelerometer, vertical (12)"), ("s", CAT[1], "VW strain gauge in soffit (4)"),
             ("D", CAT[4], "bearing displacement transducer (4)"), ("^", CAT[2], "levelling point (3)"),
             ("P", CAT[6], "temperature sensors (2 sets)"))]
    ax.legend(handles=hand, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9)
    ax.set_xlim(-2.0, P.L + 0.8); ax.set_ylim(-8.8, 9.6)
    ax.set_xticks([0, 8.25, 16.5, 24.75, 33]); ax.set_yticks([])
    ax.set_xlabel("distance from bearing A (m)")
    save(fig, "f09_sensor_layout")


# ================================================================ baseline
def fig_commissioning():
    acc = np.load("out/data/commissioning_record.npy")
    fd = np.load("out/data/commissioning_fdd.npz")
    fs = DY.FS
    fig, axs = plt.subplots(3, 1, figsize=(W, 6.6))
    t = np.arange(acc.shape[1]) / fs
    k = (t > 300) & (t < 360)
    axs[0].plot(t[k], acc[1, k] * 1e3 / 9.81, color=CAT[0], lw=0.6)
    axs[0].set_ylabel("acceleration (mg)"); axs[0].set_xlabel("time (s)")
    axs[0].set_title("(a) 60 s of channel A2 (mid-span of G1)")
    y = DY.preprocess(acc.astype(float), fs)
    X, freqs, scale = DY.segment_fft(y, fs)
    psd = (np.abs(X[1]) ** 2).mean(axis=0) * scale
    axs[1].semilogy(freqs, psd, color=CAT[0], lw=0.9)
    band = (freqs > 1.0) & (freqs < 25)
    axs[1].set_xlim(0.5, 25); axs[1].set_ylim(psd[band].min() / 3, psd[band].max() * 30)
    axs[1].set_ylabel("PSD (m$^2$s$^{-4}$/Hz)"); axs[1].set_xlabel("frequency (Hz)")
    axs[1].set_title("(b) Welch spectrum of A2 (Hann window, 8192 points, $\\Delta f$ = 0.012 Hz)")
    s = fd["s"]
    for j, col in enumerate([INK, CAT[1], CAT[2]]):
        axs[2].semilogy(fd["freqs"], s[:, j], color=col, lw=0.9 if j else 1.1, label=f"singular value {j+1}")
    names = {0: "B1", 1: "T1", 2: "TR1", 4: "B2"}
    shift = {0: -1.5, 1: 1.5, 2: 0.0, 4: 0.0}
    for m, fk in zip(fd["modes"], fd["f"]):
        if int(m) not in names:
            continue
        i = np.argmin(np.abs(fd["freqs"] - fk))
        axs[2].annotate(f"{names[int(m)]} {fk:.2f} Hz", (fk, s[i, 0]), (fk + shift[int(m)], s[i, 0] * 40),
                        ha="center", fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    bandf = (fd["freqs"] > 1.0) & (fd["freqs"] < 25)
    axs[2].set_ylim(s[bandf, 2].min() / 3, s[bandf, 0].max() * 600)
    axs[2].set_xlim(0.5, 25); axs[2].set_xlabel("frequency (Hz)"); axs[2].set_ylabel("singular value")
    axs[2].set_title("(c) Frequency domain decomposition of all 12 channels")
    axs[2].legend(loc="lower left", ncol=3, fontsize=9)
    fig.tight_layout(h_pad=0.8)
    save(fig, "f10_commissioning")


def fig_baseline_year():
    rows = FE.read_csv("out/data/baseline_year.csv")
    b = json.load(open("out/baseline.json"))["norm"]
    T = np.array([r["T"] for r in rows]); f1 = np.array([r["f1"] for r in rows]); day = np.array([r["day"] for r in rows])
    hour = np.array([r["hour"] for r in rows]); la = np.log([r["a_rms"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.0))
    for h, col, lab in ((2, CAT[0], "02:00 records"), (14, CAT[1], "14:00 records")):
        k = hour == h
        axs[0].plot(T[k], f1[k], "o", ms=2.6, color=col, alpha=0.6, mec="none", label=lab)
    co = b["f1"]["coef"]
    tt = np.linspace(T.min(), T.max(), 10)
    axs[0].plot(tt, co[0] + co[1] * (tt - 25) + co[2] * np.median(la), color=INK, lw=1.3,
                label=f"fit, {b['f1']['per_degC_pct']:+.3f} %/°C")
    axs[0].set_xlabel("effective bridge temperature (°C)"); axs[0].set_ylabel("$f_1$ (Hz)")
    axs[0].set_title("(a) $f_1$ against temperature"); axs[0].legend(fontsize=9, loc="upper right")
    X = np.column_stack([np.ones_like(T), T - 25, la])
    res = 100 * (f1 / (X @ np.array(co)) - 1)
    axs[1].axvspan(160, 275, color="#eeeeee", zorder=0)
    axs[1].plot(day, res, "o", ms=2.2, color=CAT[0], alpha=0.6, mec="none")
    sd = res.std()
    for s_ in (-3, 3):
        axs[1].axhline(s_ * sd, color=RED, lw=0.8, ls="--")
    axs[1].text(355, 3 * sd + 0.04, "±3 s.d.", fontsize=9, ha="right")
    axs[1].text(217, -3 * sd - 0.17, "monsoon", ha="center", fontsize=9, color=INK2)
    axs[1].set_ylim(-3 * sd - 0.25, 3 * sd + 0.2)
    axs[1].set_xlabel("day of the year"); axs[1].set_ylabel("$f_1$ residual (%)")
    axs[1].set_title(f"(b) Residual after normalisation (s.d. {sd:.2f} %)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "f11_baseline_year")


def fig_automac():
    r = np.load("out/twin_reference.npz")
    w = r["w"]

    def am(sens):
        Sm = np.array([w[iy, ix] for _, iy, ix in sens])
        n = Sm.shape[1]
        return np.array([[DY.mac(Sm[:, i], Sm[:, j]) for j in range(n)] for i in range(n)])
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.2))
    lab = ["B1", "T1", "TR1", "T2", "B2", "TR2", "TR3", "T3"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
    for ax, sens, title in ((axs[0], DY.SENSORS[:10], "(a) First layout, 10 accelerometers"),
                            (axs[1], DY.SENSORS, "(b) Adopted layout, 12 accelerometers")):
        M = am(sens)
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(8), lab, fontsize=9); ax.set_yticks(range(8), lab, fontsize=9)
        for i in range(8):
            for j in range(8):
                if i != j and M[i, j] > 0.25:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if M[i, j] > 0.6 else INK)
        ax.set_title(title, fontsize=10)
    fig.tight_layout(w_pad=1.5)
    save(fig, "f12_automac")


def fig_flowchart():
    """Overall flow of the study (Chapter 1)."""
    steps = [
        ("Bridge configuration", "33 m span, six girders, 16 m deck", "2"),
        ("Structural analysis and design", "grillage model, IRC:6 loads, IRC:112 checks", "3"),
        ("Prestressing system", "four 12T15 cables per girder, losses, stage stresses", "4"),
        ("Healthy baseline", "commissioning tests, model updating, one year of data", "5"),
        ("Inspection, NDT and monitoring", "inspection scheme, NDT programme, sensors", "6-9"),
        ("Deterioration and synthetic data", "six mechanisms, 3600 simulated campaigns", "10-11"),
        ("Features and AI/ML assessment", "novelty detection, classification, regression", "12-13"),
        ("Assessment of the bridge at year 18", "monitoring trends, inspection, NDT, ML output", "14"),
        ("Maintenance decision", "actions, time frames, reinspection", "15"),
    ]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(5.0, 7.0))
    ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, n)
    for k, (t1, t2, ch) in enumerate(steps):
        y = n - 0.5 - k
        ax.add_patch(Rectangle((0.4, y - 0.36), 9.2, 0.72, fc="#f2f2f2", ec=INK2, lw=0.8))
        ax.text(5.0, y + 0.13, f"{t1}  (Ch. {ch})", ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(5.0, y - 0.17, t2, ha="center", va="center", fontsize=9.5)
        if k < n - 1:
            ax.annotate("", (5.0, y - 0.64), (5.0, y - 0.36), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))
    save(fig, "f00_flowchart")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    import os; os.makedirs("out/figs", exist_ok=True)
    if which in ("design", "all"):
        fig_flowchart(); fig_cross_section(); fig_girder_section(); fig_cable_profile(); fig_prestress_force()
        fig_envelopes(); fig_stage_stresses(); fig_ll_distribution(); fig_modes(); fig_sensor_layout()
    if which in ("baseline", "all"):
        fig_commissioning(); fig_baseline_year(); fig_automac()
    if which in ("ml", "all"):
        import make_figures_ml as M2; M2.run()
    if which in ("case", "all"):
        import make_figures_ml as M2; M2.run_case()
