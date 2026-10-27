"""
Step 1 - configuration, analysis and prestressing of the healthy (as-designed) bridge.
Writes out/design.json, out/design_log.txt and out/design_arrays.npz
"""
import json
import time
import numpy as np
from psc import inputs as P, section as S, prestress as PS, design as D, vehicles as V
from psc.grillage import Grillage

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)

t0 = time.time()
R = {}

# ------------------------------------------------------------------ 1. configuration
log("== Configuration ==")
log(f"span {P.L} m, girder length {P.L_GIRDER} m, deck {P.DECK_WIDTH} m, {P.N_GIRDERS} girders @ {P.GIRDER_SPACING} m")
log("girder lines y =", np.round(P.GIRDER_Y, 3).tolist())
opts = D.option_study()
log("option study (Courbon, preliminary):")
for o in opts:
    log(f"  n={o['n']} s={o['spacing']:.2f} slab={o['slab']:.3f} k_outer={o['k_outer']:.3f} "
        f"M_LL={o['M_LL']:.0f} girder={o['girder_t']:.1f} t  Pe={o['Pe']:.0f} kN strands/girder={o['strands']:.1f} "
        f"total={o['strands_total']:.0f}  concrete={o['concrete']:.0f} m3")
R["options"] = opts

pre, pre_eb = S.precast(), S.precast(True)
b_in, b_out = S.effective_width("inner"), S.effective_width("outer")
comp_in, comp_out = S.composite(b_in), S.composite(b_out)
comp_in_eb, comp_out_eb = S.composite(b_in, True), S.composite(b_out, True)
R["sections"] = dict(precast=pre, end_block=pre_eb, composite_inner=comp_in, composite_outer=comp_out)
log(f"precast A={pre['A']:.4f} yb={pre['yb']:.4f} I={pre['I']:.4f} Zb={pre['Zb']:.4f} Zt={pre['Zt']:.4f}")
log(f"composite inner b={b_in:.2f} A={comp_in['A']:.4f} yb={comp_in['yb']:.4f} I={comp_in['I']:.4f} Zb={comp_in['Zb']:.4f}")
log(f"composite outer b={b_out:.2f} A={comp_out['A']:.4f} yb={comp_out['yb']:.4f} I={comp_out['I']:.4f}")

# ------------------------------------------------------------------ 2. actions
log("\n== Dead load (non-composite, per girder) ==")
DL = {iy: D.dead_load_actions(iy) for iy in range(P.N_GIRDERS)}
mid = P.NX // 2
for iy in (0, 1, 2):
    d = DL[iy]
    log(f"G{iy+1}: Mg={d['Mg'][mid]:.0f} Ms(slab+CG)={d['Ms'][mid]:.0f} kNm  Vg={d['Vg'][0]:.0f} Vs={d['Vs'][0]:.0f} kN")

log("\n== Composite stage (grillage) ==")
g = Grillage()
CA = D.composite_actions(g, verbose=True)
for iy in (0, 1, 2):
    log(f"G{iy+1}: M_edge={CA['edge']['M'][iy, mid]:.0f} M_wc={CA['wc']['M'][iy, mid]:.0f} "
        f"M_fpL={CA['fpL']['M'][iy, mid]:.0f} M_fpR={CA['fpR']['M'][iy, mid]:.0f} M_LL={CA['LL']['M'][iy, -1]:.0f}")
for k, v in CA["LL"]["layouts"].items():
    log(" ", k, v)

def footpath_adverse(key, iy, ix):
    a, b = CA["fpL"][key][iy, ix], CA["fpR"][key][iy, ix]
    return max(a, 0) + max(b, 0)

# Courbon check of the mid-span LL moment, 70R + Class A at the kerb
M70 = D.il_moment_max(V.R70_W) * (1 + P.IMPACT)
MA = D.il_moment_max(V.CLASS_A) * (1 + P.IMPACT)
y70 = -P.KERB_Y + 1.2 + V.R70_W.half_width
yA = y70 + V.R70_W.half_width + V.GAP + V.CLASS_A.half_width
e = (V.R70_W.total * y70 + V.CLASS_A.total * yA) / (V.R70_W.total + V.CLASS_A.total)
yg = P.GIRDER_Y
k_c = 1 / P.N_GIRDERS * (1 + P.N_GIRDERS * e * yg / np.sum(yg ** 2))
courbon = (M70 + MA) * (1 - P.LANE_REDUCTION) * k_c
log("Courbon 70RW+A mid-span LL moments:", np.round(courbon, 0).tolist())
log("Grillage  (worst of all cases)  :", np.round(CA["LL"]["M"][:, -1], 0).tolist())
R["courbon"] = dict(k=k_c.tolist(), M=courbon.tolist(), M_total_70R=M70, M_total_A=MA, e=e)

# ------------------------------------------------------------------ 3. prestress
log("\n== Prestress ==")
xs_b = D.XG[: mid + 1]                     # stations from bearing (half span)
st = D.prestress_state(xs_b)
imm = st["imm"]
for c in st["cables"]:
    n = c.name
    log(f"{n}: Pj={imm['Pj'][n]:.0f}  at mid: friction+slip {imm['Pfs'][n][-1]:.0f}, ES {imm['ES'][n][-1]:.0f}, "
        f"P0 {imm['P0'][n][-1]:.0f} kN; set length {imm['ls'][n]:.2f} m; angle to mid {float(c.angle(PS.HALF)):.3f} rad")

# quasi-permanent concrete stress at tendon centroid, mid-span, for creep
def qp_stress_at_cp(iy, ix):
    sec = PS.section_at(xs_b[ix] + P.END_OVERHANG)
    P0, zcp = D.force_and_ecc(st, {k: v[ix] for k, v in imm["P0"].items()}, ix=ix)
    ecc = sec["zc"] - zcp
    s_p = (P0 / sec["A"] + P0 * ecc ** 2 / sec["I"]) / 1e3
    Mpre = DL[iy]["Mg"][ix] + DL[iy]["Ms"][ix]
    s_g = -Mpre * ecc / sec["I"] / 1e3
    comp = comp_out if iy in D.OUTER else comp_in
    Msidl = CA["edge"]["M"][iy, ix] + CA["wc"]["M"][iy, ix]
    s_c = -Msidl * (comp["zc"] - zcp) / comp["I"] / 1e3
    return s_p + s_g + s_c

LT = {}
Pe_cable = {c.name: np.zeros(len(xs_b)) for c in st["cables"]}
for ix in range(len(xs_b)):
    sqp = qp_stress_at_cp(0, ix)
    imm_ix = dict(imm, P0={k: v[ix:ix + 1] for k, v in imm["P0"].items()})
    lt = PS.long_term_losses(np.array([xs_b[ix] + P.END_OVERHANG]), sqp, imm=imm_ix)
    for c in st["cables"]:
        Pe_cable[c.name][ix] = imm["P0"][c.name][ix] - float(np.squeeze(lt[c.name]["dP"]))
    if ix == mid:
        LT = lt
        log(f"QP stress at tendon level (mid, G1) = {sqp:.2f} MPa")
for n, v in LT.items():
    log(f"{n}: phi={float(v['phi']):.2f} ecs={float(v['ecs'])*1e6:.0f}e-6 relax={float(v['relax']):.1f} MPa "
        f"dsig={float(v['dsig']):.1f} MPa dP={float(v['dP']):.0f} kN  h0={float(v['h0']):.0f} mm")

Pj_tot = sum(imm["Pj"].values())
P0_mid = sum(imm["P0"][n][-1] for n in imm["P0"])
Pfs_mid = sum(imm["Pfs"][n][-1] for n in imm["Pfs"])
Pe_mid = sum(Pe_cable[n][-1] for n in Pe_cable)
log(f"Totals at mid-span: Pj={Pj_tot:.0f}  after friction+slip={Pfs_mid:.0f}  P0={P0_mid:.0f}  Pe={Pe_mid:.0f} kN")
log(f"  immediate loss {100*(1-P0_mid/Pj_tot):.1f} %, long-term {100*(P0_mid-Pe_mid)/Pj_tot:.1f} %, total {100*(1-Pe_mid/Pj_tot):.1f} %")
R["prestress"] = dict(Pj=Pj_tot, Pfs_mid=Pfs_mid, P0_mid=P0_mid, Pe_mid=Pe_mid,
                      per_cable={c.name: dict(Pj=imm["Pj"][c.name], Pfs_mid=float(imm["Pfs"][c.name][-1]),
                                              ES_mid=float(imm["ES"][c.name][-1]), P0_mid=float(imm["P0"][c.name][-1]),
                                              Pe_mid=float(Pe_cable[c.name][-1]), set_length=imm["ls"][c.name],
                                              phi=float(LT[c.name]["phi"]), dsig=float(LT[c.name]["dsig"]),
                                              relax=float(LT[c.name]["relax"]), ecs=float(LT[c.name]["ecs"]))
                                 for c in st["cables"]})

# ------------------------------------------------------------------ 4. stage stresses
log("\n== Stage stresses (compression +, MPa) ==")
sec2 = D.secondary_stresses(b_out)
for k, v in sec2.items():
    log(f"{k}: bottom {v['bot']:+.2f} top girder {v['top_girder']:+.2f} top slab {v['top_slab']:+.2f}")

f_lt = lambda t: PS.loss_fraction(t)
lim = dict(transfer1=0.48 * P.FCK_TRANSFER, transfer2=0.48 * P.FCK_GIRDER,
           rare_g=0.48 * P.FCK_GIRDER, rare_s=0.48 * P.FCK_DECK,
           qp_g=0.36 * P.FCK_GIRDER, qp_s=0.36 * P.FCK_DECK, fctm=P.fctm(P.FCK_GIRDER))

def comp_props(iy, xb):
    xg = xb + P.END_OVERHANG
    f = float((S.web_width_at(xg) - P.WEB_T) / (P.BOT_FLANGE_W - P.WEB_T))
    a, b = (comp_out, comp_out_eb) if iy in D.OUTER else (comp_in, comp_in_eb)
    return {k: a[k] + f * (b[k] - a[k]) for k in ("A", "I", "zc", "Zb", "Zt_girder", "Zt_slab")}

def stage_table(iy, ix):
    xb = xs_b[ix]
    sec = PS.section_at(xb + P.END_OVERHANG)
    sec["Zt"] = sec["I"] / (P.GIRDER_DEPTH - sec["zc"]); sec["Zb"] = sec["I"] / sec["zc"]
    cp = comp_props(iy, xb)
    Mg, Ms = DL[iy]["Mg"][ix], DL[iy]["Ms"][ix]
    Me, Mw = CA["edge"]["M"][iy, ix], CA["wc"]["M"][iy, ix]
    Mll = CA["LL"]["M"][iy, ix] + footpath_adverse("M", iy, ix)
    rows = {}
    s1 = [c.name for c in st["cables"] if c.stage == 1]
    F1 = {n: imm["Pfs"][n][ix] - imm["ES1"][n][ix] for n in s1}
    P1, z1 = D.force_and_ecc(st, F1, s1, ix=ix)
    rows["I transfer (stage-1 cables, 10 d)"] = D.fibre(P1, z1, Mg, sec)
    lt_frac28 = f_lt(28)
    F2 = {n: imm["P0"][n][ix] - (lt_frac28 * (imm["P0"][n][ix] - Pe_cable[n][ix]) if n in s1 else 0) for n in imm["P0"]}
    P2, z2 = D.force_and_ecc(st, F2, ix=ix)
    rows["II all cables stressed (28 d)"] = D.fibre(P2, z2, Mg, sec)
    F3 = {n: imm["P0"][n][ix] - f_lt(P.T_DECK) * (imm["P0"][n][ix] - Pe_cable[n][ix]) for n in imm["P0"]}
    P3, z3 = D.force_and_ecc(st, F3, ix=ix)
    rows["III deck slab cast (60 d)"] = D.fibre(P3, z3, Mg + Ms, sec)
    Pe_, ze = D.force_and_ecc(st, {n: Pe_cable[n][ix] for n in Pe_cable}, ix=ix)
    out = dict(rows={}, P=dict(P1=P1, P2=P2, P3=P3, Pe=Pe_), z=dict(z1=z1, z2=z2, z3=z3, ze=ze))
    for k, (t, b) in rows.items():
        out["rows"][k] = dict(top=t, bot=b, slab=None)
    # service, t = infinity
    for combo, fac in P.SLS.items():
        for rname, r in (("inf", 0.9), ("sup", 1.1)):
            t_g, b_g = D.fibre(r * Pe_, ze, Mg + Ms, sec)
            Mc = Me + Mw + fac["q"] * Mll
            top_s = Mc / cp["Zt_slab"] / 1e3
            top_g = t_g + Mc / cp["Zt_girder"] / 1e3
            bot = b_g - Mc / cp["Zb"] / 1e3
            ds = sec2["diff_shr"]
            # temperature difference and differential shrinkage only where adverse
            tp, tr = sec2["temp_pos"], sec2["temp_rev"]
            tb = min(tp["bot"], tr["bot"], 0.0) * fac["temp"]
            tt = max(tp["top_slab"], tr["top_slab"], 0.0) * fac["temp"]
            tg = max(tp["top_girder"], tr["top_girder"], 0.0) * fac["temp"]
            out["rows"][f"V {combo}, {rname}"] = dict(
                top=top_g + tg + max(ds["top_girder"], 0.0),
                bot=bot + tb + min(ds["bot"], 0.0),
                slab=top_s + tt + max(ds["top_slab"], 0.0))
    # initial service (90 d, SIDL + full LL, little loss yet)
    F4 = {n: imm["P0"][n][ix] - f_lt(P.T_SIDL) * (imm["P0"][n][ix] - Pe_cable[n][ix]) for n in imm["P0"]}
    P4, z4 = D.force_and_ecc(st, F4, ix=ix)
    t_g, b_g = D.fibre(P4, z4, Mg + Ms, sec)
    out["rows"]["IV open to traffic (90 d, rare)"] = dict(
        top=t_g + (Me + Mw + Mll) / cp["Zt_girder"] / 1e3, bot=b_g - (Me + Mw + Mll) / cp["Zb"] / 1e3,
        slab=(Me + Mw + Mll) / cp["Zt_slab"] / 1e3)
    out["M"] = dict(Mg=Mg, Ms=Ms, Medge=Me, Mwc=Mw, Mll=Mll)
    return out

STAGES = {}
for iy in (0, 1, 2):
    STAGES[iy] = {}
    for ix in (mid, mid // 2, 2, 1):
        tab = stage_table(iy, ix)
        STAGES[iy][ix] = tab
for iy in (0, 1, 2):
    log(f"-- G{iy+1} mid-span: M {', '.join(f'{k}={v:.0f}' for k, v in STAGES[iy][mid]['M'].items())}")
    for k, r in STAGES[iy][mid]["rows"].items():
        log(f"   {k:38s} top {r['top']:+6.2f}  bot {r['bot']:+6.2f}" + (f"  slab {r['slab']:+6.2f}" if r["slab"] is not None else ""))
log("-- G1 at L/4 and near support:")
for ix in (mid // 2, 2, 1):
    for k, r in STAGES[0][ix]["rows"].items():
        if k.startswith(("I ", "II", "V rare, sup", "V frequent, inf", "V rare, inf")):
            log(f"   x={xs_b[ix]:5.2f} {k:34s} top {r['top']:+6.2f} bot {r['bot']:+6.2f}")
log("limits:", {k: round(v, 2) for k, v in lim.items()})

# governing values over the three girders / checked sections
def worst(key, fibre, fn):
    vals = [STAGES[iy][ix]["rows"][key][fibre] for iy in STAGES for ix in STAGES[iy]]
    return fn(vals)
summary = dict(
    transfer1_max_comp=worst("I transfer (stage-1 cables, 10 d)", "bot", max),
    transfer1_min=min(worst("I transfer (stage-1 cables, 10 d)", "top", min), worst("I transfer (stage-1 cables, 10 d)", "bot", min)),
    transfer2_max_comp=worst("II all cables stressed (28 d)", "bot", max),
    transfer2_min_top=worst("II all cables stressed (28 d)", "top", min),
    deck_min_top=worst("III deck slab cast (60 d)", "top", min),
    rare_min_bot=worst("V rare, inf", "bot", min),
    frequent_min_bot=worst("V frequent, inf", "bot", min),
    qp_min_bot=worst("V quasi, inf", "bot", min),
    rare_max_top_girder=worst("V rare, sup", "top", max),
    rare_max_slab=worst("V rare, sup", "slab", max),
    qp_max_bot=worst("V quasi, sup", "bot", max),
    initial_service_max_top=worst("IV open to traffic (90 d, rare)", "top", max),
)
log("governing:", {k: round(v, 2) for k, v in summary.items()})
R["stages_summary"] = summary
R["limits"] = lim
R["stages"] = {f"G{iy+1}": {f"{xs_b[ix]:.3f}": dict(rows=v["rows"], M=v["M"], P=v["P"]) for ix, v in STAGES[iy].items()} for iy in STAGES}

# ------------------------------------------------------------------ 5. ULS flexure
log("\n== ULS flexure (mid-span) ==")
ULS = {}
for iy in (0, 1, 2):
    m = STAGES[iy][mid]["M"]
    MEd = P.ULS["g"] * (m["Mg"] + m["Ms"]) + P.ULS["sidl"] * m["Medge"] + P.ULS["wc"] * m["Mwc"] + P.ULS["q"] * m["Mll"]
    Pe_, ze = STAGES[iy][mid]["P"]["Pe"], STAGES[iy][mid]["z"]["ze"]
    dp = P.GIRDER_DEPTH + P.SLAB_T - ze
    cap = D.uls_moment_capacity(Pe_, dp, st["Ap"], b_out if iy in D.OUTER else b_in)
    ULS[iy] = dict(MEd=MEd, **cap)
    log(f"G{iy+1}: MEd={MEd:.0f} MRd={cap['M']:.0f} kNm  ratio={MEd/cap['M']:.2f}  x={cap['x']*1e3:.0f} mm "
        f"(x/d={cap['x_over_d']:.3f}) fp={cap['fp']:.0f} MPa eps_p={cap['eps_p']*1e3:.2f}e-3")
R["uls"] = {f"G{k+1}": v for k, v in ULS.items()}

# ------------------------------------------------------------------ 6. shear
log("\n== Shear ==")
SH = {}
for iy in (0, 1, 2):
    for ix in (1, 2):
        xb = xs_b[ix]; xg = xb + P.END_OVERHANG
        Vd = DL[iy]["Vg"][ix] + DL[iy]["Vs"][ix]
        Vsidl = CA["edge"]["V"][iy, ix]; Vwc = CA["wc"]["V"][iy, ix]
        Vll = CA["LL"]["Vp"][iy, ix] + footpath_adverse("V", iy, ix)
        VEd = P.ULS["g"] * Vd + P.ULS["sidl"] * Vsidl + P.ULS["wc"] * Vwc + P.ULS["q"] * Vll
        Vp = sum(Pe_cable[c.name][ix] * np.sin(abs(np.arctan(float(c.slope(xg))))) for c in st["cables"])
        bw = float(S.web_width_at(xg))
        bw_nom = bw - 1.2 * P.DUCT_OD if bw < 0.5 else bw - 0.5 * P.DUCT_OD
        Pe_ = sum(Pe_cable[c.name][ix] for c in st["cables"])
        cp = comp_props(iy, xb); cp["b_eff"] = b_out if iy in D.OUTER else b_in; cp["m"] = comp_in["m"]
        scp = Pe_ / cp["A"] / 1e3
        d_eff = P.GIRDER_DEPTH + P.SLAB_T - 0.5
        chk = D.shear_check(VEd, None, cp, scp, bw_nom, d_eff, Vp=0.9 * Vp)
        SH[(iy, ix)] = dict(x=xb, bw=bw, bw_nom=bw_nom, **chk)
        log(f"G{iy+1} x={xb:.3f}: VEd={VEd:.0f} Vp(0.9)={0.9*Vp:.0f} VRdc={chk['VRdc']:.0f} VRdmax={chk['VRdmax']:.0f} "
            f"Asw/s={chk['Asw_s']:.0f} mm2/m (min {chk['Asw_s_min']:.0f}) bw_nom={bw_nom:.3f}")
R["shear"] = {f"G{k[0]+1}@{v['x']:.2f}": v for k, v in SH.items()}

# ------------------------------------------------------------------ 7. deflection, camber
log("\n== Deflection and camber ==")
dLL = {}
for iy in (0, 1, 2):
    w_v = CA["LL"]["layouts"][f"w_G{iy+1}"]
    w_fp = max(-CA["fpL"]["w"][iy, mid], 0) + max(-CA["fpR"]["w"][iy, mid], 0)
    dLL[iy] = (w_v + w_fp) * 1e3
log("LL deflection mid (mm):", {f"G{k+1}": round(v, 1) for k, v in dLL.items()}, f"limit L/800 = {P.L/800*1e3:.1f}")

def camber(Pforces, zfun_E, Mfun, E):
    xg = np.linspace(0.0, P.L_GIRDER, 401)
    xb = xg - P.END_OVERHANG
    sec = PS.section_at(xg)
    Ptot = np.zeros_like(xg); zP = np.zeros_like(xg)
    for c in st["cables"]:
        Ptot += Pforces[c.name]
        zP += Pforces[c.name] * c.z(xg)
    zP /= Ptot
    M = Mfun(xb) - Ptot * (sec["zc"] - zP)
    kappa = M / (E * 1e3 * sec["I"])
    mb = np.where(xb <= P.L / 2, xb / 2, (P.L - xb) / 2)
    inside = (xb >= 0) & (xb <= P.L)
    return float(np.trapezoid((kappa * mb)[inside], xb[inside]))       # +ve = downward

d0 = DL[0]
Mg_f = lambda xb: np.interp(xb, D.XG, d0["Mg"])
Ms_f = lambda xb: np.interp(xb, D.XG, d0["Ms"])
# forces as constant along length at their mid-span values scaled by the friction/slip shape
xg_full = np.linspace(0.0, P.L_GIRDER, 401)
imm_full = PS.immediate_losses(xg_full)
ratio_e = Pe_mid / P0_mid
E28 = P.Ecm_t(P.FCK_GIRDER, 28)
c_transfer = camber(imm_full["P0"], None, Mg_f, E28)
Pe_full = {k: v * ratio_e for k, v in imm_full["P0"].items()}
c_deck = camber({k: v * (1 - f_lt(P.T_DECK) * (1 - ratio_e)) for k, v in imm_full["P0"].items()},
                None, lambda xb: Mg_f(xb) + Ms_f(xb), P.Ecm(P.FCK_GIRDER))
phi_mid = float(np.mean([LT[n]["phi"] for n in LT]))
c_P_only = camber(Pe_full, None, lambda xb: 0 * xb, P.Ecm(P.FCK_GIRDER))
log(f"camber at transfer (all cables, self weight) = {-c_transfer*1e3:.1f} mm up")
log(f"after deck casting = {-c_deck*1e3:.1f} mm up;  Pe-only camber (elastic) = {-c_P_only*1e3:.1f} mm; phi = {phi_mid:.2f}")
R["deflection"] = dict(LL_mm={f"G{k+1}": v for k, v in dLL.items()}, limit_mm=P.L / 800 * 1e3,
                       camber_transfer_mm=-c_transfer * 1e3, camber_deck_mm=-c_deck * 1e3,
                       camber_Pe_elastic_mm=-c_P_only * 1e3, phi=phi_mid)

# ------------------------------------------------------------------ 8. bearings
log("\n== Bearings ==")
BR = {}
for iy in (0, 1, 2):
    Rdl = DL[iy]["Rg"] + DL[iy]["Rs"]
    Rsidl = CA["edge"]["R"][iy] + CA["wc"]["R"][iy]
    Rll = CA["LL"]["layouts"][f"R_G{iy+1}"] + max(CA["fpL"]["R"][iy], 0) + max(CA["fpR"]["R"][iy], 0)
    Nmax, Nmin = Rdl + Rsidl + Rll, Rdl + Rsidl
    ecs = float(LT["C3"]["ecs"])
    creep_short = phi_mid * (Pe_mid / comp_out["A"] / 1e3) / P.Ecm(P.FCK_GIRDER)
    v_x = (ecs + creep_short) * P.L / 2 + P.ALPHA_T * 25 * P.L / 2
    chk = D.bearing_check(Nmax, Nmin, v_x, 0.004)
    BR[iy] = dict(Nmax=Nmax, Nmin=Nmin, v_x=v_x, **chk)
    log(f"G{iy+1}: Nmax={Nmax:.0f} Nmin={Nmin:.0f} kN v_x={v_x*1e3:.1f} mm  S={chk['S']:.1f} eps_c={chk['eps_c']:.2f} "
        f"eps_q={chk['eps_q']:.2f} eps_a={chk['eps_a']:.2f} total={chk['eps_t']:.2f} sigma_min={chk['sigma_min']:.1f} MPa")
R["bearings"] = {f"G{k+1}": v for k, v in BR.items()}
hl = g.bearing_slopes()
log(f"bearing shear movement per degC (healthy, both ends elastomeric): {hl[0,0]:.3f} mm/degC")

# ------------------------------------------------------------------ 9. modal (design model)
f, phi, ws = g.modes(10)
from psc.grillage import mode_label
log("\n== Modal (design model, Edyn = 1.2 Ecm) ==")
for i in range(8):
    log(f"  mode {i+1}: {f[i]:.3f} Hz  {mode_label(ws[:, :, i])}")
R["modes_design"] = [dict(f=float(f[i]), label=mode_label(ws[:, :, i])) for i in range(8)]

# ------------------------------------------------------------------ save
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o

json.dump(clean(R), open("out/design.json", "w"), indent=1)
np.savez("out/design_arrays.npz",
         xs_b=xs_b, XG=D.XG,
         Mg=np.array([DL[i]["Mg"] for i in range(P.N_GIRDERS)]),
         Ms=np.array([DL[i]["Ms"] for i in range(P.N_GIRDERS)]),
         Vg=np.array([DL[i]["Vg"] for i in range(P.N_GIRDERS)]),
         Vs=np.array([DL[i]["Vs"] for i in range(P.N_GIRDERS)]),
         Medge=CA["edge"]["M"], Mwc=CA["wc"]["M"], MfpL=CA["fpL"]["M"], MfpR=CA["fpR"]["M"],
         Vedge=CA["edge"]["V"], Vwc=CA["wc"]["V"], VfpL=CA["fpL"]["V"], VfpR=CA["fpR"]["V"],
         MLL=CA["LL"]["M"], VLLp=CA["LL"]["Vp"], VLLn=CA["LL"]["Vn"],
         xg_full=xg_full, **{f"Pj_{k}": np.full_like(xg_full, v) for k, v in imm_full["Pj"].items()},
         **{f"Pf_{k}": PS.friction_profile(c, xg_full, imm_full["Pj"][k]) for c in imm_full["cables"] for k in [c.name]},
         **{f"Pfs_{k}": v for k, v in imm_full["Pfs"].items()},
         **{f"P0_{k}": v for k, v in imm_full["P0"].items()},
         **{f"Pe_{k}": v for k, v in Pe_full.items()},
         modes_f=f, modes_w=ws,
         temp_z=sec2["temp_pos"]["z"], temp_pos=sec2["temp_pos"]["sig"], temp_rev=sec2["temp_rev"]["sig"],
         diff_shr=sec2["diff_shr"]["sig"])
open("out/design_log.txt", "w").write("\n".join(log_lines))
print(f"\n[done in {time.time()-t0:.0f} s]")
