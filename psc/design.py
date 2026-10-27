"""
Load effects, stage stresses, ULS and SLS checks for the adopted girder.
Sign convention for concrete stress: compression positive (MPa).
x is measured from the bearing unless the name ends in _g (from the girder end).
"""
import numpy as np
from . import inputs as P
from . import section as S
from . import prestress as PS
from . import vehicles as V
from .grillage import Grillage

XG = np.linspace(0, P.L, P.NX + 1)           # grillage stations (from bearing)
OUTER, INNER = (0, P.N_GIRDERS - 1), tuple(range(1, P.N_GIRDERS - 1))


# ---------------------------------------------------------------- dead loads (non-composite)
def tributary(iy):
    if iy in OUTER:
        return P.GIRDER_SPACING / 2 + (P.DECK_WIDTH / 2 - P.GIRDER_Y[-1])
    return P.GIRDER_SPACING


def cross_girder_point(iy):
    full = (P.GIRDER_SPACING - P.WEB_T) * P.CROSS_B * P.CROSS_D * P.GAMMA_RC
    return full / 2 if iy in OUTER else full


def ss_actions(x, w_of_xi, point_loads=()):
    """M (sagging +) and V at x for a simply supported girder with overhangs."""
    xi = np.linspace(-P.END_OVERHANG, P.L + P.END_OVERHANG, 1401)
    w = w_of_xi(xi)
    x = np.atleast_1d(x)
    M = np.zeros_like(x, float); Vv = np.zeros_like(x, float)
    for k, xk in enumerate(x):
        g = np.where(xi <= xk, xi * (P.L - xk) / P.L, xk * (P.L - xi) / P.L)
        gv = np.where(xi < xk, -xi / P.L, (P.L - xi) / P.L)
        M[k] = np.trapezoid(w * g, xi)
        Vv[k] = np.trapezoid(w * gv, xi)
        for xp, Pp in point_loads:
            M[k] += Pp * (xp * (P.L - xk) / P.L if xp <= xk else xk * (P.L - xp) / P.L)
            Vv[k] += Pp * (-xp / P.L if xp < xk else (P.L - xp) / P.L)
    R = np.trapezoid(w * (P.L - xi) / P.L, xi) + sum(Pp * (P.L - xp) / P.L for xp, Pp in point_loads)
    return M, Vv, R


def dead_load_actions(iy, x=XG):
    def w_girder(xi):
        sec = PS.section_at(xi + P.END_OVERHANG)
        return sec["A"] * P.GAMMA_RC
    def w_slab(xi):
        return np.full_like(xi, tributary(iy) * P.SLAB_T * P.GAMMA_RC)
    Mg, Vg, Rg = ss_actions(x, w_girder)
    Ms, Vs, Rs = ss_actions(x, w_slab, [(P.L / 2, cross_girder_point(iy))])
    end_diaph = (P.GIRDER_SPACING - P.BOT_FLANGE_W) * P.END_DIAPH_B * P.END_DIAPH_D * P.GAMMA_RC
    Rs += end_diaph / (2 if iy in OUTER else 1)
    return dict(Mg=Mg, Vg=Vg, Rg=Rg, Ms=Ms, Vs=Vs, Rs=Rs)


# ---------------------------------------------------------------- composite stage (grillage)
def sidl_cases(g):
    """Nodal load vectors: barriers+footpath+railing, wearing coat, footpath LL L / R."""
    yb = P.KERB_Y + P.CRASH_BARRIER_W / 2
    yf = P.KERB_Y + P.CRASH_BARRIER_W + P.FOOTPATH_W / 2
    yr = P.DECK_WIDTH / 2 - P.RAILING_W / 2
    q_fp = P.footpath_ll() * P.FOOTPATH_W
    edge, wc, fpL, fpR = [], [], [], []
    ys = np.arange(-P.KERB_Y + 0.125, P.KERB_Y, 0.25)
    for ix, x in enumerate(g.x):
        trib = g.dx if 0 < ix < g.nx else g.dx / 2
        for s in (-1, 1):
            edge += [(x, s * yb, P.BARRIER_LOAD * trib), (x, s * yf, P.FOOTPATH_LOAD * trib),
                     (x, s * yr, P.RAILING_LOAD * trib)]
        wc += [(x, y, P.WC_LOAD * 0.25 * trib) for y in ys]
        fpL.append((x, -yf, q_fp * trib))
        fpR.append((x, yf, q_fp * trib))
    return {k: g.load_vector(v) for k, v in
            dict(edge=edge, wc=wc, fpL=fpL, fpR=fpR).items()}


def composite_actions(g=None, verbose=False):
    g = g or Grillage()
    g.factor()
    cases = sidl_cases(g)
    res = {}
    for k, F in cases.items():
        U = g.solve(F)
        M, Vv = g.girder_actions(U)
        RA, RB = g.reactions(U)
        res[k] = dict(M=M, V=Vv, w=g.deflection(U), R=RA)
    infl = g.influence()
    yg = g.y
    nsec = P.NX // 2 + 1
    LLM = np.zeros((P.N_GIRDERS, nsec)); LLVp = np.zeros_like(LLM); LLVn = np.zeros_like(LLM)
    layouts = {}
    for iy in range(3):                       # G4-G6 by symmetry
        for ix in range(nsec):
            r = V.worst_combination(infl["M"][iy, ix], g.x, yg, +1)
            k = max(r, key=lambda c: r[c][0])
            LLM[iy, ix] = r[k][0]
            if ix == nsec - 1:
                layouts[f"M_G{iy+1}"] = (k, r[k][1], {c: round(v[0], 1) for c, v in r.items()})
            rp = V.worst_combination(infl["V"][iy, ix], g.x, yg, +1)
            rn = V.worst_combination(infl["V"][iy, ix], g.x, yg, -1)
            LLVp[iy, ix] = max(v[0] for v in rp.values())
            LLVn[iy, ix] = min(v[0] for v in rn.values())
        r = V.worst_combination(infl["RA"][iy], g.x, yg, +1)
        layouts[f"R_G{iy+1}"] = max(v[0] for v in r.values())
        r = V.worst_combination(-infl["w"][iy, nsec - 1], g.x, yg, +1)
        layouts[f"w_G{iy+1}"] = max(v[0] for v in r.values())
        if verbose:
            print(f"  G{iy+1}: LL M mid {LLM[iy, -1]:.0f} kNm, V end {LLVp[iy, 0]:.0f} kN")
    for iy in range(3):
        j = P.N_GIRDERS - 1 - iy
        LLM[j], LLVp[j], LLVn[j] = LLM[iy], LLVp[iy], LLVn[iy]
    res["LL"] = dict(M=LLM, Vp=LLVp, Vn=LLVn, layouts=layouts)
    res["influence"] = infl
    res["grillage"] = g
    return res


# ---------------------------------------------------------------- temperature and shrinkage eigenstresses
def _composite_strips(b_eff, n=400):
    """Discretised depth of the composite section: z from soffit, width, E (MPa)."""
    D, t = P.GIRDER_DEPTH, P.SLAB_T
    z = np.linspace(0, D + t, n)
    dz = z[1] - z[0]
    bw = np.empty_like(z)
    pts = S.girder_outline()
    for i, zz in enumerate(z):
        if zz > D:
            bw[i] = b_eff
        elif zz < P.BOT_FLANGE_T:
            bw[i] = P.BOT_FLANGE_W
        elif zz < P.BOT_FLANGE_T + P.BOT_TAPER_T:
            f = (zz - P.BOT_FLANGE_T) / P.BOT_TAPER_T
            bw[i] = P.BOT_FLANGE_W + f * (P.WEB_T - P.BOT_FLANGE_W)
        elif zz < D - P.TOP_FLANGE_T - P.TOP_TAPER_T:
            bw[i] = P.WEB_T
        elif zz < D - P.TOP_FLANGE_T:
            f = (zz - (D - P.TOP_FLANGE_T - P.TOP_TAPER_T)) / P.TOP_TAPER_T
            bw[i] = P.WEB_T + f * (P.TOP_FLANGE_W - P.WEB_T)
        else:
            bw[i] = P.TOP_FLANGE_W
    E = np.where(z > D, P.Ecm(P.FCK_DECK), P.Ecm(P.FCK_GIRDER))
    return z, bw, E, dz


def eigenstress(free_strain_fn, b_eff, E_scale=1.0):
    z, b, E, dz = _composite_strips(b_eff)
    E = E * E_scale
    e_free = free_strain_fn(z)
    # restrain to a plane: eps = a + k z
    A0 = np.sum(E * b) * dz; A1 = np.sum(E * b * z) * dz; A2 = np.sum(E * b * z * z) * dz
    N0 = np.sum(E * b * e_free) * dz; N1 = np.sum(E * b * e_free * z) * dz
    k = (N1 * A0 - N0 * A1) / (A2 * A0 - A1 * A1)
    a = (N0 - k * A1) / A0
    sig_t = E * (a + k * z - e_free)          # tension +
    D = P.GIRDER_DEPTH
    pick = lambda zz: float(np.interp(zz, z, -sig_t))   # compression +
    return dict(bot=pick(0.0), top_girder=float(np.interp(D - 1e-6, z, -sig_t)),
                top_slab=pick(D + P.SLAB_T), z=z, sig=-sig_t)


def temp_profiles():
    h = P.GIRDER_DEPTH + P.SLAB_T
    tp, tr = P.TEMP_POS, P.TEMP_REV
    h1 = min(0.3 * h, 0.15); h2 = min(max(0.3 * h, 0.10), 0.25); h3 = min(0.3 * h, 0.10 + P.WEARING_T)
    def pos(z):
        d = h - z
        T = np.where(d <= h1, tp["T1"] + (tp["T2"] - tp["T1"]) * d / h1,
                     np.where(d <= h1 + h2, tp["T2"] * (1 - (d - h1) / h2), 0.0))
        T = T + np.where(z <= h3, tp["T3"] * (1 - z / h3), 0.0)
        return P.ALPHA_T * T
    r1 = min(0.2 * h, 0.25); r2 = min(0.25 * h, 0.20)
    def rev(z):
        d = h - z
        T = np.where(d <= r1, -tr["T1"] + (tr["T1"] - tr["T2"]) * d / r1,
                     np.where(d <= r1 + r2, -tr["T2"] * (1 - (d - r1) / r2), 0.0))
        T = T + np.where(z <= r1, -tr["T4"] + (tr["T4"] - tr["T3"]) * z / r1,
                         np.where(z <= r1 + r2, -tr["T3"] * (1 - (z - r1) / r2), 0.0))
        return P.ALPHA_T * T
    return pos, rev


def secondary_stresses(b_eff):
    pos, rev = temp_profiles()
    D = P.GIRDER_DEPTH
    ds = lambda z: np.where(z > D, -P.DIFF_SHRINKAGE, 0.0)
    return dict(temp_pos=eigenstress(pos, b_eff), temp_rev=eigenstress(rev, b_eff),
                diff_shr=eigenstress(ds, b_eff, P.DIFF_SHR_RELIEF))


# ---------------------------------------------------------------- prestress at a section
def prestress_state(x_bearing):
    """Forces of all cables at a section, at the key stages."""
    xg = np.atleast_1d(x_bearing + P.END_OVERHANG)
    imm = PS.immediate_losses(xg)
    cab = imm["cables"]
    out = dict(cables=cab, imm=imm, xg=xg)
    out["z"] = {c.name: c.z(xg) for c in cab}
    out["Ap"] = sum(c.Ap for c in cab)
    return out


def force_and_ecc(state, forces, names=None, ix=None):
    names = names or [c.name for c in state["cables"]]
    z = {n: (state["z"][n] if ix is None else state["z"][n][ix]) for n in names}
    Ptot = sum(forces[n] for n in names)
    zcp = sum(forces[n] * z[n] for n in names) / np.maximum(Ptot, 1e-9)
    return Ptot, zcp


def fibre(P_kN, zcp, M_pre, sec):
    """Stresses (MPa, compression +) at top & bottom of the precast section."""
    e = sec["zc"] - zcp
    D = P.GIRDER_DEPTH
    yt = D - sec["zc"]
    top = (P_kN / sec["A"] - P_kN * e * yt / sec["I"] + M_pre * yt / sec["I"]) / 1e3
    bot = (P_kN / sec["A"] + P_kN * e * sec["zc"] / sec["I"] - M_pre * sec["zc"] / sec["I"]) / 1e3
    return top, bot


# ---------------------------------------------------------------- ULS flexure, strain compatibility
def uls_moment_capacity(Pe_kN, dp, Ap, b_eff, eps_pe=None, prestress_area_factor=1.0):
    """Mid-span composite section: rectangular block 0.8x at fcd, strands bilinear (fp01k/1.15)."""
    Ap = Ap * prestress_area_factor
    fcd_s = 0.67 * P.FCK_DECK / P.GAMMA_C
    fcd_g = 0.67 * P.FCK_GIRDER / P.GAMMA_C
    if eps_pe is None:
        eps_pe = Pe_kN * 1e3 / (Ap * 1e6 * P.EP)        # kN -> N, m2 -> mm2
    fpd = P.FP01K / P.GAMMA_S
    ecu = 0.0035
    t = P.SLAB_T
    def comp_force(x):
        a = 0.8 * x
        if a <= t:
            return fcd_s * 1e3 * b_eff * a, a / 2
        # slab fully + girder top flange
        Fs = fcd_s * 1e3 * b_eff * t
        a2 = a - t
        bt = P.TOP_FLANGE_W if a2 <= P.TOP_FLANGE_T else P.WEB_T
        Fg = fcd_g * 1e3 * bt * a2
        return Fs + Fg, (Fs * t / 2 + Fg * (t + a2 / 2)) / (Fs + Fg)
    def steel(x):
        de = ecu * (dp - x) / x
        eps = eps_pe + de
        fp = min(P.EP * eps, fpd)
        return fp * 1e3 * Ap, eps, fp
    lo, hi = 0.01, dp
    for _ in range(80):
        x = 0.5 * (lo + hi)
        C, _ = comp_force(x)
        T, _, _ = steel(x)
        if C > T:
            hi = x
        else:
            lo = x
    C, yc = comp_force(x)
    T, eps, fp = steel(x)
    M = T * (dp - yc)
    return dict(x=x, M=M, fp=fp, eps_p=eps, T=T, lever=dp - yc, yielded=fp >= fpd - 1e-6,
                x_over_d=x / dp)


# ---------------------------------------------------------------- shear (IRC:112 cl. 10.3)
def shear_check(VEd, sec_pre, sec_comp, sigma_cp, bw, d, Vp=0.0, fywd=P.FY / P.GAMMA_S):
    fctk = 0.7 * P.fctm(P.FCK_GIRDER)
    fctd = fctk / P.GAMMA_C
    # first moment of area above centroid (composite), approximated with the web strip
    zc = sec_comp["zc"]
    z, b, E, dz = _composite_strips(sec_comp["b_eff"])
    m = np.where(z > P.GIRDER_DEPTH, sec_comp["m"], 1.0)
    above = z > zc
    Sx = np.sum((b * m * (z - zc) * dz)[above])
    VRdc = sec_comp["I"] * bw / Sx * np.sqrt(fctd ** 2 + sigma_cp * fctd) * 1e3       # kN
    Vnet = VEd - Vp
    z_lever = 0.9 * d
    cot = 2.0
    fcd = 0.67 * P.FCK_GIRDER / P.GAMMA_C
    nu1 = 0.6 * (1 - P.FCK_GIRDER / 310)
    acw = 1 + sigma_cp / fcd if sigma_cp <= 0.25 * fcd else 1.25
    VRdmax = acw * bw * z_lever * nu1 * fcd * 1e3 / (cot + 1 / cot)
    Asw_s = max(Vnet, 0) / (z_lever * fywd * 1e3 * cot) * 1e6                        # mm2/m
    rho_min = 0.072 * np.sqrt(P.FCK_GIRDER) / P.FY
    Asw_s_min = rho_min * bw * 1e6                                                    # mm2/m
    return dict(VEd=VEd, Vp=Vp, Vnet=Vnet, VRdc=VRdc, VRdmax=VRdmax,
                Asw_s=max(Asw_s, Asw_s_min), Asw_s_min=Asw_s_min, cot=cot)


# ---------------------------------------------------------------- bearings (IRC:83 Part II)
def bearing_check(N_max, N_min, v_x, alpha):
    a, b = P.BRG_A - 0.01, P.BRG_B - 0.01          # effective (minus side cover)
    ti, n = P.BRG_TI, P.BRG_LAYERS
    Te = P.BRG_TE
    Ar = a * b * (1 - v_x / a)
    S1 = a * b / (2 * ti * (a + b))
    G = 0.9e3
    ec = 1.5 * N_max / (G * Ar * S1)
    eq = v_x / Te
    ea = a ** 2 * alpha / (2 * ti * n * ti)
    et = ec + eq + ea
    smin = N_min / (a * b) / 1e3
    return dict(S=S1, eps_c=ec, eps_q=eq, eps_a=ea, eps_t=et, ok_total=et <= 7.0,
                ok_shear=eq <= 1.0, sigma_min=smin, ok_min=smin >= 3.0,
                sigma_max=N_max / (a * b) / 1e3)


# ---------------------------------------------------------------- girder number study (Courbon, 1-D)
def il_moment_max(veh, L=P.L, x=None, dx=0.05):
    x = L / 2 if x is None else x
    best = 0
    for xf in np.arange(0, L + veh.length + dx, dx):
        pos = xf - veh.ax
        ok = (pos >= 0) & (pos <= L)
        il = np.where(pos <= x, pos * (L - x) / L, x * (L - pos) / L)
        best = max(best, float(np.sum(veh.loads[ok] * il[ok])))
    return best


def option_study():
    """Compare 4, 5 and 6 girders across the 16 m deck (preliminary, Courbon)."""
    rows = []
    M70 = il_moment_max(V.R70_W) * (1 + P.IMPACT)
    MA = il_moment_max(V.CLASS_A) * (1 + P.IMPACT)
    for n, s, t_slab in ((4, 4.0, 0.275), (5, 3.25, 0.250), (6, 2.70, 0.225)):
        yg = (np.arange(n) - (n - 1) / 2) * s
        Sx2 = np.sum(yg ** 2)
        # 70R hugging the kerb (1.2 m), Class A beside it: resultant eccentricity
        y70 = -P.KERB_Y + 1.2 + V.R70_W.half_width
        yA = y70 + V.R70_W.half_width + V.GAP + V.CLASS_A.half_width
        W = V.R70_W.total + V.CLASS_A.total
        e = (V.R70_W.total * y70 + V.CLASS_A.total * yA) / W
        k_outer = (1 / n) * (1 + n * abs(e) * abs(yg[0]) / Sx2)
        M_LL = (M70 + MA) * (1 - P.LANE_REDUCTION) * k_outer
        trib = s / 2 + (P.DECK_WIDTH / 2 - yg[-1])
        w_slab = max(trib, s) * t_slab * P.GAMMA_RC
        # girder area scales roughly with the top flange (wider spacing -> wider flange)
        A_g = S.precast()["A"] + (min(s * 0.45, 1.6) - P.TOP_FLANGE_W) * P.TOP_FLANGE_T
        w_g = A_g * P.GAMMA_RC
        w_sidl = (2 * (P.BARRIER_LOAD + P.FOOTPATH_LOAD + P.RAILING_LOAD)) / n * 1.4 + P.WC_LOAD * P.CARRIAGEWAY / n
        M_pre = (w_g + w_slab) * P.L ** 2 / 8
        M_comp = w_sidl * P.L ** 2 / 8 + M_LL
        pre = S.precast(); comp = S.composite(max(s, 2.0))
        Zb_pre = pre["Zb"] * A_g / pre["A"]
        need = M_pre / Zb_pre + M_comp / comp["Zb"]                    # kPa at soffit
        e_p = pre["zc"] - 0.17
        Pe = need / (1 / A_g + e_p / Zb_pre)
        strands = Pe / (0.78 * P.JACK_RATIO * P.FPK * P.STRAND_AREA * 1e-3)
        rows.append(dict(n=n, spacing=s, slab=t_slab, k_outer=k_outer, M_LL=M_LL,
                         girder_t=w_g * P.L_GIRDER / 9.81, Pe=Pe, strands=strands,
                         strands_total=strands * n, slab_span=s - P.TOP_FLANGE_W,
                         concrete=n * A_g * P.L_GIRDER + P.DECK_WIDTH * t_slab * P.L_GIRDER))
    return rows
