"""
Cable geometry and prestress losses (IRC:112-2020, Section 7.9 and Annex A2).

x is measured from the girder end (anchorage face), 0 ... L_GIRDER.
z is the height of the strand centroid above the girder soffit.
"""
import numpy as np
from . import inputs as P
from . import section as S

LG = P.L_GIRDER
HALF = LG / 2


class Cable:
    def __init__(self, name, z_a, z_m, x_flat, y_m=0.0, plan=(0.0, 0.0), stage=1,
                 n=P.STRANDS_PER_CABLE):
        self.name, self.z_a, self.z_m, self.x_flat = name, z_a, z_m, min(x_flat, HALF)
        self.y_m, self.plan, self.stage, self.n = y_m, plan, stage, n
        self.Ap = n * P.STRAND_AREA * 1e-6          # m2

    def _xx(self, x):
        return np.minimum(x, LG - x)

    def z(self, x, duct_ecc=True):
        xx = self._xx(np.asarray(x, float))
        s = np.clip(1 - xx / self.x_flat, 0, None)
        z = self.z_m + (self.z_a - self.z_m) * s ** 2
        if duct_ecc:
            # strands bear on the low side of the duct where the cable is draped
            z = z - P.DUCT_ECC_MID * (1 - s ** 2)
        return z

    def slope(self, x):
        """dz/dx (half-span from the nearer anchorage, sign for the left half)."""
        xx = self._xx(np.asarray(x, float))
        s = np.clip(1 - xx / self.x_flat, 0, None)
        return -2 * (self.z_a - self.z_m) * s / self.x_flat

    def y(self, x):
        xx = self._xx(np.asarray(x, float))
        a, b = self.plan
        if b <= a:
            return np.zeros_like(xx)
        t = np.clip((xx - a) / (b - a), 0, 1)
        return self.y_m * np.where(t < 0.5, 2 * t ** 2, 1 - 2 * (1 - t) ** 2)

    def angle(self, x):
        """Cumulative angular change from the nearer anchorage (rad)."""
        xx = self._xx(np.asarray(x, float))
        th_v = np.abs(np.arctan(self.slope(0.0)) - np.arctan(self.slope(xx)))
        a, b = self.plan
        th_h = np.zeros_like(xx)
        if b > a:
            Ld = b - a
            smax = 2 * abs(self.y_m) / Ld
            t = np.clip((xx - a) / Ld, 0, 1)
            sh = np.where(t < 0.5, 4 * abs(self.y_m) * t / Ld, 4 * abs(self.y_m) * (1 - t) / Ld)
            th_h = np.where(t < 0.5, sh, 2 * smax - sh)
        return th_v + th_h


def make_cables():
    # (name, z at anchorage, z at mid-span, end of parabola, lateral at mid-span, plan S-curve range, stage)
    return [
        Cable("C1", 0.35, 0.13, 5.0, -0.22, (1.0, 6.0), stage=2),
        Cable("C2", 0.75, 0.13, 8.0, +0.22, (4.5, 9.5), stage=1),
        Cable("C3", 1.15, 0.13, HALF, 0.0, (0, 0), stage=1),
        Cable("C4", 1.55, 0.30, HALF, 0.0, (0, 0), stage=2),
    ]


STAGE_TIME = {1: P.T_STRESS, 2: 28}
STRESS_ORDER = ["C2", "C3", "C1", "C4"]


# ---------------------------------------------------------------- immediate losses
def friction_profile(c, x, Pj):
    xx = np.minimum(x, LG - x)
    return Pj * np.exp(-(P.MU_FRICTION * c.angle(x) + P.K_WOBBLE * xx))


def slip_profile(c, x, Pj, slip=P.ANCHOR_SLIP):
    """Friction + draw-in, stressed from both ends (symmetric)."""
    xs = np.linspace(0, HALF, 2001)
    Pf = friction_profile(c, xs, Pj)
    EA = P.EP * 1e3 * c.Ap                      # kN
    target = slip * EA
    # find set length ls: 2 * int_0^ls (Pf - Pf(ls)) dx = slip*EA
    area = np.array([2 * np.trapezoid(Pf[:i + 1] - Pf[i], xs[:i + 1]) for i in range(len(xs))])
    if area[-1] >= target:
        i = int(np.searchsorted(area, target))
        ls = xs[i]
        Pset = np.where(xs < ls, 2 * Pf[i] - Pf, Pf)
    else:
        ls = HALF
        drop = (target - area[-1]) / (2 * HALF)
        Pset = 2 * Pf[-1] - Pf - drop
    xx = np.minimum(x, LG - x)
    return np.interp(xx, xs, Pset), ls


def section_at(x):
    """Precast section properties at x from girder end (end block interpolation)."""
    n, e = S.precast(False), S.precast(True)
    bw = S.web_width_at(np.asarray(x, float))
    f = (bw - P.WEB_T) / (P.BOT_FLANGE_W - P.WEB_T)
    out = {k: n[k] + f * (e[k] - n[k]) for k in ("A", "I", "zc", "perimeter")}
    return out


def immediate_losses(x, cables=None):
    """Returns per-cable force after friction+slip (Pfs), after elastic shortening (P0)."""
    cables = cables or make_cables()
    Pj = {c.name: P.JACK_RATIO * P.FPK * 1e3 * c.Ap for c in cables}   # kN
    Pfs = {}
    ls = {}
    for c in cables:
        Pfs[c.name], ls[c.name] = slip_profile(c, x, Pj[c.name])
    sec = section_at(x)
    A, I, zc = sec["A"], sec["I"], sec["zc"]
    by = {c.name: c for c in cables}
    loss = {c.name: np.zeros_like(np.asarray(x, float)) for c in cables}
    stressed = []
    ES1 = None
    for name in STRESS_ORDER:
        c = by[name]
        if c.stage == 2 and ES1 is None:
            ES1 = {k: v.copy() for k, v in loss.items()}
        t = STAGE_TIME[c.stage]
        ae = P.EP / P.Ecm_t(P.FCK_GIRDER, t)
        ei = zc - c.z(x)
        for j in stressed:
            ej = zc - by[j].z(x)
            sc = (Pfs[name] / A + Pfs[name] * ei * ej / I) / 1e3      # MPa at cable j
            loss[j] = loss[j] + ae * sc * by[j].Ap * 1e3               # kN
        stressed.append(name)
    P0 = {k: Pfs[k] - loss[k] for k in Pfs}
    return dict(Pj=Pj, Pfs=Pfs, P0=P0, ES=loss, ES1=ES1, ls=ls, cables=cables)


# ---------------------------------------------------------------- time dependent (Annex A2)
def _fcm_cyl(fck_cube):
    return 0.8 * fck_cube + 8.0


def creep_coefficient(t0, h0_mm, fck=P.FCK_GIRDER, RH=P.RH, t=1e5):
    fcm = _fcm_cyl(fck)
    a1, a2, a3 = (35 / fcm) ** 0.7, (35 / fcm) ** 0.2, (35 / fcm) ** 0.5
    phiRH = (1 + (1 - RH / 100) / (0.1 * h0_mm ** (1 / 3)) * a1) * a2
    b_fcm = 16.8 / np.sqrt(fcm)
    b_t0 = 1 / (0.1 + t0 ** 0.2)
    bH = min(1.5 * (1 + (0.012 * RH) ** 18) * h0_mm + 250 * a3, 1500 * a3)
    bc = ((t - t0) / (bH + t - t0)) ** 0.3
    return phiRH * b_fcm * b_t0 * bc, bH


def shrinkage_strain(t, ts, h0_mm, fck=P.FCK_GIRDER, RH=P.RH):
    fcm = _fcm_cyl(fck)
    fck_cyl = 0.8 * fck
    bRH = 1.55 * (1 - (RH / 100) ** 3)
    ecd0 = 0.85 * ((220 + 110 * 4) * np.exp(-0.12 * fcm / 10)) * 1e-6 * bRH
    kh = np.interp(h0_mm, [100, 200, 300, 500], [1.0, 0.85, 0.75, 0.70])
    bds = (t - ts) / ((t - ts) + 0.04 * np.sqrt(h0_mm ** 3))
    ecd = bds * kh * ecd0
    eca = (1 - np.exp(-0.2 * np.sqrt(t))) * 2.5 * (fck_cyl - 10) * 1e-6
    return ecd + eca


def relaxation_1000h(ratio):
    # low relaxation strand, IRC:112 cl. 6.3.6 values at 0.6/0.7/0.8 fpk
    return np.interp(ratio, [0.5, 0.6, 0.7, 0.8], [0.0, 1.0, 2.5, 4.5])


def long_term_losses(x, sig_qp_cp, imm=None, t_end=36500):
    """Creep + shrinkage + relaxation by the simplified interaction formula
    (IRC:112 eq. for time dependent losses, same form as EN 1992 eq. 5.46).
    sig_qp_cp: quasi-permanent concrete stress at tendon centroid (MPa, compression +)."""
    imm = imm or immediate_losses(x)
    cables = imm["cables"]
    sec = section_at(x)
    A, I, zc = sec["A"], sec["I"], sec["zc"]
    h0 = 2 * A / sec["perimeter"] * 1e3
    Ap_tot = sum(c.Ap for c in cables)
    zcp = sum(c.Ap * c.z(x) for c in cables) / Ap_tot
    ecc = zc - zcp
    Ec = P.Ecm(P.FCK_GIRDER)
    ae = P.EP / Ec
    out = {}
    for c in cables:
        t0 = STAGE_TIME[c.stage]
        phi, _ = creep_coefficient(t0, h0, t=t_end)
        ecs = shrinkage_strain(t_end, 3, h0) - shrinkage_strain(t0, 3, h0)
        sp0 = imm["P0"][c.name] / (c.Ap * 1e3)                         # MPa
        rho = relaxation_1000h(sp0 / P.FPK)
        dsr = 3 * rho / 100 * sp0                                      # final = 3 x 1000 h value
        num = ecs * P.EP + 0.8 * dsr + ae * phi * sig_qp_cp
        den = 1 + ae * Ap_tot / A * (1 + A * ecc ** 2 / I) * (1 + 0.8 * phi)
        dsig = num / den
        d = dict(phi=phi, ecs=ecs, relax=dsr, dsig=dsig, dP=dsig * c.Ap * 1e3, h0=h0)
        out[c.name] = {k: (np.asarray(v).item() if np.size(v) == 1 else v) for k, v in d.items()}
    return out


def loss_fraction(t, t0=P.T_STRESS, h0_mm=300.0):
    """Share of the long-term loss that has occurred by time t (creep time function)."""
    _, bH = creep_coefficient(t0, h0_mm)
    return ((t - t0) / (bH + t - t0)) ** 0.3 / ((1e5 - t0) / (bH + 1e5 - t0)) ** 0.3
