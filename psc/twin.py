"""
Digital twin of span S2: a deterioration state -> what the instruments would see.

State (all optional, healthy by default)
  eta[6]          ratio of the actual to the expected effective prestress, per girder
  strand[]        (girder, x0, x1, area loss)   local strand section loss (corrosion)
  stiff[]         (girder, x0, x1, d_static, d_dynamic)  local loss of EI (cracking, impact, spalling)
  conc[]          (girder, x0, x1, dE)          concrete deterioration (modulus loss)
  kh_mult[6,2]    bearing shear stiffness multiplier at ends A (x=0) and B (x=L)
"""
import json
import copy
import numpy as np
from . import inputs as P
from . import section as S
from . import dynamics as DY
from .grillage import Grillage

ROOT = __file__.rsplit("psc", 1)[0]
CHI = 0.8                # ageing coefficient for creep under a slowly developing change
PHI_REM = 1.0            # creep still to come when the change develops (mid-life)
TEST_TRUCK_Y = (-1.85, 1.85)     # lorries two abreast, symmetric about the deck centreline
TEST_TRUCK = dict(P=np.array([60.0, 120.0, 120.0]), ax=np.array([0.0, 3.8, 5.15]))   # 300 kN, 3-axle
TANDEM_GAP = 7.5                 # front axle to front axle of the two lorries in a lane
R_CRACKED = 0.35         # Icr / Ig of a cracked PSC section (bonded tendons)


def healthy_state():
    return dict(eta=np.ones(P.N_GIRDERS), strand=[], stiff=[], conc=[],
                kh_mult=np.ones((P.N_GIRDERS, 2)))


def temperature_factors(T):
    """Concrete modulus and elastomer shear modulus vs effective bridge temperature."""
    E = 1.0 - 0.0025 * (T - 25.0)
    G = np.clip(1.0 + 0.015 * (23.0 - T), 0.8, 1.6)
    return E, G


class Twin:
    def __init__(self, E_scale=1.0, barrier=0.0, E_dyn=P.E_DYN_FACTOR, design=None):
        self.g = Grillage()
        self.E_scale, self.barrier, self.E_dyn = E_scale, barrier, E_dyn
        self.nx, self.ny = self.g.nx, self.g.ny
        self.xm = (self.g.x[:-1] + self.g.x[1:]) / 2         # element mid-points
        d = design or json.load(open(ROOT + "out/design.json"))
        self.design = d
        arr = np.load(ROOT + "out/design_arrays.npz")
        # permanent moments per girder along the span (kNm), precast and composite parts
        self.M_pre = arr["Mg"] + arr["Ms"]                   # (6, 25)
        self.M_comp = arr["Medge"] + arr["Mwc"]
        self.M_LL = np.concatenate([arr["MLL"], arr["MLL"][:, -2::-1]], axis=1)   # full span
        Pe = d["prestress"]["Pe_mid"]
        self.Pe = Pe
        pre = S.precast()
        self.pre = pre
        self.comp = [S.composite(S.effective_width("outer" if iy in (0, 5) else "inner")) for iy in range(6)]
        self.zcp = 0.1605
        self.e_p = pre["zc"] - self.zcp
        self.sigP_bot = Pe * (1 / pre["A"] + self.e_p / pre["Zb"]) / 1e3        # MPa, comp +
        self.Ec = P.Ecm(P.FCK_GIRDER)
        self.phi = d["deflection"]["phi"]
        self.ref_modes = None
        self.sig_rev_bot = float(arr["temp_rev"][0])         # reverse temperature difference, soffit

    # ------------------------------------------------------------- state -> model
    def _local_eta(self, st):
        """prestress ratio at element mid-points, (ny, nx)."""
        eta = np.repeat(np.asarray(st["eta"], float)[:, None], self.nx, axis=1)
        for iy, x0, x1, c in st["strand"]:
            m = (self.xm >= x0) & (self.xm <= x1)
            eta[iy, m] *= (1 - c)
        return eta

    def soffit_stress(self, st, M_extra=None, mean_prestress=True):
        """Mid-element soffit stress (MPa, comp +) under permanent loads + M_extra (composite)."""
        eta = self._local_eta(st)
        xmid = self.xm
        Mp = np.array([np.interp(xmid, self.g.x, self.M_pre[iy]) for iy in range(self.ny)])
        Mc = np.array([np.interp(xmid, self.g.x, self.M_comp[iy]) for iy in range(self.ny)])
        # prestress eccentricity falls off towards the supports (parabolic cables)
        shape = 1 - 0.75 * ((xmid - P.L / 2) / (P.L / 2)) ** 2
        sp = (self.Pe * eta * (1 / self.pre["A"] + self.e_p * shape / self.pre["Zb"])) / 1e3
        s = sp - Mp / self.pre["Zb"] / 1e3
        for iy in range(self.ny):
            s[iy] -= Mc[iy] / self.comp[iy]["Zb"] / 1e3
            if M_extra is not None:
                s[iy] -= M_extra[iy] / self.comp[iy]["Zb"] / 1e3
        return s

    def cracking(self, st):
        """Has the girder cracked in service (rare traffic)? returns mask (ny, nx)."""
        Mll = np.array([np.interp(self.xm, self.g.x, self.M_LL[iy]) for iy in range(self.ny)])
        # heavy (not extreme) traffic on a cool night: 0.8 x design LL + 0.6 x reverse gradient
        s_rare = self.soffit_stress(st, M_extra=0.8 * Mll) + 0.6 * self.sig_rev_bot
        return s_rare < -P.fctm(P.FCK_GIRDER)

    def _ieff(self, Ma, Mdec):
        r = np.clip(Mdec / np.maximum(Ma, 1e-9), 0, 1) ** 3
        return r + (1 - r) * R_CRACKED

    def configure(self, st, T=25.0, dynamic=False, M_test=None):
        g = self.g
        g.reset()
        g.barrier_part = self.barrier
        Ef, Gf = temperature_factors(T)
        g.E_factor = self.E_scale * Ef * (self.E_dyn / P.E_DYN_FACTOR if dynamic else 1.0)
        fac = np.ones((self.ny, self.nx))
        for iy, x0, x1, dE in st["conc"]:
            m = (self.xm >= x0) & (self.xm <= x1)
            fac[iy, m] *= (1 - dE)
        for iy, x0, x1, ds, dd in st["stiff"]:
            m = (self.xm >= x0) & (self.xm <= x1)
            fac[iy, m] *= (1 - (dd if dynamic else ds))
        # flexural cracks caused by loss of precompression. Once formed they reopen
        # only when the soffit is decompressed again.
        cracked = self.cracking(st)
        if cracked.any():
            s_perm = self.soffit_stress(st)
            for iy in range(self.ny):
                Zb = self.comp[iy]["Zb"]
                for e in np.where(cracked[iy])[0]:
                    if M_test is not None:
                        M_dec = max(s_perm[iy, e], 0.0) * Zb * 1e3    # extra moment to decompress
                        if M_test[iy, e] > M_dec:
                            fac[iy, e] *= self._ieff(M_test[iy, e], M_dec)
                    elif s_perm[iy, e] < 0:        # open even at rest -> seen by ambient vibration
                        fac[iy, e] *= 1 - (1 - R_CRACKED) * min(1.0, -s_perm[iy, e] / 3.0)
        g.EI_fac = fac
        g.kh = g.kh * Gf * np.asarray(st["kh_mult"], float)
        g.mass_fac = st.get("mass", 1.0)
        return g

    # ------------------------------------------------------------- responses
    def modal(self, st, T=25.0, n=8):
        g = self.configure(st, T, dynamic=True)
        f, phi, w = g.modes(n)
        return f, w

    def test_load_vector(self, x_front=None):
        """Four 300 kN lorries: two abreast, two in tandem, group centred at mid-span."""
        g = self.g
        cg = (TEST_TRUCK["P"] * TEST_TRUCK["ax"]).sum() / TEST_TRUCK["P"].sum()
        x_front = P.L / 2 + cg + TANDEM_GAP / 2 if x_front is None else x_front
        loads = []
        for yc in TEST_TRUCK_Y:
            for lead in (0.0, TANDEM_GAP):
                for a, Pa in zip(TEST_TRUCK["ax"], TEST_TRUCK["P"]):
                    for dy in (-0.9, 0.9):
                        loads.append((x_front - lead - a, yc + dy, Pa / 2))
        return g.load_vector(loads)

    def load_test(self, st, T=25.0, iters=3):
        """Mid-span deflection (mm, down +) and soffit strain (microstrain) of the 6 girders."""
        M_test = None
        for _ in range(iters):
            g = self.configure(st, T, dynamic=False, M_test=M_test)
            g.factor()
            U = g.solve(self.test_load_vector())
            Mn, _ = g.girder_actions(U)
            M_new = 0.5 * (Mn[:, :-1] + Mn[:, 1:])
            if M_test is not None and np.allclose(M_new, M_test, rtol=1e-3, atol=1.0):
                break
            M_test = M_new
            if not self.cracking(st).any():
                break
        w = -g.deflection(U)[:, P.NX // 2] * 1e3
        eps = g.soffit_strain(U)[:, P.NX // 2]
        return w, eps

    def long_term(self, st):
        """Changes since the healthy expectation: mid-span level (mm, sag -) and
        soffit VW strain (microstrain, tension +) per girder."""
        x = self.xm
        dx = self.g.dx
        mb = np.where(x <= P.L / 2, x / 2, (P.L - x) / 2)
        eta = self._local_eta(st)
        shape = 1 - 0.75 * ((x - P.L / 2) / (P.L / 2)) ** 2
        g = self.configure(st, 25.0)
        lev = np.zeros(self.ny); vw = np.zeros(self.ny)
        creep = 1 + CHI * PHI_REM
        for iy in range(self.ny):
            I = self.comp[iy]["I"]
            EI = self.Ec * 1e3 * I
            Mperm = np.interp(x, self.g.x, self.M_pre[iy] + self.M_comp[iy])
            dP = self.Pe * (1 - eta[iy])
            dkap = dP * self.e_p * shape / (self.Ec * 1e3 * self.pre["I"])       # loss of hogging
            dkap += Mperm / EI * (1 / g.EI_fac[iy] - 1)                         # softer girder
            lev[iy] = -np.sum(dkap * mb) * dx * creep * 1e3
            e = P.NX // 2
            sig_bot = dP[e] * (1 / self.pre["A"] + self.e_p / self.pre["Zb"]) / 1e3
            vw[iy] = sig_bot / self.Ec * 1e6 * creep
            vw[iy] += Mperm[e] * self.comp[iy]["yb"] / EI * (1 / g.EI_fac[iy, e] - 1) * 1e6 * creep
        return lev, vw

    def bearings(self, st, T=25.0):
        g = self.configure(st, T)
        return g.bearing_slopes()

    def reference(self, n=8, T=25.0):
        f, w = self.modal(healthy_state(), T, n)
        self.ref_modes = (f, w, DY.sensor_shapes(w))
        return self.ref_modes
