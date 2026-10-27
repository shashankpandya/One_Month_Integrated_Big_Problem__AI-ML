"""
Grillage model of one span (6 girder lines x 25 node lines).

Node dof: [w (up), rx (rotation about X, along span), ry (about Y, transverse)].
Longitudinal members = composite girders, transverse members = deck slab strips,
replaced by T-beams (cross girder + slab) at the cross-girder lines.
Supports are elastomeric bearings: vertical springs, plus an optional
longitudinal restraint that couples the two end rotations of a girder line
(this is how a seized bearing shows up in a 2-D grillage, see report 1.11).
"""
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from . import inputs as P
from . import section as S

G_OVER_E = 1 / (2 * (1 + 0.2))


def _beam_local(EI, GJ, l):
    k = np.zeros((6, 6))
    b = EI / l ** 3 * np.array([[12, 6 * l, -12, 6 * l],
                                [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                                [-12, -6 * l, 12, -6 * l],
                                [6 * l, 2 * l * l, -6 * l, 4 * l * l]])
    ib = [0, 1, 3, 4]
    k[np.ix_(ib, ib)] = b
    k[np.ix_([2, 5], [2, 5])] = GJ / l * np.array([[1, -1], [-1, 1]])
    return k


def _mass_local(m, Im, l):
    M = np.zeros((6, 6))
    b = m * l / 420 * np.array([[156, 22 * l, 54, -13 * l],
                                [22 * l, 4 * l * l, 13 * l, -3 * l * l],
                                [54, 13 * l, 156, -22 * l],
                                [-13 * l, -3 * l * l, -22 * l, 4 * l * l]])
    ib = [0, 1, 3, 4]
    M[np.ix_(ib, ib)] = b
    M[np.ix_([2, 5], [2, 5])] = Im * l / 6 * np.array([[2, 1], [1, 2]])
    return M


def _T(direction):
    T = np.zeros((6, 6))
    for o in (0, 3):
        T[o, o] = 1
        if direction == "x":      # slope dw/dx = -ry, twist = rx
            T[o + 1, o + 2] = -1
            T[o + 2, o + 1] = 1
        else:                     # slope dw/dy = +rx, twist = ry
            T[o + 1, o + 1] = 1
            T[o + 2, o + 2] = 1
    return T


TX, TY = _T("x"), _T("y")


def tee_props(b_web, d_web, b_fl, t_fl):
    """Cross girder (rectangle below slab) acting with a slab flange."""
    pts = [(-b_web / 2, 0), (b_web / 2, 0), (b_web / 2, d_web), (b_fl / 2, d_web),
           (b_fl / 2, d_web + t_fl), (-b_fl / 2, d_web + t_fl), (-b_fl / 2, d_web), (-b_web / 2, d_web)]
    s = S.polygon_props(pts)
    J = S._rect_J(d_web, b_web) + b_fl * t_fl ** 3 / 6
    return s["I"], J, s["A"]


class Grillage:
    def __init__(self, nx=P.NX, cross_x=None):
        self.nx = nx
        self.ny = P.N_GIRDERS
        self.x = np.linspace(0, P.L, nx + 1)
        self.y = P.GIRDER_Y.copy()
        self.dx = P.L / nx
        self.cross_x = P.CROSS_X if cross_x is None else cross_x
        self.nn = (nx + 1) * self.ny
        self.ndof = 3 * self.nn
        self._sections()
        self.reset()

    # ------------------------------------------------------------- properties
    def _sections(self):
        Ec_g = P.Ecm(P.FCK_GIRDER) * 1e3      # kPa
        Ec_d = P.Ecm(P.FCK_DECK) * 1e3
        self.Ec_g, self.Ec_d = Ec_g, Ec_d
        inner, outer = S.effective_width("inner"), S.effective_width("outer")
        self.sec = {}
        for name, b in (("inner", inner), ("outer", outer)):
            n = S.composite(b, end_block=False)
            e = S.composite(b, end_block=True)
            self.sec[name] = (n, e)
        xm = (self.x[:-1] + self.x[1:]) / 2 + P.END_OVERHANG      # from girder end
        bw = S.web_width_at(xm)
        self.endblock_frac = (bw - P.WEB_T) / (P.BOT_FLANGE_W - P.WEB_T)

        # masses (kg/m) per girder line, SIDL of the edges kept on the outer lines
        g = 9.81
        A_pre = S.precast()["A"]
        trib = np.full(self.ny, P.GIRDER_SPACING)
        trib[[0, -1]] = P.GIRDER_SPACING / 2 + (P.DECK_WIDTH / 2 - P.GIRDER_Y[-1])
        self.trib = trib
        wc_w = np.clip(np.minimum(self.y + trib / 2, P.KERB_Y) -
                       np.maximum(self.y - trib / 2, -P.KERB_Y), 0, None)
        w_line = (A_pre + trib * P.SLAB_T) * P.GAMMA_RC + wc_w * P.WC_LOAD     # kN/m
        edge = P.BARRIER_LOAD + P.FOOTPATH_LOAD + P.RAILING_LOAD
        w_line[[0, -1]] += edge
        self.w_line = w_line
        self.m_line = w_line / g * 1e3
        # rotary inertia (kg m2/m): slab strip + edge items at their lever arms
        Im = self.m_line * 0.0 + (trib * P.SLAB_T * P.GAMMA_RC / g * 1e3) * trib ** 2 / 12
        yo = P.GIRDER_Y[-1]
        arms = [(P.BARRIER_LOAD, P.KERB_Y + P.CRASH_BARRIER_W / 2 - yo),
                (P.FOOTPATH_LOAD, P.KERB_Y + P.CRASH_BARRIER_W + P.FOOTPATH_W / 2 - yo),
                (P.RAILING_LOAD, P.DECK_WIDTH / 2 - P.RAILING_W / 2 - yo)]
        Im_edge = sum(w / g * 1e3 * a ** 2 for w, a in arms)
        Im[[0, -1]] += Im_edge
        Im += S.precast()["Iz"] * P.GAMMA_RC / g * 1e3
        self.Im_line = Im
        self.m_endblock = (S.precast(True)["A"] - A_pre) * P.GAMMA_RC / g * 1e3

    def reset(self):
        """Healthy, as-designed state."""
        self.E_factor = 1.0                      # global (temperature, ageing)
        self.EI_fac = np.ones((self.ny, self.nx))  # longitudinal element multipliers
        self.kv = np.full((self.ny, 2), P.BRG_KV)
        G = P.BRG_G
        kh = G * P.BRG_A * P.BRG_B / P.BRG_TE
        self.kh = np.full((self.ny, 2), kh)      # bearing shear stiffness, per bearing
        self.ksub = np.array([P.K_SUB_PIER, P.K_SUB_PIER])
        self.mass_fac = 1.0
        self.barrier_part = 0.0                  # share of crash-barrier stiffness acting with deck
        self.dynamic = False

    # ------------------------------------------------------------- assembly
    def node(self, ix, iy):
        return iy * (self.nx + 1) + ix

    def dofs(self, n):
        return [3 * n, 3 * n + 1, 3 * n + 2]

    def barrier_EI_ratio(self, iy):
        """Extra I (girder-concrete units) of a crash barrier hung on the outer line."""
        if iy not in (0, self.ny - 1) or self.barrier_part == 0:
            return 0.0
        n, _ = self.sec["outer"]
        A_b, h_b, I_b = 0.28, 0.384, 0.017
        d = P.GIRDER_DEPTH + P.SLAB_T + h_b - n["yb"]
        m = P.Ecm(P.FCK_DECK) / P.Ecm(P.FCK_GIRDER)
        return self.barrier_part * m * (I_b + A_b * d * d)

    def long_props(self, iy, ex):
        name = "outer" if iy in (0, self.ny - 1) else "inner"
        n, e = self.sec[name]
        f = self.endblock_frac[ex]
        I = n["I"] + f * (e["I"] - n["I"]) + self.barrier_EI_ratio(iy)
        J = n["J"] + f * (e["J"] - n["J"])
        A = n["A"] + f * (e["A"] - n["A"])
        yb = n["yb"] + f * (e["yb"] - n["yb"])
        return I, J, A, yb

    def assemble(self, with_mass=False):
        Ef = self.E_factor * (P.E_DYN_FACTOR if self.dynamic else 1.0)
        Eg, Ed = self.Ec_g * Ef, self.Ec_d * Ef
        K = np.zeros((self.ndof, self.ndof))
        M = np.zeros((self.ndof, self.ndof)) if with_mass else None
        dx = self.dx
        # longitudinal girders
        for iy in range(self.ny):
            for ex in range(self.nx):
                I, J, A, _ = self.long_props(iy, ex)
                kl = _beam_local(Eg * I * self.EI_fac[iy, ex], Eg * G_OVER_E * J, dx)
                d = self.dofs(self.node(ex, iy)) + self.dofs(self.node(ex + 1, iy))
                K[np.ix_(d, d)] += TX.T @ kl @ TX
                if with_mass:
                    m = (self.m_line[iy] + self.endblock_frac[ex] * self.m_endblock) * self.mass_fac
                    ml = _mass_local(m, self.Im_line[iy] * self.mass_fac, dx)
                    M[np.ix_(d, d)] += TX.T @ ml @ TX
        # transverse: slab strips, T-beams at cross girders
        for ix in range(self.nx + 1):
            trib = dx if 0 < ix < self.nx else dx / 2
            xc = self.x[ix]
            is_cross = any(abs(xc - c) < 1e-6 for c in self.cross_x)
            if is_cross:
                end = ix in (0, self.nx)
                bw, dw = (P.END_DIAPH_B, P.END_DIAPH_D - P.SLAB_T) if end else (P.CROSS_B, P.CROSS_D)
                I, J, A = tee_props(bw, dw, trib, P.SLAB_T)
                m_cg = bw * dw * P.GAMMA_RC / 9.81 * 1e3
            else:
                I, J = trib * P.SLAB_T ** 3 / 12, trib * P.SLAB_T ** 3 / 6
                m_cg = 0.0
            for iy in range(self.ny - 1):
                l = self.y[iy + 1] - self.y[iy]
                kl = _beam_local(Ed * I, Ed * G_OVER_E * J, l)
                d = self.dofs(self.node(ix, iy)) + self.dofs(self.node(ix, iy + 1))
                K[np.ix_(d, d)] += TY.T @ kl @ TY
                if with_mass and m_cg > 0:
                    ml = _mass_local(m_cg * self.mass_fac, m_cg * bw ** 2 / 12, l)
                    M[np.ix_(d, d)] += TY.T @ ml @ TY
        # bearings
        for iy in range(self.ny):
            nA, nB = self.node(0, iy), self.node(self.nx, iy)
            K[3 * nA, 3 * nA] += self.kv[iy, 0]
            K[3 * nB, 3 * nB] += self.kv[iy, 1]
            kr = self.rot_restraint(iy, Eg)
            dA, dB = 3 * nA + 2, 3 * nB + 2
            K[np.ix_([dA, dB], [dA, dB])] += kr * np.array([[1, -1], [-1, 1]])
        self.K, self.M = K, M
        return K, M

    def chain_stiffness(self, iy, Eg=None):
        """Series stiffness of substructure - bearing - deck - bearing - substructure."""
        Eg = Eg or self.Ec_g
        _, _, A, _ = self.long_props(iy, self.nx // 2)
        flex = (1 / self.kh[iy, 0] + 1 / self.ksub[0] + 1 / self.kh[iy, 1] + 1 / self.ksub[1]
                + P.L / (Eg * A))
        return 1 / flex

    def rot_restraint(self, iy, Eg=None):
        _, _, _, yb = self.long_props(iy, self.nx // 2)
        return self.chain_stiffness(iy, Eg) * yb ** 2

    def bearing_slopes(self, dT=1.0):
        """Bearing shear deformation per degC at end A and end B (mm/degC), per girder line."""
        out = np.zeros((self.ny, 2))
        for iy in range(self.ny):
            H = P.ALPHA_T * P.L * dT * self.chain_stiffness(iy)
            out[iy] = H / self.kh[iy] * 1e3
        return out

    # ------------------------------------------------------------- solutions
    def factor(self):
        self.assemble(with_mass=False)
        self._cf = cho_factor(self.K)

    def solve(self, F):
        return cho_solve(self._cf, F)

    def load_vector(self, loads):
        """loads: iterable of (x, y, Fz_down[kN]); bilinear share to grid nodes,
        loads outside the outer girder lines go to the outer line as force + torque."""
        F = np.zeros(self.ndof)
        for x, y, p in loads:
            if x < 0 or x > P.L:
                continue
            fx = np.clip(x / self.dx, 0, self.nx - 1e-9)
            ix = int(fx)
            ax = fx - ix
            if y <= self.y[0] or y >= self.y[-1]:
                iy = 0 if y <= self.y[0] else self.ny - 1
                arm = y - self.y[iy]
                for jx, wx in ((ix, 1 - ax), (ix + 1, ax)):
                    n = self.node(jx, iy)
                    F[3 * n] -= p * wx
                    F[3 * n + 1] -= p * wx * arm      # torque about X: +rx turns +y edge up
                continue
            iy = np.searchsorted(self.y, y) - 1
            ay = (y - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
            for jx, wx in ((ix, 1 - ax), (ix + 1, ax)):
                for jy, wy in ((iy, 1 - ay), (iy + 1, ay)):
                    F[3 * self.node(jx, jy)] -= p * wx * wy
        return F

    def girder_actions(self, U):
        """Sagging moment (kNm) and shear (kN, V = dM/dx) at every node of every
        girder line. U may be (ndof,) or (ndof, ncases)."""
        U2 = U if U.ndim == 2 else U[:, None]
        nc = U2.shape[1]
        Eg = self.Ec_g * self.E_factor * (P.E_DYN_FACTOR if self.dynamic else 1.0)
        Ma = np.zeros((self.ny, self.nx, nc)); Mb = np.zeros_like(Ma)
        Va = np.zeros_like(Ma); Vb = np.zeros_like(Ma)
        for iy in range(self.ny):
            for ex in range(self.nx):
                I, J, A, _ = self.long_props(iy, ex)
                kl = _beam_local(Eg * I * self.EI_fac[iy, ex], Eg * G_OVER_E * J, self.dx)
                d = self.dofs(self.node(ex, iy)) + self.dofs(self.node(ex + 1, iy))
                f = kl @ (TX @ U2[d])
                Ma[iy, ex], Mb[iy, ex] = -f[1], f[4]
                Va[iy, ex], Vb[iy, ex] = f[0], -f[3]
        Mn = np.zeros((self.ny, self.nx + 1, nc))
        Mn[:, 0], Mn[:, -1] = Ma[:, 0], Mb[:, -1]
        Mn[:, 1:-1] = 0.5 * (Mb[:, :-1] + Ma[:, 1:])
        Vn = np.zeros_like(Mn)
        Vn[:, 0], Vn[:, -1] = Va[:, 0], Vb[:, -1]
        left, right = Vb[:, :-1], Va[:, 1:]
        Vn[:, 1:-1] = np.where(np.abs(left) > np.abs(right), left, right)
        if U.ndim == 1:
            return Mn[..., 0], Vn[..., 0]
        return Mn, Vn

    def deflection(self, U):
        w = U[0::3]
        if U.ndim == 1:
            return w.reshape(self.ny, self.nx + 1)
        return w.reshape(self.ny, self.nx + 1, -1)

    def reactions(self, U):
        w = self.deflection(U)
        RA = -w[:, 0] * self.kv[:, 0, None] if w.ndim == 3 else -w[:, 0] * self.kv[:, 0]
        RB = -w[:, -1] * self.kv[:, 1, None] if w.ndim == 3 else -w[:, -1] * self.kv[:, 1]
        return RA, RB

    def soffit_strain(self, U):
        """Bottom-fibre strain (microstrain, tension +) at all nodes."""
        Mn, _ = self.girder_actions(U)
        Eg = self.Ec_g * self.E_factor
        eps = np.zeros_like(Mn)
        for iy in range(self.ny):
            for ix in range(self.nx + 1):
                ex = min(ix, self.nx - 1)
                I, _, _, yb = self.long_props(iy, ex)
                fac = self.EI_fac[iy, min(ix, self.nx - 1)]
                if 0 < ix < self.nx:
                    fac = 0.5 * (self.EI_fac[iy, ix - 1] + self.EI_fac[iy, ix])
                eps[iy, ix] = Mn[iy, ix] * yb / (Eg * I * fac) * 1e6
        return eps

    def modes(self, n=8):
        self.dynamic = True
        K, M = self.assemble(with_mass=True)
        self.dynamic = False
        # K in kN/m, M assembled in kg -> tonnes so that omega^2 is in 1/s^2
        lam, phi = eigh(K, M * 1e-3, subset_by_index=[0, n - 1])
        f = np.sqrt(np.maximum(lam, 0)) / (2 * np.pi)
        # phi is mass normalised (tonne units); vertical component reshaped
        w = phi[0::3].reshape(self.ny, self.nx + 1, n)
        return f, phi, w

    def influence(self):
        """Unit downward load at each grid node -> responses (for vehicle search)."""
        self.factor()
        F = np.zeros((self.ndof, self.nn))
        for n in range(self.nn):
            F[3 * n, n] = -1.0
        U = self.solve(F)
        Mn, Vn = self.girder_actions(U)
        w = self.deflection(U)
        RA, RB = self.reactions(U)
        # reshape load index to (iy_load, ix_load)
        shp = (self.ny, self.nx + 1, self.ny, self.nx + 1)
        return dict(M=Mn.reshape(shp), V=Vn.reshape(shp), w=w.reshape(shp),
                    RA=RA.reshape(self.ny, self.ny, self.nx + 1),
                    RB=RB.reshape(self.ny, self.ny, self.nx + 1))


def mode_label(wshape):
    """Crude classification of a mode from its vertical shape (ny, nx+1)."""
    mid = wshape.shape[1] // 2
    tr = wshape[:, mid] if np.abs(wshape[:, mid]).max() > 0.2 * np.abs(wshape).max() else wshape[:, mid // 2]
    lg = wshape[wshape.shape[0] // 2 if wshape.shape[0] % 2 else 2, :]
    nlong = int(np.sum(np.diff(np.sign(lg[1:-1][np.abs(lg[1:-1]) > 1e-9])) != 0)) + 1
    ntr = int(np.sum(np.diff(np.sign(tr[np.abs(tr) > 1e-3 * np.abs(tr).max()])) != 0))
    kind = {0: "bending", 1: "torsion"}.get(ntr, f"transverse-{ntr}")
    return f"{kind} (long. {nlong})"
