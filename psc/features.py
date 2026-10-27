"""
Feature extraction: raw campaign measurements -> physically meaningful features,
each expressed as a change from the structure's own healthy baseline.
"""
import csv
import numpy as np
from .twin import temperature_factors

REF_T = 25.0
GROUPS = {
    "VIB": ["r_f1", "r_f2", "r_f3", "r_f4", "mac1", "mac2", "phi_asym", "z1"],
    "LT":  ["r_d_mean", "r_d_asym", "r_d_max", "r_e_mean", "r_e_max"],
    "GEO": ["lev_mean", "lev_min", "lev_asym", "vw_mean", "vw_max"],
    "BRG": ["brg_min", "brg_asym"],
    "NDT": ["UPV_min", "R_min", "HCP_min", "rho_min", "cover_min", "carb_margin"],
    "VI":  ["crack_flex", "crack_long", "spall_area", "n_rust", "cs_girder", "cs_bearing", "joint_leak"],
}
SOURCES = {
    "Vibration only": GROUPS["VIB"],
    "Static + geometry + bearings": GROUPS["LT"] + GROUPS["GEO"] + GROUPS["BRG"],
    "Inspection + NDT": GROUPS["NDT"] + GROUPS["VI"],
    "Continuous monitoring (no closure)": GROUPS["VIB"] + GROUPS["GEO"] + GROUPS["BRG"],
    "All sources fused": sum(GROUPS.values(), []),
}
LABELS = {
    "r_f1": "f1 change (%)", "r_f2": "f2 change (%)", "r_f3": "f3 change (%)", "r_f4": "f4 change (%)",
    "mac1": "MAC mode 1", "mac2": "MAC mode 2", "phi_asym": "mode-1 edge asymmetry", "z1": "damping mode 1",
    "r_d_mean": "load-test deflection change (%)", "r_d_asym": "deflection asymmetry G1-G6 (%)",
    "r_d_max": "max girder deflection change (%)", "r_e_mean": "soffit strain change (%)",
    "r_e_max": "max girder strain change (%)", "lev_mean": "mid-span level change (mm)",
    "lev_asym": "level asymmetry G1-G6 (mm)", "lev_min": "largest sag of a girder (mm)", "vw_mean": "VW strain change (ue)", "vw_max": "max VW change (ue)",
    "brg_min": "min bearing movement ratio", "brg_asym": "bearing movement A-B (mm/C)",
    "UPV_min": "min UPV (km/s)", "R_min": "min rebound no.", "HCP_min": "most negative HCP (mV)",
    "rho_min": "min resistivity (kOhm cm)", "cover_min": "min cover (mm)", "carb_margin": "carbonation - cover (mm)",
    "crack_w": "max crack width (mm)", "crack_flex": "flexural crack width (mm)", "crack_long": "crack along bars/ducts (mm)", "spall_area": "spalled area (m2)", "n_rust": "rust-stain locations",
    "cs_girder": "girder condition state", "cs_bearing": "bearing condition state", "joint_leak": "joint leaking",
}


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    def conv(v):
        try:
            return float(v)
        except ValueError:
            return v
    return [{k: conv(v) for k, v in r.items()} for r in rows]


class Normaliser:
    """Healthy reference of one 'world' (twin library or the real bridge)."""
    def fit(self, rows):
        T = np.array([r["T"] for r in rows]); la = np.log(np.array([r["a_rms"] for r in rows]))
        X = np.column_stack([np.ones_like(T), T - REF_T, la])
        self.fb = {}
        for k in ("f1", "f2", "f3", "f4"):
            y = np.array([r[k] for r in rows])
            self.fb[k], *_ = np.linalg.lstsq(X, y, rcond=None)
        if "d1" in rows[0]:
            self.d_ref = np.array([[self._tc(r) * r[f"d{i}"] for i in range(1, 7)] for r in rows]).mean(axis=0)
            self.e_ref = np.array([[self._tc(r) * r[f"e{i}"] for i in range(1, 7)] for r in rows]).mean(axis=0)
        self.brg_ref = np.mean([0.5 * (r["brg_A"] + r["brg_B"]) for r in rows])
        return self

    def set_static_reference(self, d_ref, e_ref):
        self.d_ref, self.e_ref = np.asarray(d_ref), np.asarray(e_ref)
        return self

    @staticmethod
    def _tc(r):
        return temperature_factors(r["T"])[0] / temperature_factors(REF_T)[0]

    def transform(self, r):
        x = {}
        X = np.array([1.0, r["T"] - REF_T, np.log(r["a_rms"])])
        for k in ("f1", "f2", "f3", "f4"):
            fh = X @ self.fb[k]
            x[f"r_{k}"] = 100 * (r[k] - fh) / fh
        x["mac1"], x["mac2"] = r["mac1"], r["mac2"]
        x["phi_asym"] = r["phi_G1"] - r["phi_G6"]
        x["z1"] = r["z1"] if np.isfinite(r["z1"]) else 0.015
        if "d1" in r:
            tc = self._tc(r)
            d = np.array([tc * r[f"d{i}"] for i in range(1, 7)]) / self.d_ref * 100 - 100
            e = np.array([tc * r[f"e{i}"] for i in range(1, 7)]) / self.e_ref * 100 - 100
            x["r_d_mean"], x["r_d_asym"], x["r_d_max"] = d.mean(), d[0] - d[5], d.max()
            x["r_e_mean"], x["r_e_max"] = e.mean(), e.max()
            x["_d"] = d; x["_e"] = e
        x["lev_mean"] = (r["lev_G1"] + r["lev_C"] + r["lev_G6"]) / 3
        x["lev_asym"] = r["lev_G1"] - r["lev_G6"]
        x["lev_min"] = min(r["lev_G1"], r["lev_C"], r["lev_G6"])
        vw = np.array([r["vw_G1"], r["vw_G3"], r["vw_G4"], r["vw_G6"]])
        x["vw_mean"], x["vw_max"] = vw.mean(), vw.max()
        x["brg_min"] = min(r["brg_A"], r["brg_B"]) / self.brg_ref
        x["brg_asym"] = r["brg_A"] - r["brg_B"]
        for k in ("UPV_min", "R_min", "HCP_min", "rho_min", "cover_min", "carb_margin",
                  "crack_w", "crack_flex", "crack_long", "spall_area", "n_rust", "cs_girder", "cs_bearing", "joint_leak"):
            x[k] = r[k]
        return x


def matrix(feats, names):
    return np.array([[f[n] for n in names] for f in feats], float)
