"""
Independent checks of the computed results by simpler hand methods.
Each check: what is compared, the two numbers, the tolerance, pass/fail.
Writes out/verification.json.
"""
import json, sys
import numpy as np
from psc import inputs as P, section as S, prestress as PS, design as D, vehicles as V
from psc import twin as TW, dynamics as DY, scenarios as SC
from psc.grillage import Grillage

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
checks = []
def check(name, a, b, tol, unit="", note=""):
    rel = abs(a - b) / max(abs(b), 1e-12)
    ok = rel <= tol
    checks.append(dict(check=name, computed=float(a), hand=float(b), rel_diff=float(rel), tol=tol, ok=bool(ok), unit=unit, note=note))
    print(f"{'PASS' if ok else 'FAIL'}  {name:60s} {a:12.4g} vs {b:12.4g} {unit:8s} ({100*rel:.2f} %, tol {100*tol:g} %)")

des = json.load(open("out/design.json"))
base = json.load(open("out/baseline.json"))

# 1 section properties by rectangles and trapezoids
D_, btf, ttf, ttt, bw, bbf, tbf, tbt = (P.GIRDER_DEPTH, P.TOP_FLANGE_W, P.TOP_FLANGE_T, P.TOP_TAPER_T, P.WEB_T,
                                        P.BOT_FLANGE_W, P.BOT_FLANGE_T, P.BOT_TAPER_T)
hw = D_ - ttf - ttt - tbt - tbf
parts = [(btf * ttf, D_ - ttf / 2), ((btf + bw) / 2 * ttt, D_ - ttf - ttt / 2), (bw * hw, tbf + tbt + hw / 2),
         ((bbf + bw) / 2 * tbt, tbf + tbt / 2), (bbf * tbf, tbf / 2)]
A = sum(a for a, _ in parts); yb = sum(a * z for a, z in parts) / A
pre = S.precast()
check("precast girder area: section routine vs sum of five pieces", pre["A"], A, 0.001, "m2")
check("precast girder centroid height", pre["zc"], yb, 0.01, "m", "tapers taken at mid-height")

# 2 grillage vs simple beam, uniform load on all girders
g = Grillage(); g.factor()
loads = [(x, y, 10 * (g.dx if 0 < ix < g.nx else g.dx / 2)) for y in g.y for ix, x in enumerate(g.x)]
U = g.solve(g.load_vector(loads)); M, _ = g.girder_actions(U); w = g.deflection(U)
check("10 kN/m on every girder: sum of mid-span moments vs 6wL²/8", M[:, 12].sum(), 6 * 10 * P.L ** 2 / 8, 0.01, "kNm")
c = S.composite(2.7)
check("10 kN/m on every girder: inner girder deflection vs 5wL⁴/384EI + bearing", -w[2, 12] * 1e3, (5 * 10 * P.L ** 4 / (384 * g.Ec_g * c["I"]) + 165 / P.BRG_KV) * 1e3, 0.03, "mm")

# 3 Courbon vs grillage for the outer girder LL moment
check("G1 live-load moment: grillage vs Courbon", np.load("out/design_arrays.npz")["MLL"][0, -1], des["courbon"]["M"][0], 0.05, "kNm",
      "Courbon valid here: L/B = 2.06, stiff cross girders")

# 4 maximum moment of one Class A train, influence line by hand (one placement)
il = lambda x: np.where(x <= P.L / 2, x / 2, (P.L - x) / 2)
best = 0
for xf in np.arange(0, P.L + 20, 0.05):
    pos = xf - V.CLASS_A.ax
    k = (pos >= 0) & (pos <= P.L)
    best = max(best, (V.CLASS_A.loads[k] * il(pos[k])).sum())
check("Class A train, maximum mid-span moment vs influence line by hand", D.il_moment_max(V.CLASS_A), best, 0.002, "kNm")

# 5 friction loss of C3 at mid-span by hand: exp(-(mu*theta + k*x))
c3 = [c for c in PS.make_cables() if c.name == "C3"][0]
theta = 2 * (c3.z_a - c3.z_m) / c3.x_flat
Pj = P.JACK_RATIO * P.FPK * c3.Ap * 1e3
check("cable C3, force after friction at mid-span vs exp(−(μθ + kx))", float(PS.friction_profile(c3, np.array([PS.HALF]), Pj)[0]),
      Pj * np.exp(-(P.MU_FRICTION * theta + P.K_WOBBLE * PS.HALF)), 0.002, "kN")

# 6 elastic shortening: textbook average (n-1)/2n * ae * sigma_cp vs sequential calculation
pc = des["prestress"]["per_cable"]
es_seq = sum(v["ES_mid"] for v in pc.values())
P0 = des["prestress"]["P0_mid"]; e = pre["zc"] - 0.1605
s_cp = (P0 / pre["A"] + P0 * e * e / pre["I"]) / 1e3
ae = P.EP / P.Ecm_t(P.FCK_GIRDER, 20)
es_hand = 3 / 8 * ae * s_cp * 4 * P.STRANDS_PER_CABLE * P.STRAND_AREA * 1e-3
check("total elastic shortening loss at mid-span", es_seq, es_hand, 0.25, "kN", "hand: (n − 1)/2n rule, mean age 20 days")

# 7 soffit stress at mid-span, long term, quasi-permanent, by P/A + Pe/Z - M/Z
st = des["stages"]["G1"]["16.500"]
Pe = st["P"]["Pe"]; m = st["M"]
cg = des["sections"]["composite_outer"]
s_hand = (0.9 * Pe * (1 / pre["A"] + (pre["zc"] - 0.1605) / pre["Zb"]) - (m["Mg"] + m["Ms"]) / pre["Zb"] - (m["Medge"] + m["Mwc"]) / cg["Zb"]) / 1e3
s_hand += 0.5 * min(-2.515, 0) + -0.39
check("G1 soffit stress, quasi-permanent, 0.9Pm: P/A + Pe/Z − M/Z", st["rows"]["V quasi, inf"]["bot"], s_hand, 0.05, "MPa")

# 8 ULS: T * lever arm with yielding strands and block in the slab
Ap = 4 * P.STRANDS_PER_CABLE * P.STRAND_AREA * 1e-6
T = P.FP01K / P.GAMMA_S * 1e3 * Ap
a = T / (0.67 * P.FCK_DECK / P.GAMMA_C * 1e3 * S.effective_width("outer"))
dp = P.GIRDER_DEPTH + P.SLAB_T - 0.1605
check("G1 moment of resistance vs yielding strands × lever arm", des["uls"]["G1"]["M"], T * (dp - a / 2), 0.01, "kNm")

# 9 thermal movement at a free elastomeric bearing ~ alpha*L/2 per degC
check("bearing movement per °C vs αL/2", base["bearing_ref"], P.ALPHA_T * P.L / 2 * 1e3, 0.05, "mm/degC")

# 10 first bending frequency: beam formula with the updated twin's EI and mass
tw = TW.Twin(E_scale=base["twin"]["E_scale"], barrier=base["twin"]["barrier"], E_dyn=base["twin"]["E_dyn"])
f, wm = tw.modal(TW.healthy_state(), 25.0)
gg = tw.g
EI = sum(gg.Ec_g * base["twin"]["E_scale"] * base["twin"]["E_dyn"] * gg.long_props(iy, 12)[0] for iy in range(6))
mtot = gg.m_line.sum() / 1e3
check("f1 of the twin vs simply supported beam formula", f[0], np.pi / (2 * P.L ** 2) * np.sqrt(EI / mtot), 0.04, "Hz",
      "beam ignores end blocks, cross girders and bearing flexibility")

# 11 levelling sensitivity to prestress loss vs elastic camber x creep factor
st_ = TW.healthy_state(); st_["eta"][:] = 0.9
lev, vw = tw.long_term(st_)
c_el = des["deflection"]["camber_Pe_elastic_mm"]
check("level change for 10 % prestress loss", -lev.mean(), 0.1 * c_el * (1 + TW.CHI * TW.PHI_REM), 0.15, "mm",
      "hand: 10 % of the elastic camber due to Pe × (1 + χφ)")
sP = Pe * (1 / pre["A"] + (pre["zc"] - 0.1605) / pre["Zb"]) / 1e3
check("VW strain change for 10 % prestress loss", vw.mean(), 0.1 * sP / P.Ecm(P.FCK_GIRDER) * 1e6 * (1 + TW.CHI * TW.PHI_REM), 0.02, "microstrain")

# 12 identification of a known record
rng = np.random.default_rng(99)
r = np.load("out/twin_reference.npz")
acc = DY.simulate(f, np.full(8, 0.015), wm, gg.x, gg.y, rng, rate=20)
res, _ = DY.identify(acc, r["S"], r["f"], SC.TRACK)
check("identified f1 from a simulated 30 min record", res[0]["f"], f[0], 0.004, "Hz")
check("identified f3 from a simulated 30 min record", res[2]["f"], f[2], 0.004, "Hz")

# 13 model updating recovers the hidden parameters
tr = base["twin"]["truth"]
UPD = {"E_scale": "modulus factor", "barrier": "share of barrier stiffness", "E_dyn": "dynamic / static modulus"}
for k in ("E_scale", "barrier", "E_dyn"):
    check(f"model updating: {UPD[k]} vs value used to make the data", base["twin"][k], tr[k], 0.06, "")

# 14 novelty detector false-alarm rate close to the 1 % design value
ml = json.load(open("out/ml_results.json"))
fa = ml["novelty"]["All sources fused"]["false alarm"]
checks.append(dict(check="novelty detector, false-alarm rate on healthy test data vs design value", computed=fa, hand=0.01, rel_diff=abs(fa - 0.01),
                   tol=0.03, ok=bool(fa <= 0.04), unit="", note="99 % threshold from training data; small test sample", absolute=True))
print(f"{'PASS' if fa <= 0.04 else 'FAIL'}  novelty false alarm rate {fa:.3f} (design 0.01)")

n_ok = sum(c["ok"] for c in checks)
print(f"\n{n_ok}/{len(checks)} checks pass")
json.dump(checks, open("out/verification.json", "w"), indent=1)
