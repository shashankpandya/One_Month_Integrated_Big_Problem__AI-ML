"""
Deterioration scenarios and one complete measurement campaign on span S2:
ambient/traffic vibration, a static load test, levelling and VW strain,
bearing movement, NDT and a visual inspection.
"""
import numpy as np
from . import inputs as P
from . import twin as TW
from . import dynamics as DY
from . import condition as C

MECHS = ("PL", "CD", "CR", "CO", "BR", "LS")
MECH_NAMES = {"PL": "Loss of prestress", "CD": "Concrete deterioration", "CR": "Flexural cracking",
              "CO": "Corrosion (strand / rebar)", "BR": "Bearing deterioration", "LS": "Local stiffness loss"}
TRACK = (0, 1, 2, 4)       # B1, T1, TR1, B2 in the mode list of the twin


# ---------------------------------------------------------------- environment
def site_temperature(rng, day=None, hour=None):
    """Effective bridge temperature for a coastal-plain site in eastern India."""
    day = rng.uniform(0, 365) if day is None else day
    hour = rng.choice([2, 14]) if hour is None else hour
    seasonal = 29 + 7 * np.sin(2 * np.pi * (day - 105) / 365)
    daily = -3.5 if hour < 8 else 3.5
    monsoon = 1.0 if 160 < day < 275 else 0.0
    T = seasonal + daily - 2.0 * monsoon + rng.normal(0, 1.5)
    return float(T), monsoon


def traffic_rate(rng, hour):
    return float(rng.uniform(3, 60) if hour < 8 else rng.uniform(150, 420))


# ---------------------------------------------------------------- mechanisms
def sample_mechanism(t, rng):
    if t == "PL":
        loss = rng.uniform(0.03, 0.35)
        if rng.random() < 0.4:                     # excess long-term loss in every girder
            gs = list(range(P.N_GIRDERS))
            eta = 1 - loss * rng.uniform(0.8, 1.2, P.N_GIRDERS)
        else:                                      # one girder (grout voids, anchorage problem)
            gs = [int(rng.integers(P.N_GIRDERS))]
            eta = np.ones(P.N_GIRDERS); eta[gs[0]] = 1 - loss
        return dict(type="PL", girders=gs, loss=loss, eta=np.clip(eta, 0.5, 1.0), severity=loss)
    if t == "CD":
        g = int(rng.choice(P.N_GIRDERS, p=[0.25, 0.125, 0.125, 0.125, 0.125, 0.25]))
        if rng.random() < 0.4:
            x0, x1 = 0.0, P.L
        else:
            ln = rng.uniform(6, 15); c = rng.uniform(ln / 2, P.L - ln / 2)
            x0, x1 = c - ln / 2, c + ln / 2
        dE = rng.uniform(0.05, 0.30)
        return dict(type="CD", girder=g, x0=x0, x1=x1, dE=dE, severity=dE)
    if t == "CR":
        g = int(rng.integers(P.N_GIRDERS))
        ln = rng.uniform(4, 10); c = P.L / 2 + rng.uniform(-4, 4)
        ds = rng.uniform(0.15, 0.50)
        return dict(type="CR", girder=g, x0=c - ln / 2, x1=c + ln / 2, d_static=ds,
                    d_dyn=ds * rng.uniform(0.2, 0.5), w=0.05 + 0.5 * ds * rng.uniform(0.6, 1.2), severity=ds)
    if t == "CO":
        g = int(rng.choice([0, P.N_GIRDERS - 1])) if rng.random() < 0.8 else int(rng.integers(1, P.N_GIRDERS - 1))
        ln = rng.uniform(2, 8)
        if rng.random() < 0.5:            # wet joint end
            c = rng.choice([rng.uniform(1, 4), P.L - rng.uniform(1, 4)])
        else:                             # under a leaking drain spout
            c = rng.choice(C.DRAIN_X) + rng.uniform(-1, 1)
        loss = rng.uniform(0.02, 0.25)
        return dict(type="CO", girder=g, x0=max(c - ln / 2, 0), x1=min(c + ln / 2, P.L), loss=loss,
                    d_spall=0.25 * loss, severity=loss)
    if t == "BR":
        which = rng.choice(["both", "A", "B"], p=[0.4, 0.3, 0.3])
        mult = np.ones((P.N_GIRDERS, 2))
        lvl = 10 ** rng.uniform(np.log10(5), np.log10(500))
        for e, flag in ((0, which in ("both", "A")), (1, which in ("both", "B"))):
            if flag:
                mult[:, e] = lvl * rng.uniform(0.5, 1.5, P.N_GIRDERS) * (rng.uniform(0.3, 1.0) if which == "both" and e == 1 else 1)
        return dict(type="BR", ends=str(which), kh_mult=mult, severity=float(np.log10(lvl)),
                    joint_leak=bool(rng.random() < 0.6))
    if t == "LS":
        g = int(rng.choice([0, P.N_GIRDERS - 1])) if rng.random() < 0.6 else int(rng.integers(P.N_GIRDERS))
        ln = rng.uniform(1.0, 3.0); c = rng.uniform(5, 28)
        ds = rng.uniform(0.15, 0.60)
        visible = bool(rng.random() < (0.75 if g in (0, P.N_GIRDERS - 1) else 0.4))
        return dict(type="LS", girder=g, x0=c - ln / 2, x1=c + ln / 2, d_static=ds,
                    d_dyn=ds * rng.uniform(0.8, 1.0), severity=ds, visible=visible)
    raise ValueError(t)


def sample_scenario(rng, p_healthy=0.25, p_pair=0.15):
    u = rng.random()
    if u < p_healthy:
        return []
    if u < p_healthy + p_pair:
        a, b = rng.choice(len(MECHS), 2, replace=False)
        return [sample_mechanism(MECHS[a], rng), sample_mechanism(MECHS[b], rng)]
    return [sample_mechanism(MECHS[int(rng.integers(len(MECHS)))], rng)]


def build_state(mechs):
    st = TW.healthy_state()
    for m in mechs:
        t = m["type"]
        if t == "PL":
            st["eta"] = st["eta"] * m["eta"]
        elif t == "CD":
            st["conc"].append((m["girder"], m["x0"], m["x1"], m["dE"]))
        elif t == "CR":
            st["stiff"].append((m["girder"], m["x0"], m["x1"], m["d_static"], m["d_dyn"]))
        elif t == "CO":
            st["strand"].append((m["girder"], m["x0"], m["x1"], m["loss"]))
            st["stiff"].append((m["girder"], m["x0"], m["x1"], m["d_spall"], m["d_spall"]))
        elif t == "BR":
            st["kh_mult"] = st["kh_mult"] * m["kh_mult"]
        elif t == "LS":
            st["stiff"].append((m["girder"], m["x0"], m["x1"], m["d_static"], m["d_dyn"]))
    return st


# ---------------------------------------------------------------- one campaign
NOISE = dict(defl=0.05, truck=0.01, strain=3.0, lev_model=2.5, lev=0.8,
             vw_model=25.0, vw=3.0, brg=0.012)


def campaign(mechs, tw, ref, rng, age=None, T=None, hour=None, signals=True, mass=None):
    """Everything measured in one assessment campaign. ref = (f, w, S) of the
    healthy twin at 25 degC, used as spatial filter for identification."""
    age = rng.uniform(5, 30) if age is None else age
    if T is None:
        T, monsoon = site_temperature(rng, hour=hour)
    else:
        monsoon = 0.0
    hour = 2 if hour is None else hour
    st = build_state(mechs)
    st["mass"] = (1 + 0.006 * monsoon + rng.normal(0, 0.002)) if mass is None else mass
    feat = dict(T=T, age=age)
    # ---- vibration
    f, w = tw.modal(st, T)
    if signals:
        zeta = np.clip(rng.normal(0.015, 0.002, len(f)), 0.008, 0.03)
        rate = traffic_rate(rng, hour)
        acc = DY.simulate(f, zeta, w, tw.g.x, tw.g.y, rng, rate=rate)
        res, ex = DY.identify(acc, ref[2], ref[0], TRACK)
        for j, k in enumerate(TRACK):
            feat[f"f{j+1}"] = res[k]["f"]
        feat["z1"] = res[TRACK[0]]["zeta"]
        feat["mac1"] = res[TRACK[0]]["mac"]; feat["mac2"] = res[TRACK[1]]["mac"]
        feat["a_rms"] = ex["a_rms"] * 1e3 / 9.81       # mg
        feat["a_max"] = ex["a_max"] * 1e3 / 9.81
        sh = res[TRACK[0]]["shape"]
        mids = [i for i, (_, iy, ix) in enumerate(DY.SENSORS) if ix == 12]
        m = sh[mids] / np.mean(sh[mids])
        feat["phi_G1"] = float(m[0]); feat["phi_G6"] = float(m[-1])
    else:
        for j, k in enumerate(TRACK):
            feat[f"f{j+1}"] = f[k] * (1 + rng.normal(0, 0.0012))
    feat["f_true"] = [float(f[k]) for k in TRACK]
    # ---- static load test (four 3-axle lorries, two abreast x two in tandem)
    d, e = tw.load_test(st, T)
    tr = 1 + rng.normal(0, NOISE["truck"])
    d = d * tr + rng.normal(0, NOISE["defl"], 6)
    e = e * tr + rng.normal(0, NOISE["strain"], 6)
    for i in range(6):
        feat[f"d{i+1}"] = float(d[i]); feat[f"e{i+1}"] = float(e[i])
    # ---- long-term geometry and strain (changes beyond the healthy creep prediction)
    lev, vw = tw.long_term(st)
    common = rng.normal(0, NOISE["lev_model"])
    feat["lev_G1"] = float(lev[0] + common + rng.normal(0, NOISE["lev"]))
    feat["lev_C"] = float(0.5 * (lev[2] + lev[3]) + common + rng.normal(0, NOISE["lev"]))
    feat["lev_G6"] = float(lev[5] + common + rng.normal(0, NOISE["lev"]))
    vcom = rng.normal(0, NOISE["vw_model"])
    for i, gi in ((1, 0), (3, 2), (4, 3), (6, 5)):
        feat[f"vw_G{i}"] = float(vw[gi] + vcom + rng.normal(0, NOISE["vw"]))
    # ---- bearing movement (from a week of displacement vs temperature)
    s = tw.bearings(st, T)
    feat["brg_A"] = float(np.mean(s[[0, 5], 0]) + rng.normal(0, NOISE["brg"]))
    feat["brg_B"] = float(np.mean(s[[0, 5], 1]) + rng.normal(0, NOISE["brg"]))
    # ---- NDT and visual inspection (span S2 only for the library)
    cr_mask = tw.cracking(st)
    for m in mechs:
        if m["type"] == "PL":
            m["crack_w"] = 0.08 if cr_mask[m["girders"]].any() else 0.0
    locs, fld = C.healthy_fields(rng, age, spans=("S2",))
    fld = C.apply_mechanisms(locs, fld, mechs, rng)
    rd = C.measure_ndt(locs, fld, rng)
    obs = C.visual_inspection(locs, fld, rng, mechs, bearing_state=st["kh_mult"], spans=("S2",))
    feat.update(C.summarise(locs, rd, obs))
    return feat, dict(state=st, ndt=rd, locs=locs, obs=obs)


def labels(mechs):
    y = {m: 0 for m in MECHS}
    sev = {m: 0.0 for m in MECHS}
    loc = -1
    for m in mechs:
        y[m["type"]] = 1
        sev[m["type"]] = float(m["severity"])
        if "girder" in m:
            loc = m["girder"]
        elif m["type"] == "PL" and len(m["girders"]) == 1:
            loc = m["girders"][0]
    return y, sev, loc
