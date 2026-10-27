"""
Visual inspection and NDT: the inspection hierarchy, defect catalogue,
condition states and synthetic readings.

Bridge -> Span -> Component -> Location -> Observation
"""
import numpy as np
from . import inputs as P

SPANS = ("S1", "S2", "S3")
GIRDERS = tuple(f"G{i+1}" for i in range(P.N_GIRDERS))
# NDT / inspection zones along a girder (from bearing A), m
ZONES = {"Z1": (0.0, 3.3), "Z2": (3.3, 11.0), "Z3": (11.0, 22.0), "Z4": (22.0, 29.7), "Z5": (29.7, 33.0)}
DRAIN_X = (3.0, 8.5, 14.0, 19.5, 25.0, 30.5)      # drainage spouts along each kerb

COMPONENTS = {
    "DS": "Deck slab", "G": "PSC I-girder", "CG": "Cross girder / diaphragm",
    "FP": "Footpath", "CB": "Crash barrier", "BRG": "Bearing", "EJ": "Expansion joint",
    "DR": "Drainage spout", "PC": "Pier cap", "PR": "Pier", "AB": "Abutment", "WC": "Wearing coat",
}
WEIGHTS = {"G": 0.30, "DS": 0.12, "CG": 0.06, "BRG": 0.12, "EJ": 0.08, "DR": 0.04,
           "PC": 0.08, "PR": 0.08, "AB": 0.06, "FP": 0.02, "CB": 0.02, "WC": 0.02}
OBS = ("Cracking", "Spalling", "Corrosion staining", "Leakage", "Honeycombing",
       "Surface deterioration", "Efflorescence", "Bearing distress", "Joint distress",
       "Blocked drain", "Impact damage")
CS_TEXT = {1: "Good", 2: "Fair", 3: "Poor", 4: "Serious", 5: "Critical"}


def cs_crack(w_mm, member="PSC"):
    """Condition state from crack width. Prestressed members are judged harder:
    a flexural crack in a fully prestressed girder means precompression is gone."""
    if member == "PSC":
        edges = [0.05, 0.10, 0.20, 0.30]
    else:
        edges = [0.10, 0.20, 0.30, 1.00]
    return 1 + int(np.searchsorted(edges, w_mm, side="right"))


def location_table(spans=SPANS):
    rows = []
    for s in spans:
        for g in GIRDERS:
            for z, (x0, x1) in ZONES.items():
                rows.append(dict(id=f"{s}-{g}-{z}", span=s, girder=int(g[1:]) - 1, zone=z, x0=x0, x1=x1))
    return rows


# ---------------------------------------------------------------- material state of each location
def healthy_fields(rng, age, spans=SPANS):
    """As-built variability + normal ageing for every girder zone."""
    locs = location_table(spans)
    n = len(locs)
    f = dict(
        fc=rng.normal(55.0, 3.5, n),               # in-situ cube strength, MPa
        Eratio=np.ones(n),
        cover=np.clip(rng.normal(53.0, 4.5, n), 38, 70),   # to links, mm
        kc=rng.normal(3.0, 0.4, n),                # carbonation coefficient, mm/sqrt(yr)
        wet=np.zeros(n),                           # 0 dry ... 1 persistently wet
        corr=np.zeros(n),                          # corrosion activity 0 ... 1
        cracked=np.zeros(n),                       # visible crack width, mm
        spall=np.zeros(n),                         # spalled area, m2
        void=np.zeros(n),                          # honeycomb / delamination (UPV path damage)
        exposed=np.zeros(n, bool),
    )
    # outer girders get wet near drainage spouts and at the joint ends
    for i, l in enumerate(locs):
        if l["girder"] in (0, P.N_GIRDERS - 1):
            f["wet"][i] = rng.uniform(0.0, 0.25)
        if l["zone"] in ("Z1", "Z5"):
            f["wet"][i] += rng.uniform(0.0, 0.15)
    f["carb"] = f["kc"] * np.sqrt(age) * (1 - 0.3 * np.clip(f["wet"], 0, 1))
    return locs, f


def _mask(locs, span, girder, x0, x1):
    return np.array([(l["span"] == span) and (l["girder"] == girder) and (l["x1"] > x0) and (l["x0"] < x1)
                     for l in locs])


def apply_mechanisms(locs, f, mechs, rng, span="S2"):
    """Effect of each deterioration mechanism on the material state."""
    for m in mechs:
        t = m["type"]
        if t == "CD":
            k = _mask(locs, span, m["girder"], m["x0"], m["x1"])
            f["Eratio"][k] *= (1 - m["dE"])
            f["fc"][k] *= (1 - 0.9 * m["dE"])
            f["void"][k] = np.maximum(f["void"][k], m["dE"] * rng.uniform(0.3, 0.8, k.sum()))
            f["kc"][k] *= 1 + 2 * m["dE"]
        elif t == "CO":
            k = _mask(locs, span, m["girder"], m["x0"], m["x1"])
            c = m["loss"]
            f["wet"][k] = np.clip(f["wet"][k] + 0.6, 0, 1)
            f["cover"][k] = np.minimum(f["cover"][k], rng.uniform(28, 40, k.sum()))
            f["corr"][k] = np.clip(0.3 + 4 * c, 0, 1)
            # bursting cracks along the bars only once enough rust has formed
            w_c = np.where(c > 0.03, np.clip(6 * (c - 0.03) + 0.05, 0.05, 1.5), 0.0)
            f["cracked"][k] = np.maximum(f["cracked"][k], w_c * rng.uniform(0.7, 1.3, k.sum()))
            f["spall"][k] = np.where(c > 0.06, rng.uniform(0.05, 0.6, k.sum()) * c / 0.1, 0.0)
            f["exposed"][k] |= c > 0.10
            f["void"][k] = np.maximum(f["void"][k], 0.3 * f["corr"][k])
        elif t == "CR":
            k = _mask(locs, span, m["girder"], m["x0"], m["x1"])
            f["cracked"][k] = np.maximum(f["cracked"][k], m["w"] * rng.uniform(0.8, 1.2, k.sum()))
            f["void"][k] = np.maximum(f["void"][k], 0.15 * m["d_static"])
        elif t == "LS":
            k = _mask(locs, span, m["girder"], m["x0"], m["x1"])
            f["void"][k] = np.maximum(f["void"][k], 0.8 * m["d_static"])
            if m.get("visible", True):        # impact / spalling seen from below
                f["spall"][k] = np.maximum(f["spall"][k], 0.3 + 1.5 * m["d_static"])
                f["exposed"][k] |= m["d_static"] > 0.3
                f["cracked"][k] = np.maximum(f["cracked"][k], 0.2 + m["d_static"])
            else:                             # internal void / delamination: only hollow sound, UPV
                f["cracked"][k] = np.maximum(f["cracked"][k], 0.03)
        elif t == "PL":
            # visible only once the girder has cracked (see twin.cracking)
            if m.get("crack_w", 0) > 0:
                for g in m["girders"]:
                    k = _mask(locs, span, g, 8.0, 25.0)
                    f["cracked"][k] = np.maximum(f["cracked"][k], m["crack_w"])
    # carbonation front reaching the links in wet concrete starts corrosion
    depass = (f["carb"] >= f["cover"]) & (f["wet"] > 0.3)
    f["corr"] = np.maximum(f["corr"], np.where(depass, 0.4, 0.0))
    return f


# ---------------------------------------------------------------- readings
def _rebound(fc, carb, rng):
    R = (np.maximum(fc, 5) / 0.0377) ** (1 / 1.95)
    R *= 1 + 0.004 * np.clip(carb, 0, 30)            # carbonated skin reads hard
    return R + rng.normal(0, 1.6, len(fc))


def measure_ndt(locs, f, rng):
    n = len(locs)
    wet = np.clip(f["wet"], 0, 1)
    upv = 4.55 * np.sqrt(f["Eratio"]) * (1 - 0.35 * f["void"]) * (1 + 0.02 * wet)
    upv = upv + rng.normal(0, 0.07, n)
    reb = _rebound(f["fc"] * (1 - 0.3 * f["void"]), f["carb"], rng) - 3 * wet
    cover = f["cover"] + rng.normal(0, 2.0, n)
    hcp = (-120 - 160 * wet - 330 * f["corr"]) + rng.normal(0, 25, n)        # mV vs CSE
    rho = 55 * np.exp(-1.6 * wet) * (1 - 0.5 * f["corr"]) * np.exp(rng.normal(0, 0.15, n))   # kOhm cm
    carb = np.clip(f["carb"] + rng.normal(0, 1.5, n), 0, None)
    chl = np.clip(0.06 + 0.04 * wet + 0.10 * f["corr"] + rng.normal(0, 0.02, n), 0.01, None)   # % cement
    return dict(rebound=reb, upv=upv, cover=cover, hcp=hcp, resistivity=rho,
                carbonation=carb, chloride=chl)


def interpret_ndt(r):
    """Engineering reading of each NDT value (IS 516 Part 5 / ASTM C876 style bands)."""
    out = []
    for i in range(len(r["upv"])):
        u, h, rho = r["upv"][i], r["hcp"][i], r["resistivity"][i]
        q = "Good" if u > 4.40 else ("Medium" if u >= 3.75 else "Doubtful")
        c = "low risk" if h > -200 else ("uncertain" if h > -350 else "corrosion likely")
        rr = "low" if rho > 20 else ("moderate" if rho > 10 else ("high" if rho > 5 else "very high"))
        out.append(dict(upv_quality=q, hcp_risk=c, resistivity_rate=rr,
                        carbonated_to_steel=bool(r["carbonation"][i] >= r["cover"][i])))
    return out


def visual_inspection(locs, f, rng, mechs=(), bearing_state=None, span="S2", spans=SPANS):
    """Observations for the girders plus the other components of the bridge."""
    obs = []
    def add(span_, comp, loc, kind, cs, size="", note=""):
        cs = int(np.clip(cs + (rng.random() < 0.15) * rng.choice([-1, 1]), 1, 5))   # inspector scatter
        obs.append(dict(span=span_, component=comp, location=loc, observation=kind,
                        size=size, cs=cs, note=note))
    for i, l in enumerate(locs):
        w = f["cracked"][i]
        if w > 0 and rng.random() < 1 - np.exp(-w / 0.06):            # hairlines get missed
            wm = round(max(0.05, w + rng.normal(0, 0.03)), 2)
            if f["corr"][i] > 0.35:      # splitting crack along a bar / duct, judged as for RC
                add(l["span"], f"G{l['girder']+1}", l["zone"], "Cracking", cs_crack(wm, "RC"), f"{wm} mm",
                    "longitudinal, along bars / duct, rust in crack")
            else:                        # flexural crack in a prestressed girder
                add(l["span"], f"G{l['girder']+1}", l["zone"], "Cracking", cs_crack(wm), f"{wm} mm", "flexural, soffit")
        if f["spall"][i] > 0.02:
            cs = 5 if f["exposed"][i] and f["spall"][i] > 0.5 else (4 if f["exposed"][i] else 3)
            add(l["span"], f"G{l['girder']+1}", l["zone"], "Spalling", cs, f"{f['spall'][i]:.2f} m2",
                "strands/links exposed" if f["exposed"][i] else "")
        if f["corr"][i] > 0.35 and rng.random() < np.clip(2.5 * (f["corr"][i] - 0.3), 0, 1):
            add(l["span"], f"G{l['girder']+1}", l["zone"], "Corrosion staining", 3 + int(f["corr"][i] > 0.7))
        if f["wet"][i] > 0.5:
            add(l["span"], f"G{l['girder']+1}", l["zone"], "Leakage", 2 + int(f["wet"][i] > 0.8), "damp patches / efflorescence")
        if f["void"][i] > 0.2 and f["spall"][i] <= 0.02:
            add(l["span"], f"G{l['girder']+1}", l["zone"], "Honeycombing", 2 + int(f["void"][i] > 0.4))
        if f["Eratio"][i] < 0.88 and rng.random() < 0.8:
            add(l["span"], f"G{l['girder']+1}", l["zone"], "Surface deterioration", 3, "map cracking / scaling")
    # bearings, joints and drains of the studied span
    leak_ends = set()
    for m in mechs:
        if m["type"] == "BR" and m.get("joint_leak", True):
            leak_ends |= {0, 1} if m.get("ends") == "both" else ({0} if m.get("ends") == "A" else {1})
    if rng.random() < 0.08:                    # leaking joints are common even with sound bearings
        leak_ends.add(int(rng.integers(2)))
    if bearing_state is not None:
        for end, joint in ((0, "EJ-P1"), (1, "EJ-P2")):
            mult = np.asarray(bearing_state)[:, end]
            for gi, m in enumerate(mult):
                # a mildly hardened bearing looks normal; a seized one shows no shear deformation, cracks, debris
                p_see = np.clip((np.log10(max(m, 1.0)) - 0.8) / 1.4, 0.0, 1.0)
                if m > 3 and rng.random() < p_see:
                    cs = 3 if m < 30 else (4 if m < 300 else 5)
                    add(span, "BRG", f"G{gi+1}-{'A' if end == 0 else 'B'}", "Bearing distress", cs,
                        "", "hardened / restrained, debris around bearing")
            if end in leak_ends:
                add(span, "EJ", joint, "Joint distress", 3 + int(mult.max() > 100), "", "seal torn, debris packed in gap")
                add(span, "EJ", joint, "Leakage", 3, "", "water and silt onto pier cap")
    for m in mechs:
        if m["type"] == "CO" and m["girder"] in (0, P.N_GIRDERS - 1):
            add(span, "DR", f"{'U' if m['girder'] == 0 else 'D'}-{int(np.argmin(np.abs(np.array(DRAIN_X) - 0.5 * (m['x0'] + m['x1'])))) + 1}",
                "Blocked drain", 3, "", "spout too short, water runs on girder")
        if m["type"] == "LS" and m["girder"] in (0, P.N_GIRDERS - 1):
            add(span, f"G{m['girder']+1}", "soffit", "Impact damage", 4, "", "scraped soffit over service road")
    return obs


def condition_rating(obs, spans=SPANS):
    """Component rating = worst CS; bridge index 0-100 weighted over component groups."""
    worst = {}
    for o in obs:
        grp = "G" if o["component"].startswith("G") else o["component"]
        key = (o["span"], grp)
        worst[key] = max(worst.get(key, 1), o["cs"])
    groups = {}
    for (s, grp), cs in worst.items():
        groups[grp] = max(groups.get(grp, 1), cs)
    total = sum(WEIGHTS.values())
    bci = 100 * sum(w * (5 - groups.get(k, 1)) / 4 for k, w in WEIGHTS.items()) / total
    return worst, groups, bci


def summarise(locs, r, obs, span="S2"):
    """Span-level NDT / VI features for the learning data set."""
    k = np.array([l["span"] == span for l in locs])
    g_obs = [o for o in obs if o["span"] == span and o["component"].startswith("G")]
    crack = [float(o["size"].split()[0]) for o in g_obs if o["observation"] == "Cracking"]
    flex = [float(o["size"].split()[0]) for o in g_obs if o["observation"] == "Cracking" and o["note"].startswith("flexural")]
    long_ = [float(o["size"].split()[0]) for o in g_obs if o["observation"] == "Cracking" and o["note"].startswith("longitudinal")]
    spall = [float(o["size"].split()[0]) for o in g_obs if o["observation"] == "Spalling"]
    return dict(
        R_min=float(np.min(r["rebound"][k])), R_med=float(np.median(r["rebound"][k])),
        UPV_min=float(np.min(r["upv"][k])), UPV_med=float(np.median(r["upv"][k])),
        HCP_min=float(np.min(r["hcp"][k])), rho_min=float(np.min(r["resistivity"][k])),
        cover_min=float(np.min(r["cover"][k])),
        carb_margin=float(np.max(r["carbonation"][k] - r["cover"][k])),
        crack_w=max(crack) if crack else 0.0, crack_flex=max(flex) if flex else 0.0,
        crack_long=max(long_) if long_ else 0.0, spall_area=sum(spall),
        n_rust=sum(o["observation"] == "Corrosion staining" for o in g_obs),
        cs_girder=max([o["cs"] for o in g_obs], default=1),
        cs_bearing=max([o["cs"] for o in obs if o["span"] == span and o["component"] == "BRG"], default=1),
        joint_leak=int(any(o["component"] == "EJ" and o["observation"] == "Leakage" for o in obs if o["span"] == span)),
    )
