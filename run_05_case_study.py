"""
Step 5 - the story of this bridge: 18 years of monitoring and the year-18
detailed assessment, diagnosed with the models trained on the twin library.

The 'real bridge' here is the synthetic truth model (parameters that the twin had
to identify). Its deterioration history below is hidden from the assessment; it
is only used to generate the measurements and, at the end, to check the diagnosis.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker process
from psc.perf import boost
boost()
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv, json, pickle, time
import numpy as np
from multiprocessing import Pool
from psc import inputs as P, twin as TW, dynamics as DY, scenarios as SC, condition as C
from psc import features as FE, mlkit as ML, maintenance as MT, design as D, section as S

YEAR_NOW = 18.0
ONSET = dict(joint=9.0, bearing=10.0, corrosion=11.0)


def history(t):
    """Hidden deterioration state of span S2 at age t (years)."""
    mechs = []
    if t > ONSET["bearing"]:
        mult = np.ones((P.N_GIRDERS, 2))
        mult[:, 1] = 10 ** (0.25 * (t - ONSET["bearing"]))                 # end B (pier P2, leaking joint)
        mult[:, 0] = 1 + 0.30 * (t - ONSET["bearing"])                    # end A, slower ageing
        mechs.append(dict(type="BR", ends="both", kh_mult=mult, severity=float(np.log10(mult[:, 1].mean()))))
    if t > ONSET["corrosion"]:
        c = 0.012 * (t - ONSET["corrosion"])
        mechs.append(dict(type="CO", girder=0, x0=28.5, x1=33.0, loss=c, d_spall=0.25 * c, severity=c))
        loss = 0.009 * (t - ONSET["corrosion"])
        eta = np.ones(P.N_GIRDERS); eta[0] = 1 - loss
        mechs.append(dict(type="PL", girders=[0], loss=loss, eta=eta, severity=loss))
    return mechs


_TW = None; _REF = None
def _init(truth):
    boost()
    global _TW, _REF
    _TW = TW.Twin(**truth)
    r = np.load("out/twin_reference.npz"); _REF = (r["f"], r["w"], r["S"])


def _month(args):
    m, seed = args
    rng = np.random.default_rng(seed)
    t = m / 12.0
    day = (m % 12) * 30.4 + 15
    T, monsoon = SC.site_temperature(rng, day=day, hour=2)
    st = SC.build_state(history(t)); st["mass"] = 1 + 0.006 * monsoon + rng.normal(0, 0.002)
    f, w = _TW.modal(st, T)
    acc = DY.simulate(f, np.clip(rng.normal(0.015, 0.002, len(f)), 0.008, 0.03), w, _TW.g.x, _TW.g.y, rng,
                      rate=SC.traffic_rate(rng, 2))
    res, ex = DY.identify(acc, _REF[2], _REF[0], SC.TRACK)
    s = _TW.bearings(st, T)
    lev, vw = _TW.long_term(st)
    return dict(month=m, t=t, T=T, **{f"f{j+1}": res[k]["f"] for j, k in enumerate(SC.TRACK)},
                a_rms=ex["a_rms"] * 1e3 / 9.81, mac1=res[0]["mac"],
                brg_A=float(s[0, 0] + rng.normal(0, 0.012)), brg_B=float(s[0, 1] + rng.normal(0, 0.012)),
                vw_G1=float(vw[0] + rng.normal(0, 3) + rng.normal(0, 8)),
                vw_G6=float(vw[5] + rng.normal(0, 3) + rng.normal(0, 8)),
                lev_G1=float(lev[0] + rng.normal(0, 1.0)), lev_G6=float(lev[5] + rng.normal(0, 1.0)))


def ewma_alarm(t, x, base_mask, lam=0.2, L=3.0, side=0):
    mu, sd = x[base_mask].mean(), x[base_mask].std()
    z = ML.ewma((x - mu) / sd, lam)
    lim = L * np.sqrt(lam / (2 - lam))
    hit = (z > lim) if side > 0 else (z < -lim) if side < 0 else (np.abs(z) > lim)
    hit &= ~base_mask
    first = float(t[np.argmax(hit)]) if hit.any() else None
    return z, lim, first


def site_survey(rng, age, mechs_S2):
    """NDT + VI of all three spans at the year-18 detailed inspection."""
    locs, fld = C.healthy_fields(rng, age)
    fld = C.apply_mechanisms(locs, fld, mechs_S2, rng, span="S2")
    # the same leaking joint wets the S3 girder ends at pier P2 (S3 end A)
    for i, l in enumerate(locs):
        if l["span"] == "S3" and l["zone"] == "Z1" and l["girder"] in (0, 1):
            fld["wet"][i] = min(1.0, fld["wet"][i] + 0.55)
            fld["cover"][i] = min(fld["cover"][i], 41.0)
        # the older joint at pier P1 seeps a little onto the outer girder ends either side
        if ((l["span"] == "S1" and l["zone"] == "Z5") or (l["span"] == "S2" and l["zone"] == "Z1")) and l["girder"] in (0, 5):
            fld["wet"][i] = min(1.0, fld["wet"][i] + 0.5)
        if l["span"] == "S1" and l["zone"] == "Z3" and l["girder"] == 5:
            fld["void"][i] = 0.25                                        # old honeycomb patch, as-built
    depass = (fld["carb"] >= fld["cover"]) & (fld["wet"] > 0.3)
    fld["corr"] = np.maximum(fld["corr"], np.where(depass, 0.45, 0.0))
    rd = C.measure_ndt(locs, fld, rng)
    st = SC.build_state(mechs_S2)
    obs = C.visual_inspection(locs, fld, rng, mechs_S2, bearing_state=st["kh_mult"])
    # other components (not girders) seen at the inspection
    extra = [
        ("S2", "PC", "P2", "Leakage", 3, "", "silt and water stains below S2 end B bearings"),
        ("S2", "PC", "P2", "Spalling", 3, "0.15 m2", "pedestal edge spalled under G1-B"),
        ("S2", "DR", "U-6", "Blocked drain", 3, "", "spout 150 mm long, discharges on G1"),
        ("S2", "WC", "near EJ-P2", "Surface deterioration", 3, "", "ravelling and potholes 2 m either side of the joint"),
        ("S3", "BRG", "G1-A", "Bearing distress", 3, "", "bulging, silt around bearing"),
        ("S3", "BRG", "G2-A", "Bearing distress", 3, "", "silt around bearing"),
        ("S1", "DS", "bay G2-G3", "Efflorescence", 2, "", "white deposits along a construction joint"),
        ("S1", "CB", "upstream", "Cracking", 2, "0.15 mm", "vertical shrinkage cracks at 3-4 m"),
        ("S1", "AB", "A1", "Surface deterioration", 2, "", "vegetation in bearing shelf drain hole"),
        ("S3", "FP", "downstream", "Surface deterioration", 2, "", "loose kerb cover slab"),
        ("S2", "PR", "P2", "Leakage", 2, "", "stains running down pier face"),
        ("S1", "PC", "P1", "Cracking", 2, "0.10 mm", "hairline vertical crack, not active"),
    ]
    for s, comp, loc, kind, cs, size, note in extra:
        obs.append(dict(span=s, component=comp, location=loc, observation=kind, size=size, cs=cs, note=note))
    return locs, fld, rd, obs


if __name__ == "__main__":
    t0 = time.time()
    base = json.load(open("out/baseline.json"))
    truth = base["twin"]["truth"]
    rng = np.random.default_rng(1818)
    out = {}

    # ------------------------------------------------------------ 18 years of monthly monitoring
    months = list(range(0, int(YEAR_NOW * 12) + 1))
    with Pool(14, initializer=_init, initargs=(truth,)) as pool:
        mon = pool.map(_month, [(m, 50_000 + m) for m in months], chunksize=6)
    with open("out/data/case_monitoring.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mon[0].keys())); w.writeheader(); w.writerows(mon)

    # own-baseline normalisation of the real bridge (first year of data)
    base_rows = FE.read_csv("out/data/baseline_year.csv")
    norm = FE.Normaliser().fit(base_rows)
    tp = base["twin"]
    tc = FE.Normaliser._tc(dict(T=tp["T"]))
    norm.set_static_reference(np.array(tp["meas_d"]) * tc, np.array(tp["meas_e"]) * tc)
    tm = np.array([r["t"] for r in mon])
    X1 = np.column_stack([np.ones(len(mon)), np.array([r["T"] for r in mon]) - FE.REF_T, np.log([r["a_rms"] for r in mon])])
    rf1 = 100 * (np.array([r["f1"] for r in mon]) / (X1 @ norm.fb["f1"]) - 1)
    rf2 = 100 * (np.array([r["f2"] for r in mon]) / (X1 @ norm.fb["f2"]) - 1)
    bmask = tm < 2.0
    trends = {}
    for name, x, side in (("f1 residual", rf1, 0), ("f2 residual", rf2, 0),
                          ("bearing B movement", np.array([r["brg_B"] for r in mon]), -1),
                          ("bearing A movement", np.array([r["brg_A"] for r in mon]), +1),
                          ("VW strain G1", np.array([r["vw_G1"] for r in mon]), +1),
                          ("level G1", np.array([r["lev_G1"] for r in mon]), -1)):
        z, lim, first = ewma_alarm(tm, x, bmask, side=side)
        trends[name] = dict(first_alarm_year=first, limit=lim)
        # Theil-Sen slope over the last 3 years
        k = tm > YEAR_NOW - 3
        tt, xx = tm[k], x[k]
        sl = np.median([(xx[j] - xx[i]) / (tt[j] - tt[i]) for i in range(len(tt)) for j in range(i + 1, len(tt)) if tt[j] > tt[i]])
        trends[name]["slope_last3y_per_year"] = float(sl)
    print("trend alarms:", {k: v["first_alarm_year"] for k, v in trends.items()})
    out["trends"] = trends
    out["onset"] = ONSET

    # ------------------------------------------------------------ year-18 detailed campaign
    tw_true = TW.Twin(**truth)
    r = np.load("out/twin_reference.npz"); ref = (r["f"], r["w"], r["S"])
    mechs_now = history(YEAR_NOW)
    feat, aux = SC.campaign(mechs_now, tw_true, ref, rng, age=YEAR_NOW, T=27.5, hour=2)
    feat.pop("f_true")
    locs, fld, rd, obs = site_survey(rng, YEAR_NOW, mechs_now)
    summ = C.summarise(locs, rd, obs)
    feat.update(summ)
    x = norm.transform(feat)
    out["campaign_raw"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in feat.items()}
    out["campaign_features"] = {k: float(v) for k, v in x.items() if not k.startswith("_")}
    out["campaign_girders"] = dict(d=x["_d"].tolist(), e=x["_e"].tolist())

    # ------------------------------------------------------------ AI/ML assessment
    M = pickle.load(open("out/models.pkl", "rb"))
    sc, md, names = M["novelty"]
    xv = np.array([[x[n] for n in names]])
    score = float(md.score(sc.transform(xv))[0])
    p_fused = {m: float(M["fused"][m].predict_proba(np.array([[x[n] for n in FE.SOURCES["All sources fused"]]]))[0, 1]) for m in SC.MECHS}
    p_vib = {m: float(M["vib"][m].predict_proba(np.array([[x[n] for n in FE.SOURCES["Vibration only"]]]))[0, 1]) for m in SC.MECHS}
    sc_pl, gp, pl_names = M["gpr"]
    mu, sd = gp.predict(sc_pl.transform(np.array([[x[n] for n in pl_names]])))
    lv = np.concatenate([x["_d"], x["_e"], [feat["lev_G1"], feat["lev_G6"], feat["vw_G1"], feat["vw_G3"], feat["vw_G4"],
                                            feat["vw_G6"], x["phi_asym"]]])
    p_loc = M["loc"].predict_proba(lv[None])[0]
    # engineer's cross-check of the G1 prestress loss from physics (twin sensitivities)
    est_lev = -feat["lev_G1"] / 1.04
    est_vw = feat["vw_G1"] / 12.4
    w1, w2 = 1 / 2.6 ** 2, 1 / 2.1 ** 2
    est_g1 = (w1 * est_lev + w2 * est_vw) / (w1 + w2); sd_g1 = (w1 + w2) ** -0.5
    out["ml"] = dict(novelty_score=score, novelty_threshold=md.threshold, novelty=bool(score > md.threshold),
                     p_fused=p_fused, p_vibration=p_vib, pl_gpr_mean=float(mu[0]), pl_gpr_sd=float(sd[0]),
                     pl_G1_levelling=est_lev, pl_G1_vw=est_vw, pl_G1_combined=est_g1, pl_G1_sd=sd_g1,
                     loc_proba=p_loc.tolist())
    print("novelty", round(score, 1), ">", round(md.threshold, 1), "| fused p", {k: round(v, 2) for k, v in p_fused.items()})
    print("vibration-only p", {k: round(v, 2) for k, v in p_vib.items()})
    print(f"PL GPR {mu[0]:.1f} +- {sd[0]:.1f} %; G1 physics {est_g1:.1f} +- {sd_g1:.1f} % (lev {est_lev:.1f}, vw {est_vw:.1f})")
    print("girder localisation p:", np.round(p_loc, 2))

    # ------------------------------------------------------------ NDT clustering (all spans)
    Xn = np.column_stack([rd["upv"], rd["rebound"], rd["hcp"], np.log(rd["resistivity"]), rd["cover"] - rd["carbonation"]])
    scn = ML.Scaler().fit(Xn); Zn = scn.transform(Xn)
    bics = {}
    gms = {}
    for k in (1, 2, 3, 4, 5):
        gm = ML.GaussianMixture(k, seed=2).fit(Zn)
        bics[k] = float(gm.bic); gms[k] = gm
    k_best = min(bics, key=bics.get)
    k_use = max(k_best, 2)
    gm = gms[k_use]
    lab = gm.predict(Zn)
    lab_km, _, _ = ML.kmeans(Zn, k_use, seed=1)
    clusters = []
    for j in range(k_use):
        k = lab == j
        clusters.append(dict(n=int(k.sum()), upv=float(rd["upv"][k].mean()), rebound=float(rd["rebound"][k].mean()),
                             hcp=float(rd["hcp"][k].mean()), rho=float(rd["resistivity"][k].mean()),
                             margin=float((rd["cover"] - rd["carbonation"])[k].mean()),
                             members=[locs[i]["id"] for i in np.where(k)[0]]))
    order = np.argsort([c["hcp"] for c in clusters])[::-1]            # least negative first
    names_c = {2: ["sound / dry", "wet / corrosion risk"], 3: ["sound / dry", "wet, passive", "corrosion likely"],
               4: ["sound / dry", "slightly damp", "wet, passive", "corrosion likely"],
               5: ["sound / dry", "slightly damp", "damp", "wet, passive", "corrosion likely"]}[k_use]
    for rank, j in enumerate(order):
        clusters[j]["name"] = names_c[rank]
    print("GMM BIC", {k: round(v) for k, v in bics.items()}, "-> best k", k_best)
    for c in clusters:
        print(f"  {c['name']:18s} n={c['n']:3d} UPV {c['upv']:.2f} R {c['rebound']:.1f} HCP {c['hcp']:.0f} rho {c['rho']:.0f} margin {c['margin']:.0f}",
              c["members"][:8] if c["name"] != "sound / dry" else "")
    agree = float(np.mean([(lab_km[i] == lab_km[j]) == (lab[i] == lab[j]) for i in range(len(lab)) for j in range(i)]))
    out["ndt_clusters"] = dict(bic=bics, k_best=k_best, k_used=k_use, clusters=clusters, kmeans_agreement=agree)

    # ------------------------------------------------------------ structural check with the estimated loss
    d = json.load(open("out/design.json"))
    loss = max(est_g1, 0) / 100
    sP = tw_true.sigP_bot
    rare_inf = d["stages_summary"]["rare_min_bot"]                 # design basis, 0.9 Pm
    rare_m = rare_inf + 0.1 * sP                                  # same with mean prestress (Pm instead of 0.9 Pm)
    margin_design = rare_inf - 0.9 * loss * sP + P.fctm(P.FCK_GIRDER)
    margin_mean = rare_m - loss * sP + P.fctm(P.FCK_GIRDER)
    qp_mean = d["stages"]["G1"]["16.500"]["rows"]["V quasi, inf"]["bot"] + 0.1 * sP - loss * sP
    st_now = SC.build_state(mechs_now)
    uls_ratio = d["uls"]["G1"]["MEd"] / d["uls"]["G1"]["M"]
    # shear at the B end of G1 with reduced Vp (local strand loss + prestress loss)
    sh = d["shear"]["G1@2.75"]
    c_loc = [m for m in mechs_now if m["type"] == "CO"][0]["loss"]
    Vp_red = sh["Vp"] * (1 - loss) * (1 - c_loc)
    Vnet = sh["VEd"] - Vp_red
    out["assessment"] = dict(loss_used_pct=100 * loss, sls_margin_design=margin_design, sls_margin_mean=margin_mean,
                             qp_soffit_mean=qp_mean, uls_ratio=uls_ratio, shear_Vnet=Vnet, shear_VRdmax=sh["VRdmax"],
                             shear_Vnet_design=sh["Vnet"], Asw_req=(Vnet / sh["Vnet"]) * sh["Asw_s"],
                             cracked_now=bool(tw_true.cracking(st_now).any()))
    print("assessment:", {k: round(v, 2) if isinstance(v, float) else v for k, v in out["assessment"].items()})

    # ------------------------------------------------------------ decision
    worst, groups, bci = C.condition_rating(obs)
    near_anch = any(o["span"] == "S2" and o["component"] == "G1" and o["location"] == "Z5"
                    and o["observation"] in ("Corrosion staining", "Spalling") for o in obs)
    ev = dict(vi_groups=groups, crack_w=summ["crack_w"], hcp_min=summ["HCP_min"], rho_min=summ["rho_min"],
              carb_margin=summ["carb_margin"], bearing_ratio=x["brg_min"], novelty=out["ml"]["novelty"],
              p=p_fused, pl_est=est_g1, pl_sd=sd_g1, sls_margin=margin_design, uls_ratio=uls_ratio,
              trend_alarms=[k for k, v in trends.items() if v["first_alarm_year"] is not None],
              near_anchorage=near_anch)
    cat, acts, intervals = MT.decide(ev)
    out["decision"] = dict(category=cat, actions=[(MT.ACTIONS[k][0], MT.ACTIONS[k][1], reason, MT.ACTIONS[k][2]) for k, reason in acts],
                           intervals=intervals, bci=bci, groups=groups)
    print("BCI", round(bci, 1), "groups", groups)
    print("category:", cat)
    for a in out["decision"]["actions"]:
        print("  -", a[0], "|", a[1], "|", a[2])

    # ------------------------------------------------------------ files for the report and the app
    out["truth"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()} for m in mechs_now]
    with open("out/data/case_ndt_survey.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["location", "span", "girder", "zone", "cluster"] + list(rd.keys()))
        for i, l in enumerate(locs):
            w.writerow([l["id"], l["span"], f"G{l['girder']+1}", l["zone"], clusters[lab[i]]["name"]]
                       + [round(float(rd[k][i]), 3) for k in rd])
    with open("out/data/case_visual_inspection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(obs[0].keys())); w.writeheader(); w.writerows(obs)
    json.dump(obs, open("inspection_app/seed_observations.json", "w"), indent=1)
    np.savez("out/case_arrays.npz", tm=tm, rf1=rf1, rf2=rf2, Zn=Zn, lab=lab, lab_km=lab_km)
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, np.ndarray): return clean(o.tolist())
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.bool_): return bool(o)
        return o
    json.dump(clean(out), open("out/case_study.json", "w"), indent=1)
    print(f"[done in {time.time()-t0:.0f} s]")
