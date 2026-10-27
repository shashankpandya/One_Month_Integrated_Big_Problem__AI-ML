"""
Step 4 - AI/ML on the scenario library.
  A novelty detection (healthy-only training) per data source
  B supervised detection of each mechanism, per data source (data fusion study)
  C severity: prestress loss by Gaussian-process regression (with uncertainty)
  D localisation of the affected girder
  E permutation importance and an effect-size matrix (physics check)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker process
from psc.perf import boost
boost()
import json, pickle, time
import numpy as np
from multiprocessing import Pool
from psc import mlkit as ML, features as FE, scenarios as SC

MECHS = SC.MECHS
RF_KW = dict(n_trees=100, max_depth=12, min_leaf=3, max_features="sqrt")


def _fit_rf(args):
    boost()
    key, Xtr, ytr, Xte, seed = args
    rf = ML.RandomForest(seed=seed, **RF_KW).fit(Xtr, ytr)
    return key, rf.predict_proba(Xte), rf


def class_key(r):
    if r["n_mech"] == 0:
        return "H"
    if r["n_mech"] == 1:
        return [m for m in MECHS if r[f"y_{m}"] == 1][0]
    return "pair"


if __name__ == "__main__":
    t0 = time.time()
    rows = FE.read_csv("out/data/library.csv")
    keys = np.array([class_key(r) for r in rows])
    folds = ML.stratified_folds(keys, k=10, seed=7)
    test = folds < 3                                         # 30 % hold-out
    train = ~test
    healthy_tr = [r for r, t, k in zip(rows, train, keys) if t and k == "H"]
    norm = FE.Normaliser().fit(healthy_tr)
    F = [norm.transform(r) for r in rows]
    Y = {m: np.array([int(r[f"y_{m}"]) for r in rows]) for m in MECHS}
    SEV = {m: np.array([r[f"sev_{m}"] for r in rows]) for m in MECHS}
    H = keys == "H"
    out = dict(n=len(rows), n_train=int(train.sum()), n_test=int(test.sum()),
               counts={k: int((keys == k).sum()) for k in np.unique(keys)})
    print(out)

    # ------------------------------------------------------------ A. novelty detection
    nov = {}
    curves = {}
    for sname, names in FE.SOURCES.items():
        X = FE.matrix(F, names)
        sc = ML.Scaler().fit(X[train & H])
        Z = sc.transform(X)
        md = ML.Mahalanobis().fit(Z[train & H], q=0.99)
        s = md.score(Z)
        alarm = s > md.threshold
        rate = {"false alarm": float(alarm[test & H].mean())}
        for m in MECHS:
            sel = test & (keys == m)
            rate[m] = float(alarm[sel].mean())
        nov[sname] = rate
        # detection vs severity (single-mechanism test cases)
        cv = {}
        for m in MECHS:
            sel = test & (keys == m)
            sv = SEV[m][sel]; al = alarm[sel]
            edges = np.quantile(sv, np.linspace(0, 1, 6))
            mids, pod = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                k = (sv >= a) & (sv <= b)
                mids.append(float(sv[k].mean())); pod.append(float(al[k].mean()))
            cv[m] = (mids, pod)
        curves[sname] = cv
        if sname == "All sources fused":
            nov_model = (sc, md, names)
        print(f"novelty {sname:38s}", {k: round(v, 2) for k, v in rate.items()})
    out["novelty"] = nov; out["novelty_curves"] = curves

    # ------------------------------------------------------------ B. supervised, per source
    jobs = []
    Xs = {s: FE.matrix(F, n) for s, n in FE.SOURCES.items()}
    for sname in FE.SOURCES:
        for j, m in enumerate(MECHS):
            jobs.append(((sname, m), Xs[sname][train], Y[m][train], Xs[sname][test], 11 + j))
    with Pool(14) as pool:
        res = pool.map(_fit_rf, jobs)
    proba = {k: p[:, 1] for k, p, _ in res}
    models = {k: rf for k, _, rf in res}
    sup = {}
    for sname in FE.SOURCES:
        sup[sname] = {}
        for m in MECHS:
            p = proba[(sname, m)]; y = Y[m][test]
            _, _, auc = ML.roc(y, p)
            pr, rc, f1 = ML.prf(y, (p > 0.5).astype(int))
            sup[sname][m] = dict(AUC=auc, F1=f1, precision=pr, recall=rc)
        mf1 = np.mean([sup[sname][m]["F1"] for m in MECHS])
        sup[sname]["macro_F1"] = float(mf1)
        print(f"RF {sname:38s} macro-F1 {mf1:.3f}  AUC", {m: round(sup[sname][m]['AUC'], 3) for m in MECHS})
    out["supervised"] = sup
    rocs = {m: ML.roc(Y[m][test], proba[("All sources fused", m)])[:2] for m in MECHS}

    # logistic regression on the fused set, for comparison with a linear model
    sc_all = ML.Scaler().fit(Xs["All sources fused"][train])
    Ztr, Zte = sc_all.transform(Xs["All sources fused"][train]), sc_all.transform(Xs["All sources fused"][test])
    lr_res = {}
    for m in MECHS:
        lr = ML.LogisticRegression(C=0.5).fit(Ztr, Y[m][train])
        p = lr.predict_proba(Zte)[:, 1]
        lr_res[m] = dict(AUC=ML.roc(Y[m][test], p)[2], F1=ML.prf(Y[m][test], (p > 0.5).astype(int))[2])
    out["logistic_fused"] = lr_res
    print("logistic fused AUC", {m: round(v["AUC"], 3) for m, v in lr_res.items()})

    # dominant mechanism on healthy + single-mechanism test cases
    labels = ["H"] + list(MECHS)
    conf = {}
    for sname in ("Vibration only", "Static + geometry + bearings", "Inspection + NDT", "All sources fused"):
        Pm = np.column_stack([proba[(sname, m)] for m in MECHS])
        sel = np.isin(keys[test], labels)
        pred = np.where(Pm.max(axis=1) > 0.5, np.array(MECHS)[np.argmax(Pm, axis=1)], "H")
        conf[sname] = ML.confusion(list(keys[test][sel]), list(pred[sel]), labels).tolist()
        acc = np.trace(np.array(conf[sname])) / sel.sum()
        print(f"dominant-mechanism accuracy {sname}: {acc:.3f}")
    out["confusion"] = conf; out["confusion_labels"] = labels

    # ------------------------------------------------------------ C. prestress-loss severity
    pl_names = ["lev_min", "lev_mean", "vw_max", "vw_mean", "r_d_mean", "crack_flex"]
    Xpl = FE.matrix(F, pl_names)
    has = Y["PL"] == 1
    tr_i = np.where(train & has)[0]; te_i = np.where(test & has)[0]
    rng = np.random.default_rng(5)
    tr_sub = rng.choice(tr_i, min(450, len(tr_i)), replace=False)
    sc_pl = ML.Scaler().fit(Xpl[tr_sub])
    gp = ML.GPR().fit(sc_pl.transform(Xpl[tr_sub]), 100 * SEV["PL"][tr_sub], restarts=2)
    mu, sd = gp.predict(sc_pl.transform(Xpl[te_i]))
    yt = 100 * SEV["PL"][te_i]
    gpr_sc = ML.regression_scores(yt, mu)
    cover95 = float(np.mean(np.abs(yt - mu) <= 1.96 * sd))
    # physics-only inverse of the worst instrumented girder: -1.04 mm and +12.4 microstrain per 1 %
    # (twin sensitivities, chapter 10), combined by inverse variance (2.6 mm, 25 microstrain noise)
    e_lev, e_vw = -Xpl[te_i, 0] / 1.04, Xpl[te_i, 2] / 12.4
    w1, w2 = 1 / 2.5 ** 2, 1 / 2.0 ** 2
    phys = (w1 * e_lev + w2 * e_vw) / (w1 + w2)
    phys_sc = ML.regression_scores(yt, phys)
    rfr = ML.RandomForest(task="reg", n_trees=120, max_features="third", seed=3).fit(Xpl[tr_i], 100 * SEV["PL"][tr_i])
    rf_sc = ML.regression_scores(yt, rfr.predict(Xpl[te_i]))
    print("PL severity  GPR", {k: round(v, 2) for k, v in gpr_sc.items()}, "95% coverage", round(cover95, 2),
          "| RF", {k: round(v, 2) for k, v in rf_sc.items()}, "| levelling only", {k: round(v, 2) for k, v in phys_sc.items()})
    print("   GPR length scales:", dict(zip(pl_names, np.round(gp.ell, 2))))
    out["pl_regression"] = dict(gpr=gpr_sc, coverage95=cover95, rf=rf_sc, levelling_only=phys_sc,
                                length_scales=dict(zip(pl_names, gp.ell.tolist())))
    pl_arrays = dict(y=yt, mu=mu, sd=sd, phys=phys, other=np.array([rows[i]["n_mech"] for i in te_i]))

    # ------------------------------------------------------------ D. localisation (which girder)
    def loc_vec(f, r):
        return np.concatenate([f["_d"], f["_e"], [r["lev_G1"], r["lev_G6"], r["vw_G1"], r["vw_G3"], r["vw_G4"], r["vw_G6"],
                                                    f["phi_asym"]]])
    loc = np.array([r["loc"] for r in rows]).astype(int)
    sel = (np.array([r["n_mech"] for r in rows]) == 1) & (loc >= 0) & ~np.isin(keys, ["BR"])
    Xl = np.array([loc_vec(F[i], rows[i]) for i in range(len(rows))])
    rfl = ML.RandomForest(n_trees=150, max_features="sqrt", seed=9).fit(Xl[train & sel], loc[train & sel])
    pl_ = rfl.predict(Xl[test & sel])
    acc_loc = float((pl_ == loc[test & sel]).mean())
    acc_by = {m: float((pl_[keys[test & sel] == m] == loc[test & sel][keys[test & sel] == m]).mean()) for m in ("PL", "CD", "CR", "CO", "LS")}
    conf_loc = ML.confusion(list(loc[test & sel]), list(pl_), list(range(6)))
    print("localisation accuracy", round(acc_loc, 3), {k: round(v, 2) for k, v in acc_by.items()})
    out["localisation"] = dict(accuracy=acc_loc, by_mechanism=acc_by, confusion=conf_loc.tolist())

    # ------------------------------------------------------------ E. importance + effect sizes
    names = FE.SOURCES["All sources fused"]
    Xte = Xs["All sources fused"][test]
    imp = {}
    for m in MECHS:
        rf = models[("All sources fused", m)]
        fn = lambda X_, y_: ML.roc(y_, rf.predict_proba(X_)[:, 1])[2]
        im, base = ML.permutation_importance(fn, Xte, Y[m][test], n_rep=2, seed=1)
        imp[m] = im.tolist()
    out["importance"] = dict(names=names, values=imp)
    Xall = Xs["All sources fused"]
    mu_h, sd_h = Xall[H].mean(axis=0), Xall[H].std(axis=0) + 1e-9
    eff = {}
    for m in MECHS:
        k = (keys == m) & (SEV[m] >= np.median(SEV[m][keys == m]))
        eff[m] = ((Xall[k].mean(axis=0) - mu_h) / sd_h).tolist()
    out["effect_size"] = dict(names=names, values=eff)

    # ------------------------------------------------------------ save
    fused = {m: models[("All sources fused", m)] for m in MECHS}
    vib = {m: models[("Vibration only", m)] for m in MECHS}
    pickle.dump(dict(norm_library=norm, novelty=nov_model, fused=fused, vib=vib, gpr=(sc_pl, gp, pl_names),
                     loc=rfl, sources=FE.SOURCES), open("out/models.pkl", "wb"))
    np.savez("out/ml_arrays.npz", **{f"pl_{k}": v for k, v in pl_arrays.items()},
             **{f"roc_{m}_fpr": rocs[m][0] for m in MECHS}, **{f"roc_{m}_tpr": rocs[m][1] for m in MECHS})
    json.dump(out, open("out/ml_results.json", "w"), indent=1)
    print(f"[done in {time.time()-t0:.0f} s]")
