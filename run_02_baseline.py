"""
Step 2 - healthy baseline.
 a) commissioning tests on the as-built bridge (ambient vibration + static load test)
 b) updating of the design model -> digital twin
 c) one year of healthy monitoring (2 records a day) -> temperature / traffic normalisation
 d) baseline NDT survey of all three spans
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker process
from psc.perf import boost
boost()
import json, csv, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
from psc import inputs as P, twin as TW, dynamics as DY, scenarios as SC, condition as C

# "Truth" for the synthetic world: the real bridge is a little stiffer than Ecm suggests
# and the crash barriers work partly with the deck. None of these numbers is used by
# the twin - it has to find them from the commissioning measurements.
TRUE = dict(E_scale=1.06, barrier=0.35, E_dyn=1.24)
REF_T = 25.0


def measure(tw_true, ref, rng, T, rate=10):
    """Commissioning: model-free FDD (the design model only names the modes)."""
    st = TW.healthy_state()
    f, w = tw_true.modal(st, T)
    acc = DY.simulate(f, np.full(len(f), 0.015), w, tw_true.g.x, tw_true.g.y, rng, rate=rate)
    res, fdd_out = DY.oma_fdd(acc, ref[0], ref[2])
    fm = np.array([res[k]["f"] for k in SC.TRACK])
    d, e = tw_true.load_test(st, T)
    d = d + rng.normal(0, 0.05, 6); e = e + rng.normal(0, 3, 6)
    return fm, d, e, acc, res, fdd_out


def model_response(theta, T):
    tw = TW.Twin(E_scale=theta[0], barrier=theta[1], E_dyn=theta[2])
    f, w = tw.modal(TW.healthy_state(), T)
    d, e = tw.load_test(TW.healthy_state(), T)
    return np.array([f[k] for k in SC.TRACK]), d, e


# ---------------------------------------------------------------- baseline year (parallel)
_TW = None
def _init(theta_true):
    boost()
    global _TW
    _TW = TW.Twin(**theta_true)

def _record(args):
    day, hour, seed, ref = args
    rng = np.random.default_rng(seed)
    T, monsoon = SC.site_temperature(rng, day=day, hour=hour)
    st = TW.healthy_state(); st["mass"] = 1 + 0.006 * monsoon + rng.normal(0, 0.002)
    f, w = _TW.modal(st, T)
    rate = SC.traffic_rate(rng, hour)
    acc = DY.simulate(f, np.clip(rng.normal(0.015, 0.002, len(f)), 0.008, 0.03), w, _TW.g.x, _TW.g.y, rng, rate=rate)
    res, ex = DY.identify(acc, ref[2], ref[0], SC.TRACK)
    s = _TW.bearings(st, T)
    return dict(day=day, hour=hour, T=T, monsoon=monsoon, rate=rate,
                **{f"f{j+1}": res[k]["f"] for j, k in enumerate(SC.TRACK)},
                z1=res[SC.TRACK[0]]["zeta"], mac1=res[SC.TRACK[0]]["mac"],
                a_rms=ex["a_rms"] * 1e3 / 9.81, a_max=ex["a_max"] * 1e3 / 9.81,
                brg_A=float(np.mean(s[[0, 5], 0]) + rng.normal(0, 0.012)),
                brg_B=float(np.mean(s[[0, 5], 1]) + rng.normal(0, 0.012)))


def fit_normalisation(rows, keys=("f1", "f2", "f3", "f4")):
    T = np.array([r["T"] for r in rows]); la = np.log(np.array([r["a_rms"] for r in rows]))
    X = np.column_stack([np.ones_like(T), T - REF_T, la])
    out = {}
    for k in keys:
        y = np.array([r[k] for r in rows])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        out[k] = dict(coef=b.tolist(), sd=float(res.std(ddof=3)), mean=float(y.mean()),
                      per_degC_pct=float(100 * b[1] / b[0]))
    return out


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(2026)
    log = []
    def say(*a):
        s = " ".join(str(x) for x in a); print(s); log.append(s)

    # ---------------------------------------------- a) commissioning
    design_tw = TW.Twin()                                   # as designed
    ref_design = design_tw.reference(T=REF_T)
    true_tw = TW.Twin(**TRUE)
    T_comm = 27.0
    fm, dm, em, acc, res, fdd_out = measure(true_tw, ref_design, rng, T_comm)
    np.save("out/data/commissioning_record.npy", acc.astype(np.float32))
    np.savez("out/data/commissioning_fdd.npz", freqs=fdd_out[0], s=fdd_out[1][:, :4],
             shapes=np.array([res[k]["shape"] for k in sorted(res)]), modes=np.array(sorted(res)),
             f=np.array([res[k]["f"] for k in sorted(res)]))
    say("identified modes (FDD):", {k: (round(v["f"], 3), round(v["mac_model"], 3)) for k, v in sorted(res.items())})
    fd, dd, ed = model_response([1.0, 0.0, P.E_DYN_FACTOR], T_comm)
    say("== Commissioning (T = 27 degC, night) ==")
    say("measured f :", np.round(fm, 3).tolist(), " design model f:", np.round(fd, 3).tolist())
    say("measured d :", np.round(dm, 2).tolist(), " design model d:", np.round(dd, 2).tolist())
    say("measured e :", np.round(em, 1).tolist())

    # ---------------------------------------------- b) model updating
    def resid(th):
        f, d, e = model_response(th, T_comm)
        return np.concatenate([(f / fm - 1) / 0.002, (d - dm) / 0.10])
    sol = least_squares(resid, x0=[1.0, 0.1, 1.2], bounds=([0.8, 0.0, 1.0], [1.3, 1.0, 1.5]),
                        diff_step=1e-3, xtol=1e-6)
    th = sol.x
    J = sol.jac
    cov = np.linalg.inv(J.T @ J)
    sd = np.sqrt(np.diag(cov))
    fu, du, eu = model_response(th, T_comm)
    say("== Model updating ==")
    say(f"E_scale = {th[0]:.3f} +- {sd[0]:.3f}, barrier = {th[1]:.3f} +- {sd[1]:.3f}, E_dyn = {th[2]:.3f} +- {sd[2]:.3f}")
    say(f"(truth used to make the data: {TRUE})")
    say("updated f:", np.round(fu, 3).tolist(), " d:", np.round(du, 2).tolist())
    twin_params = dict(E_scale=float(th[0]), barrier=float(th[1]), E_dyn=float(th[2]),
                       sd=sd.tolist(), corr=(cov / np.outer(sd, sd)).tolist(),
                       design_f=fd.tolist(), design_d=dd.tolist(), meas_f=fm.tolist(),
                       meas_d=dm.tolist(), meas_e=em.tolist(), updated_f=fu.tolist(),
                       updated_d=du.tolist(), T=T_comm, truth=TRUE)
    tw = TW.Twin(E_scale=th[0], barrier=th[1], E_dyn=th[2])
    fref, wref, Sref = tw.reference(T=REF_T)
    # spatial filter: measured (commissioning) shapes where available, twin shapes elsewhere
    macs = {}
    for k, v in res.items():
        sm = v["shape"] * np.sign(np.dot(v["shape"], Sref[:, k]))
        macs[k] = DY.mac(sm, Sref[:, k])
        if macs[k] >= 0.9:          # a mixed shape of the near-repeated 2nd pair is not used
            Sref[:, k] = sm * np.abs(Sref[:, k]).max()
    say("MAC measured vs updated twin shapes:", {k: round(v, 3) for k, v in macs.items()})
    twin_params["mac_updated"] = {str(k): v for k, v in macs.items()}
    np.savez("out/twin_reference.npz", f=fref, w=wref, S=Sref)

    # ---------------------------------------------- c) baseline year
    say("== Baseline year (730 records) ==")
    jobs = []
    seed = 10_000
    for day in range(365):
        for hour in (2, 14):
            jobs.append((day + 0.5, hour, seed, (fref, wref, Sref))); seed += 1
    with Pool(14, initializer=_init, initargs=(TRUE,)) as pool:
        rows = pool.map(_record, jobs, chunksize=8)
    keys = list(rows[0].keys())
    with open("out/data/baseline_year.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    night = [r for r in rows if r["hour"] == 2]
    norm_all = fit_normalisation(rows)
    norm_night = fit_normalisation(night)
    for k in ("f1", "f2", "f3", "f4"):
        say(f"{k}: mean {norm_all[k]['mean']:.4f} Hz, {norm_all[k]['per_degC_pct']:+.3f} %/degC, "
            f"residual sd {norm_all[k]['sd']*1e3:.2f} mHz (night only {norm_night[k]['sd']*1e3:.2f} mHz)")
    Ts = np.array([r["T"] for r in rows])
    say(f"temperature range {Ts.min():.1f} - {Ts.max():.1f} degC; f1 range {min(r['f1'] for r in rows):.3f} - {max(r['f1'] for r in rows):.3f} Hz")
    say(f"bearing slopes A/B: {np.mean([r['brg_A'] for r in rows]):.3f} / {np.mean([r['brg_B'] for r in rows]):.3f} mm/degC")

    # ---------------------------------------------- d) baseline NDT (all spans, year 1)
    locs, fld = C.healthy_fields(rng, age=1.0)
    rd = C.measure_ndt(locs, fld, rng)
    with open("out/data/ndt_baseline.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["location", "span", "girder", "zone"] + list(rd.keys()))
        for i, l in enumerate(locs):
            w.writerow([l["id"], l["span"], f"G{l['girder']+1}", l["zone"]] + [round(float(rd[k][i]), 3) for k in rd])
    base_ndt = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), p05=float(np.percentile(v, 5)),
                        p95=float(np.percentile(v, 95))) for k, v in rd.items()}
    say("baseline NDT:", {k: (round(v["mean"], 2), round(v["sd"], 2)) for k, v in base_ndt.items()})

    # healthy static references at 25 degC for the record
    d25, e25 = tw.load_test(TW.healthy_state(), REF_T)
    json.dump(dict(twin=twin_params, norm=norm_all, norm_night=norm_night, ndt=base_ndt,
                   load_test_ref=dict(d=d25.tolist(), e=e25.tolist()),
                   f_ref=[float(fref[k]) for k in SC.TRACK],
                   bearing_ref=float(np.mean([r['brg_A'] for r in rows]))),
              open("out/baseline.json", "w"), indent=1)
    open("out/baseline_log.txt", "w").write("\n".join(log))
    print(f"[done in {time.time()-t0:.0f} s]")
