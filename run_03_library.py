"""
Step 3 - synthetic scenario library from the updated twin.
Each row is one assessment campaign of span S2 (vibration, load test, levelling,
VW strain, bearing movement, NDT, visual inspection) with its true labels.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker process
from psc.perf import boost
boost()
import csv, json, time, sys
import numpy as np
from multiprocessing import Pool
from psc import twin as TW, scenarios as SC

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
_TW = None; _REF = None


def _init():
    boost()
    global _TW, _REF
    b = json.load(open("out/baseline.json"))["twin"]
    _TW = TW.Twin(E_scale=b["E_scale"], barrier=b["barrier"], E_dyn=b["E_dyn"])
    r = np.load("out/twin_reference.npz")
    _REF = (r["f"], r["w"], r["S"])


def _job(seed):
    rng = np.random.default_rng(seed)
    mechs = SC.sample_scenario(rng)
    hour = int(rng.choice([2, 14], p=[0.7, 0.3]))
    feat, aux = SC.campaign(mechs, _TW, _REF, rng, hour=hour)
    y, sev, loc = SC.labels(mechs)
    row = dict(id=seed, hour=hour)
    ft = feat.pop("f_true")
    row.update({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in feat.items()})
    for j, v in enumerate(ft):
        row[f"f{j+1}_true"] = v
    for m in SC.MECHS:
        row[f"y_{m}"] = y[m]
        row[f"sev_{m}"] = sev[m]
    row["loc"] = loc
    row["n_mech"] = len(mechs)
    row["mechs"] = json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()} for m in mechs])
    return row


if __name__ == "__main__":
    t0 = time.time()
    with Pool(14, initializer=_init) as pool:
        rows = pool.map(_job, range(N), chunksize=10)
    keys = list(rows[0].keys())
    with open("out/data/library.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    counts = {m: sum(r[f"y_{m}"] for r in rows) for m in SC.MECHS}
    print(f"{N} campaigns, healthy {sum(r['n_mech'] == 0 for r in rows)}, pairs {sum(r['n_mech'] == 2 for r in rows)}, per mechanism {counts}")
    print(f"[done in {time.time()-t0:.0f} s]")
