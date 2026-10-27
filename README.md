# B33 - 33 m post-tensioned PSC I-girder bridge, CE60258 integrated big problem

One bridge followed from configuration to a maintenance decision:
design and prestressing -> healthy baseline -> inspection, NDT and monitoring -> deterioration ->
synthetic data -> features -> AI/ML assessment -> year-18 case study -> maintenance scheme.

**Report:** `report/B33_PSC_Bridge_Integrated_Study.pdf`
**Inspection app:** `inspection_app/index.html` (open in a browser, works offline)

## The bridge

- 3 simply supported spans of 33 m, deck 16.0 m (3 lanes of 3.5 m, 1.75 m footpaths, crash barriers)
- 6 precast M45 I-girders, 1.80 m deep, at 2.70 m; 225 mm M40 deck slab; end diaphragms + one cross girder at L/2
- 4 x 12T15 cables per girder, two-stage stressing, Pe = 7.17 MN per girder (23.5 % total loss)
- laminated elastomeric bearings 400 x 500, strip-seal joints at every pier and abutment

## Running it

Python 3 with numpy, scipy and matplotlib (nothing else; the ML methods are in `psc/mlkit.py`).

```
python run_01_design.py        # analysis, prestressing, checks          (~2 min)
python run_02_baseline.py      # commissioning, model updating, 1 year    (~5 min)
python run_03_library.py       # 3600 synthetic campaigns                 (~10-20 min)
python run_04_ml.py            # novelty, random forests, GPR, ...        (~2 min)
python run_05_case_study.py    # 18 years of monitoring + year-18 assessment
python verify.py               # independent hand checks -> out/verification.json
python make_figures.py all
python inspection_app/build_app.py
python build_report.py         # HTML + PDF (prints with Edge / Chrome headless)
```

## Layout

| Path | What |
|---|---|
| `psc/inputs.py` | all adopted data (the decisions) |
| `psc/section.py`, `grillage.py`, `vehicles.py` | sections, grillage FE, IRC:6 vehicles and worst-position search |
| `psc/prestress.py`, `design.py` | cables, losses, stage stresses, ULS, shear, bearings |
| `psc/twin.py`, `dynamics.py` | digital twin, traffic/ambient vibration, FDD and modal filtering |
| `psc/condition.py`, `scenarios.py`, `features.py` | inspection/NDT model, deterioration, campaigns, features |
| `psc/mlkit.py`, `maintenance.py` | ML methods, decision rules |
| `out/` | results (json), data (csv), figures |

All measurements are synthetic. The "real bridge" in the case study is a model with parameters hidden
from the assessment; see chapter 5 and 14 of the report.
