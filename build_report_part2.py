# Second half of the report (chapters 10-17, references, appendices). Executed from build_report.py.
from psc import twin as TW

twn = TW.Twin(E_scale=tw["E_scale"], barrier=tw["barrier"], E_dyn=tw["E_dyn"])
H0 = TW.healthy_state()
f_h, _ = twn.modal(H0); d_h, e_h = twn.load_test(H0)


def resp(st):
    f_, _ = twn.modal(st); d_, e_ = twn.load_test(st); lev, vw = twn.long_term(st); br = twn.bearings(st)
    return f_, d_, e_, lev, vw, br


sup, nov = ml["supervised"], ml["novelty"]
SRC = list(FE.SOURCES.keys())
SH = {"PL": "Prestress loss", "CD": "Concrete deterioration", "CR": "Flexural cracking", "CO": "Corrosion",
      "BR": "Bearing deterioration", "LS": "Local stiffness loss"}

# ====================================================================== 10
R.h1("Deterioration scenarios", "10")
R.p("""The brief leaves the type, location and size of deterioration to the student. I chose six mechanisms that are
realistic for a post-tensioned girder bridge in a wet coastal climate, gave each a cause on <i>this</i> bridge, a range of
values that I could defend, and a physical model of how it changes the structure. The ranges are deliberately wide, from
barely detectable to clearly serious, because the AI/ML part has to learn where detection starts.""")
R.table(["Mechanism", "Cause on B33", "How it is modelled in the twin", "Range and reason"], [
    ["Loss of prestress (PL)", "creep, shrinkage and relaxation larger than predicted; grout voids and strand corrosion near anchorages; anchorage slip",
     "ratio &eta; of actual to expected P<sub>e</sub> for one girder (60 %) or all girders (40 %). No change of uncracked stiffness (bonded tendons). Camber and VW strain change with creep factor 1 + &chi;&phi; = 1.8. Cracking when the soffit stress under 0.8 x design LL plus 0.6 x reverse temperature difference exceeds f<sub>ctm</sub>; cracked zones reopen in a load test beyond the decompression moment (I<sub>eff</sub> with I<sub>cr</sub>/I<sub>g</sub> = 0.35)",
     "3-35 % of P<sub>e</sub>. Uncertainty of the design long-term loss alone is &plusmn;30 % of 12.7 %, i.e. about 4 % of P<sub>e</sub>; the upper end covers corroded tendons reported in practice"],
    ["Concrete deterioration (CD)", "poorly compacted or chemically attacked concrete in one girder",
     "modulus reduction d<sub>E</sub> over a 6-15 m zone or the whole girder, strength reduction 0.9 d<sub>E</sub>, voids for UPV, faster carbonation",
     "d<sub>E</sub> = 5-30 %; above 30 % the girder would show obvious distress"],
    ["Flexural cracking (CR)", "an overloaded vehicle beyond the design load",
     "EI loss d<sub>s</sub> over 4-10 m near mid-span in the load test; in ambient vibration only 20-50 % of it, because cracks in a prestressed girder close again; residual crack 0.05-0.35 mm",
     "d<sub>s</sub> = 15-50 % local, i.e. a few per cent of the girder's overall stiffness"],
    ["Corrosion (CO)", "water from a leaking joint or a short drain spout on low-cover concrete of an outer girder; carbonation reaches the links, then strands in poorly grouted ducts",
     "strand area loss c over 2-8 m at a joint end or under a spout, local prestress loss = c, EI loss 0.25c from spalling; wetting, low cover, HCP shift and lower resistivity in the NDT; rust and spalling visible only once enough rust has formed",
     "c = 2-25 %; 80 % of cases on G1/G6, which carry the drains"],
    ["Bearing deterioration (BR)", "hardening of the elastomer with age and debris from a leaking joint packing the bearing",
     "shear stiffness multiplier 5-500 at one or both ends. In the 2-D grillage a horizontal restraint at bearing level becomes a coupled end-rotation spring k<sub>s</sub>e&sup2; [1 -1; -1 1], with k<sub>s</sub> the series stiffness of substructure, bearings and girder",
     "a bearing locked by debris behaves as several hundred times stiffer in shear; 60 % of cases come with a visibly leaking joint"],
    ["Local stiffness loss (LS)", "a tall vehicle on the service road striking a fascia girder; internal void or delamination",
     "EI loss 15-60 % over 1-3 m anywhere between 5 and 28 m; 75 % visible on outer girders, 40 % on inner girders (otherwise found only by UPV or hammer)",
     "local damage that removes a large part of one section but little of the span stiffness"],
], "Deterioration mechanisms and their models", widths=[13, 22, 41, 24], small=True)
R.p("""Healthy campaigns also vary: the age of the bridge (5-30 years), carbonation depth growing with the square root of age,
temperature, deck moisture and traffic. A campaign can contain one mechanism (60&nbsp;%), two at once (15&nbsp;%) or none (25&nbsp;%).""")
R.h2("What each mechanism does to the measured quantities", "10.1")
R.p("""Before generating any random data I swept each mechanism from zero to the top of its range on girder G1 (or on all
girders where that is the natural case) with the noise switched off. Figure [[fig:f13_sensitivity]] is the most important figure of the
monitoring part, because it shows what the measurements can and cannot see before any machine learning is involved.""")
R.fig("f13_sensitivity", "Noise-free response of the twin to each mechanism, from healthy (0) to the top of its range (1).")
pl = sens["PL"]; br = sens["BR"]; cdd = sens["CD"]
R.ul([f"<b>Prestress loss</b> leaves the frequencies and the load-test deflection unchanged until the girders crack, at roughly 25&nbsp;% loss here; then the load-test deflection jumps (+{max(pl['d']):.0f}&nbsp;% at 35&nbsp;%). Levelling ({pl['lev'][3]:.1f}&nbsp;mm at {35*3/6:.0f}&nbsp;% loss) and VW strain ({pl['vw'][3]:.0f}&nbsp;microstrain) respond from the start. This agrees with the theory of bonded prestressed beams, where the axial force does not soften the member (Hamed and Frostig, 2006).",
      f"<b>Bearing restraint</b> <i>raises</i> the frequencies, by up to {max(br['f1']):.1f}&nbsp;% for f<sub>1</sub> when both ends lock, because a horizontal force at the bearing level, below the neutral axis, resists the end rotation. A rise in frequency is therefore not proof of a healthy bridge. Restraint at only one end hardly changes the frequency (the free end takes the movement) but shows clearly in the bearing transducers.",
      f"<b>Concrete deterioration</b> of a whole girder lowers f<sub>1</sub> by up to {abs(min(cdd['f1'])):.1f}&nbsp;% and raises the load-test deflection by {max(cdd['d']):.0f}&nbsp;%; a mechanism that changes stiffness over a large volume is the one global vibration sees best.",
      "<b>Cracking, corrosion and local stiffness loss</b> change the global quantities by a fraction of a per cent to a few per cent. They are local, and the six-girder deck redistributes load around them."])
R.p(f"""These changes have to be read against the temperature effect: a 10&nbsp;&deg;C warmer day lowers f<sub>1</sub> by about
{abs(10*b['norm']['f1']['per_degC_pct']):.1f}&nbsp;%, which is as much as a 20&nbsp;% modulus loss of one girder. The brief warns that a change of
frequency should not be read as a particular type of damage, and Figure [[fig:f13_sensitivity]] shows why: four mechanisms lower the frequency,
one raises it, one leaves it alone, and the environment does both.""")

# ====================================================================== 11
R.h1("Synthetic data", "11")
R.p(f"""The synthetic data are produced by the updated twin (chapter 5), not by the 'true' model of the bridge, exactly as they
would have to be in practice: one can only simulate the bridge one has a model of. Every campaign goes through the same chain
as a real measurement campaign.""")
R.table(["Step", "What is generated", "Variability and noise"], [
    ["Scenario", "0, 1 or 2 mechanisms with random size and location (chapter 10)", "25 % healthy, 60 % single, 15 % pairs"],
    ["Environment", "effective bridge temperature from a seasonal-daily model of the site; monsoon deck moisture", "16-41 &deg;C; mass +0.6 % in the monsoon, &plusmn;0.2 % random"],
    ["Vibration", "modal superposition of 8 modes with moving lorries (bounce 1.3-2.6 Hz, 2-6 %), cars and broadband ambient forces; 12 channels, 30 min, 100 Hz", "traffic 3-60 vehicles per 10 min at night, 150-420 by day; damping 1.5 &plusmn; 0.2 %; sensor noise 2 &micro;g/&radic;Hz"],
    ["Identification", "the full chain of chapter 9 on each record", "f, MAC, damping, a<sub>RMS</sub>, a<sub>max</sub> as identified, not as simulated"],
    ["Load test", "4 x 300 kN lorries, deflection and strain of 6 girders", "lorry weight &plusmn;1 %, 0.05 mm, 3 microstrain"],
    ["Geometry and VW strain", "change beyond the healthy creep prediction", "creep-model error 2.5 mm and 25 microstrain (common to all girders), instrument 0.8 mm and 3 microstrain"],
    ["Bearings", "movement per &deg;C at both ends", "&plusmn;0.012 mm/&deg;C"],
    ["NDT", "30 test areas of span S2: rebound, UPV, cover, HCP, resistivity, carbonation, chloride", "as-built scatter of strength, cover and carbonation rate; reading errors of each test"],
    ["Visual inspection", "observations with sizes and condition states", "hairline cracks missed with probability e<sup>-w/0.06</sup>; condition state &plusmn;1 for 15 % of entries; early defects often not visible"],
], "Generation of one synthetic campaign", widths=[15, 48, 37], small=True)
R.p(f"""{ml['n']} campaigns were generated ({ml['counts']['H']} healthy, about {int(np.mean([ml['counts'][m] for m in SC.MECHS]))} for each single
mechanism and {ml['counts']['pair']} with two mechanisms), each with 52 raw quantities and its true labels, in
<code>out/data/library.csv</code>. The examples below are in the format of the brief.""")
# example tables
st_list = [("Healthy", H0)]
for loss in (0.20, 0.27, 0.33):
    st = TW.healthy_state(); st["eta"][:] = 1 - loss; st_list.append((f"Prestress loss {loss*100:.0f} % (all girders)", st))
rows = []
for name, st in st_list:
    f_, d_, e_, lev, vw, brr = resp(st)
    rows.append([name, f"{d_[0]:.2f}", f"{d_[2]:.2f}", f"{e_[0]:.0f}", f"{lev.mean():+.1f}", f"{vw.mean():+.0f}", f"{f_[0]:.3f}"])
R.table(["Condition", "Load-test deflection G1 (mm)", "Load-test deflection G3 (mm)", "Strain G1 (microstrain)", "Level change (mm)",
         "VW change (microstrain)", "f<sub>1</sub> (Hz)"], rows,
        "Synthetic displacement, strain and frequency for increasing prestress loss (noise-free, 25 &deg;C)", num_cols=(1, 2, 3, 4, 5, 6), small=True)
R.p("""The displacement table makes the point of chapter 10 in numbers: up to 20&nbsp;% loss the load test and the
frequency show nothing, whereas the level and the VW strain have already moved by about 21&nbsp;mm and 250&nbsp;microstrain.
Once the girders crack (27&nbsp;%) the load-test deflection increases sharply.""")
ndt_rows = [r for r in csv.DictReader(open("out/data/case_ndt_survey.csv")) if r["span"] == "S2" and r["girder"] in ("G1",)]
rows = []
for r in ndt_rows:
    rows.append([r["location"], f"{float(r['rebound']):.0f}", f"{float(r['upv']):.2f}", f"{float(r['cover']):.0f}",
                 f"{float(r['carbonation']):.0f}", f"{float(r['hcp']):.0f}", f"{float(r['resistivity']):.0f}", r["cluster"]])
R.table(["Location", "Rebound", "UPV (km/s)", "Cover (mm)", "Carbonation (mm)", "HCP (mV)", "Resistivity (k&Omega;cm)", "Cluster (chapter 13)"], rows,
        "Synthetic NDT data along girder G1 of span S2 at year 18", num_cols=(1, 2, 3, 4, 5, 6), small=True)
R.p("""Along G1 the readings change towards the joint end (Z5): lower resistivity and more negative potentials where the
concrete is wet, cover lower than specified and carbonation close to or beyond it. UPV and rebound are less affected, because
the concrete itself is sound; corrosion is a problem of the steel and its environment, not of concrete quality.""")
st = TW.healthy_state(); st["kh_mult"][:, :] = 100
f_br, *_ = resp(st)
st = TW.healthy_state(); st["conc"].append((0, 0, P.L, 0.2))
f_cd, *_ = resp(st)
R.table(["Mode", "Healthy (Hz)", "Concrete deterioration, G1 -20 % E (Hz)", "Both bearing lines restrained x100 (Hz)", "Temperature +10 &deg;C (Hz)"],
        [[n, f"{f_h[k]:.3f}", f"{f_cd[k]:.3f} ({100*(f_cd[k]/f_h[k]-1):+.1f} %)", f"{f_br[k]:.3f} ({100*(f_br[k]/f_h[k]-1):+.1f} %)",
          f"{f_h[k]*(1+10*b['norm']['f1']['per_degC_pct']/100):.3f} ({10*b['norm']['f1']['per_degC_pct']:+.1f} %)"]
         for n, k in (("f<sub>1</sub> bending", 0), ("f<sub>2</sub> torsion", 1), ("f<sub>3</sub> transverse", 2))],
        "Synthetic vibration data: two deterioration cases and a temperature change", num_cols=(1, 2, 3, 4), small=True)
R.p("""The brief's example changes f<sub>1</sub> from 3.20 to 3.05&nbsp;Hz (-4.7&nbsp;%). Since f varies with &radic;EI, a change of that
size on B33 would need a loss of about 9&nbsp;% of the bending stiffness of the whole deck, more than any single mechanism in my ranges
produces, or simply a temperature rise of about 30&nbsp;&deg;C. The frequency alone cannot tell these apart. The pattern across the
modes carries more information than any one of them: deterioration of G1 lowers f<sub>1</sub> more than f<sub>2</sub> and f<sub>3</sub>,
bearing restraint raises f<sub>1</sub> and f<sub>2</sub> but hardly f<sub>3</sub>, and temperature moves all three by nearly the same
percentage. That is why all four tracked modes are kept as features.""")

# ====================================================================== 12
R.h1("Feature extraction", "12")
R.p("""A feature should have a physical meaning, respond to at least one mechanism, and be stable on the healthy bridge. Every
feature here is expressed as a change from the structure's <i>own</i> healthy baseline: the frequency residual after the
temperature/traffic model of chapter 5, the load-test deflection relative to the commissioning test (corrected to 25&nbsp;&deg;C
with the modulus-temperature relation), levels and VW strain relative to the creep prediction, bearing movement relative to
&alpha;L/2. This matters because the models are trained on twin data and applied to the real bridge; a 0.5&nbsp;% error in the
twin's absolute frequency would otherwise look like damage.""")
rows = []
mean = {"VIB": "global stiffness, mass, boundary conditions; MAC and edge asymmetry indicate which side changed",
        "LT": "static stiffness of each girder under a known load; asymmetry locates a soft girder",
        "GEO": "permanent deformation: loss of precompression (camber, soffit strain) and creep of a softened girder",
        "BRG": "freedom of the bearings; restraint at one end doubles the movement at the other",
        "NDT": "condition of concrete (UPV, rebound) and of steel (HCP, resistivity, cover vs carbonation)",
        "VI": "what an inspector sees: cracks, spalls, rust, bearing and joint distress"}
for g, names in FE.GROUPS.items():
    rows.append([g, ", ".join(FE.LABELS[n] for n in names), mean[g]])
R.table(["Group", "Features", "Physical meaning"], rows, f"The {sum(len(v) for v in FE.GROUPS.values())} features used by the AI/ML models", widths=[8, 50, 42], small=True)
R.p("""The brief's suggested vector x = [f<sub>1</sub>, f<sub>2</sub>, a<sub>RMS</sub>, &delta;<sub>max</sub>, &epsilon;<sub>max</sub>, UPV,
rebound] was the starting point. Two of its entries were changed after looking at the data. a<sub>RMS</sub> and a<sub>max</sub>
depend on traffic far more than on condition (the night-day ratio is about 2), so they are used to <i>normalise</i> the frequencies,
not as condition features. And &delta;<sub>max</sub> was replaced by the change of mean deflection and the G1-G6 asymmetry, which
separate a global from a local change. Figure [[fig:f14_effect_size]] shows, before any classifier, how strongly each feature reacts to each
mechanism, as the shift of its mean in units of the healthy standard deviation.""")
R.fig("f14_effect_size", "Effect size of each mechanism on each feature (library, upper-half severities). Blue: decrease, red: increase.", 80)
R.fig("f15_distributions", "Distributions of six features for healthy (H) and single-mechanism campaigns (boxes: quartiles, whiskers 1.5 IQR).")
R.p("""Reading Figure [[fig:f14_effect_size]] column by column already gives most of the diagnosis rules an engineer would write: prestress loss
moves levels and VW strain; bearing restraint moves the bearing movement ratio, raises f<sub>1</sub> and often comes with a leaking
joint; corrosion moves potentials, resistivity and cover against carbonation; concrete deterioration lowers UPV and rebound and,
over a whole girder, the frequencies; cracking and local damage show in the load test and in inspection. The value of the ML is
to combine these weak and overlapping signals consistently, to put a number on the confidence and to estimate the size.""")

# ====================================================================== 13
R.h1("AI/ML-assisted condition assessment", "13")
R.p(f"""All models were written in numpy and scipy for this work (scikit-learn could not be installed on the computer used, and
writing them made the assumptions explicit): a Mahalanobis-distance novelty detector with a shrunk covariance, CART trees and
random forests on histogram-binned features, multinomial logistic regression, Gaussian-process regression with one length scale
per input, k-means and a Gaussian mixture fitted by EM. The library was split 70/30, stratified by class
({ml['n_train']} training and {ml['n_test']} test campaigns); every number below is on the test set. The question each model answers
is stated with it.""")
R.h2("Is the bridge different from its healthy self? (novelty detection)", "13.1")
R.p("""The detector is trained on healthy campaigns only, which is the realistic situation: a real bridge has a healthy baseline
but no labelled examples of its own damage. The threshold is the 99th percentile of the healthy training distances, i.e. a
designed false-alarm rate of 1&nbsp;%.""")
R.table(["Data source"] + ["False alarm"] + [SH[m] for m in SC.MECHS],
        [[s, pct(nov[s]["false alarm"])] + [pct(nov[s][m]) for m in SC.MECHS] for s in SRC],
        "Share of test campaigns flagged by the novelty detector", num_cols=tuple(range(1, 8)), small=True)
R.fig("f16_novelty", "Novelty detection by data source, and probability of detection against severity (fused: solid; vibration only: dotted).")
R.p(f"""With vibration alone the detector finds most cases of concrete deterioration ({pct(nov['Vibration only']['CD'])}) and local
stiffness loss ({pct(nov['Vibration only']['LS'])}), fewer bearing cases ({pct(nov['Vibration only']['BR'])}, only those restrained at both ends),
and almost no prestress loss ({pct(nov['Vibration only']['PL'])}) or corrosion ({pct(nov['Vibration only']['CO'])}). This is
Figure [[fig:f13_sensitivity]] again, now with noise and environment: vibration monitoring alone is the wrong tool for the two mechanisms that most often
govern the life of a post-tensioned bridge. Adding levelling, VW strain and bearing movement, all of which can run without closing
the bridge, lifts the detection of prestress loss to {pct(nov['Continuous monitoring (no closure)']['PL'])} and of bearing problems
to {pct(nov['Continuous monitoring (no closure)']['BR'])}. With all sources fused, detection is {pct(min(nov['All sources fused'][m] for m in SC.MECHS))}
or better for every mechanism at a false-alarm rate of {pct(nov['All sources fused']['false alarm'])}; the misses are the smallest
severities (Figure [[fig:f16_novelty]], right).""")
R.h2("Which mechanism? (supervised detection and data fusion)", "13.2")
R.p("""For each mechanism a random forest (100 trees, depth 12, &radic;p features per split) was trained to answer 'is this
mechanism present?', so two mechanisms can be reported at once. The same model was trained on each data source, which is the
data-fusion study: what does each source add?""")
R.table(["Data source"] + [SH[m] for m in SC.MECHS] + ["macro-F1"],
        [[s] + [f"{sup[s][m]['AUC']:.2f} / {sup[s][m]['F1']:.2f}" for m in SC.MECHS] + [f"<b>{sup[s]['macro_F1']:.2f}</b>"] for s in SRC],
        "ROC-AUC / F1 per mechanism and data source (test set)", num_cols=tuple(range(1, 8)), small=True)
R.fig("f17_supervised", "ROC-AUC per mechanism and data source, and the macro-averaged F1 score.")
lr = ml["logistic_fused"]
R.p(f"""The macro-F1 rises from {sup['Vibration only']['macro_F1']:.2f} with vibration alone to {sup['Static + geometry + bearings']['macro_F1']:.2f}
with static, geometric and bearing data, {sup['Inspection + NDT']['macro_F1']:.2f} with inspection and NDT, and
{sup['All sources fused']['macro_F1']:.2f} with everything. No single source is enough, and the sources complement each other in a
way that follows the physics: NDT and inspection see corrosion and surface damage, levelling and VW strain see prestress, the
bearing transducers see the bearings, the load test sees cracking. A logistic regression on the fused features reaches AUC
{min(v['AUC'] for v in lr.values()):.2f}-{max(v['AUC'] for v in lr.values()):.2f}, close to the forest; the choice of features mattered
far more than the choice of algorithm.""")
R.fig("f18_confusion", "Dominant mechanism on healthy and single-mechanism test campaigns: vibration only against all sources fused.")
R.p("""The confusion matrices show what goes wrong. With vibration alone, prestress loss and corrosion are mostly called healthy,
and cracking, local stiffness loss and concrete deterioration are confused with each other, because all three lower the
frequencies a little and differ mainly in where and how much. With all sources, the remaining errors are early prestress loss and
the smallest cases of cracking and concrete deterioration, i.e. cases that are hard to see by any means.""")
R.h2("How much prestress has been lost? (regression with uncertainty)", "13.3")
g = ml["pl_regression"]
R.p(f"""The size of a prestress loss decides between monitoring and strengthening, so a point estimate without its uncertainty
is of little use. A Gaussian-process regression with inputs worst-girder sag, mean level, largest and mean VW strain, load-test
deflection change and flexural crack width was trained on campaigns with prestress loss. On the test set it gives a mean absolute error of
<b>{g['gpr']['MAE']:.1f}&nbsp;%</b> of P<sub>e</sub> (R&sup2; = {g['gpr']['R2']:.2f}), and the 95&nbsp;% band contains the true value in
{pct(g['coverage95'])} of cases, so the stated uncertainty is honest (Figure [[fig:f20_pl_regression]]). A random forest gives a similar error
({g['rf']['MAE']:.1f}&nbsp;%). A purely physical inverse from the twin's sensitivities (1.04&nbsp;mm and 12.4&nbsp;microstrain per 1&nbsp;% loss)
on the worst instrumented girder is worse (MAE {g['levelling_only']['MAE']:.1f}&nbsp;%), mainly because it cannot see losses in G2 and G5,
which have no gauges, and is disturbed by other mechanisms in the two-mechanism cases. The learned length scales tell which inputs
the GP relies on: the shortest belong to {' and '.join(FE.LABELS[k] for k in sorted(g['length_scales'], key=g['length_scales'].get)[:2])}.""")
R.fig("f20_pl_regression", "Estimated against true extra prestress loss on test campaigns: Gaussian process and physics-only inverse.")
loc = ml["localisation"]
R.h2("Where? (localisation)", "13.4")
R.p(f"""A random forest on the girder-wise load-test deflections and strains, levels, VW strains and mode-shape asymmetry identifies
the affected girder in <b>{pct(loc['accuracy'])}</b> of single-mechanism test cases ({', '.join(f'{SH[k].lower()} {pct(v)}' for k, v in loc['by_mechanism'].items())}).
Corrosion is located worst because its global signature is small and often near a support, where the load test is insensitive; for
corrosion the location comes from the NDT grid instead (section 14.3).""")
R.h2("Do the models use the physics? (importance)", "13.5")
R.fig("f19_importance", "Permutation importance (drop of test AUC when a feature is shuffled) for the fused random forests.")
R.p("""The permutation importances (Figure [[fig:f19_importance]]) agree with the effect sizes of Figure [[fig:f14_effect_size]] and with the mechanisms of chapter 10: the
prestress model relies on levels and VW strain, the bearing model on the bearing movement ratio, the corrosion model on potentials,
resistivity and rust, the concrete model on UPV and rebound. I regard this agreement as a necessary condition for trusting the
models; a model that detected corrosion from the damping of mode 1, say, would have learned a quirk of the simulation.""")
R.h2("Limitations", "13.6")
R.ul(["""All data are synthetic. The scores measure how well the method works <i>if the twin and the noise models are right</i>; on a
real bridge they would be lower and would need recalibration with the bridge's own data.""",
      """The twin is a linear grillage. Bond loss, crack interface behaviour, bearing friction and nonlinear soil-structure effects are
represented only through simple rules.""",
      """Labels are perfect in the library. In practice the label of a past campaign is often unknown, which is why novelty detection on
the healthy baseline, not the supervised models, is the first line of the assessment.""",
      """Six mechanisms were considered. A different mechanism (fatigue of a strand, scour of a pier) would appear as 'novel but
unclassified', which is the right output for an unknown situation."""])

# ====================================================================== 14
R.h1("Year-18 assessment of B33", "14")
cm = cs["ml"]; asm = cs["assessment"]; tr = cs["trends"]
R.p("""This chapter puts everything together for the bridge itself. The 'real' bridge (the hidden model of chapter 5) was aged for
18 years with a deterioration history that was not used by the assessment. Monthly monitoring records (night-time vibration,
bearing movement, VW strain) and annual levels were produced for the whole period, and in year 18 a principal inspection with NDT
of all spans and a load test was carried out. The assessment below uses only these measurements, the baseline of chapter 5 and the
models of chapter 13.""")
R.h2("What the monitoring showed", "14.1")
R.fig("f21_case_trends", "18 years of monitoring of span S2: monthly values and EWMA (&lambda; = 0.2) with &plusmn;3&sigma; limits from the first two years. Grey bars: onset of the hidden changes.")
rows = [[k, (f"{v['first_alarm_year']:.1f}" if v['first_alarm_year'] is not None else "no alarm"), f"{v['slope_last3y_per_year']:+.3f}"]
        for k, v in tr.items()]
R.table(["Monitored quantity", "First EWMA alarm (year)", "Theil-Sen slope, last 3 years (per year)"], rows,
        "Trend identification on the monitoring record", num_cols=(1, 2))
R.p(f"""The bearing transducers gave the first and clearest signal (year {tr['bearing B movement']['first_alarm_year']:.1f}): the
movement at end B fell towards zero and that at end A rose, the signature of a bearing that is being locked at one end while the
other end takes over the movement. The f<sub>1</sub> residual also alarmed, in year {tr['f1 residual']['first_alarm_year']:.1f}, but
<i>upwards</i>: the span had become stiffer. On its own that could pass for a healthy sign or a temperature-model error; next to the
bearing record it is the restraint of chapter 10. The G1 level dropped (alarm in year {tr['level G1']['first_alarm_year']:.1f}) and the
VW gauge in G1 drifted towards tension (year {tr['VW strain G1']['first_alarm_year']:.1f}), the pattern of lost precompression in one
girder. All alarms came one to two years after the hidden onsets, which is about what EWMA charts with these limits can do with
monthly data. With vibration alone the record would have said 'stiffer', which is the wrong way round for a deteriorating bridge.""")
R.h2("Principal inspection and NDT, year 18", "14.2")
obs = [o for o in json.load(open("inspection_app/seed_observations.json"))]
key = sorted([o for o in obs if o["cs"] >= 3], key=lambda o: -o["cs"])[:14]
R.table(["Span", "Component", "Location", "Observation", "Size", "CS", "Note"],
        [[o["span"], o["component"], o["location"], o["observation"], o["size"], o["cs"], o["note"]] for o in key],
        "Main observations of the year-18 inspection (condition state 3 and worse)", num_cols=(5,), small=True)
gr = cs["decision"]["groups"]
R.p(f"""The bridge condition index is <b>{cs['decision']['bci']:.0f}</b> out of 100. The worst component groups are
{', '.join(f'{k} (CS{v})' for k, v in sorted(gr.items(), key=lambda kv: -kv[1])[:4])}. All the serious findings are within a few
metres of the expansion joint over pier P2: the torn seal, silt around the S2 bearings at end B and the S3 bearings at end A,
leakage and spalling on the pier cap, a short drain spout and rust-stained splitting cracks and spalling at the anchorage end of G1.
The inspector rated two of these cracks, 0.39 and 0.45&nbsp;mm along the duct line in zones Z4-Z5, as CS5; by the width rule they
would be CS4. I kept the inspector's rating in the record, because rust coming out of a crack along a tendon duct next to an
anchorage is exactly the warning sign that preceded the Ynys-y-Gwas collapse (Woodward and Williams, 1988), but the category of the
bridge was decided after the structural check of section 14.3, not from that single rating.""")
cl = cs["ndt_clusters"]
hot = [r["location"] for r in csv.DictReader(open("out/data/case_ndt_survey.csv")) if float(r["hcp"]) < -350]
risk = [c for c in cl["clusters"] if c["name"] != "sound / dry"]
R.p(f"""The 90 NDT test areas were clustered with a Gaussian mixture on standardised UPV, rebound, half-cell potential,
log-resistivity and cover minus carbonation depth. The BIC was lowest for {cl['k_best']} components (Table below), so two groups were
kept: {cl['clusters'][0]['n'] if cl['clusters'][0]['name']=='sound / dry' else cl['clusters'][1]['n']} areas of sound, dry concrete and
{risk[0]['n']} areas of wet concrete with corrosion risk. All {risk[0]['n']} lie at girder ends next to the joints over the two piers
({', '.join(risk[0]['members'])}). Inside that group the ASTM C876 bands do the rest of the work: only {', '.join(hot)} have potentials
below -350&nbsp;mV together with resistivity near 10&nbsp;k&Omega;cm, i.e. probable active corrosion; the others are wet but still
passive. k-means with the same number of groups agrees with the mixture for {pct(cl['kmeans_agreement'])} of location pairs.""")
R.fig("f22_case_ndt", "Year-18 NDT survey of all girders, grouped by a Gaussian mixture; right: potential against resistivity with the ASTM C876 bands.")
rows = []
for c in sorted(cl["clusters"], key=lambda c: c["hcp"], reverse=True):
    rows.append([c["name"], c["n"], f"{c['upv']:.2f}", f"{c['rebound']:.1f}", f"{c['hcp']:.0f}", f"{c['rho']:.0f}", f"{c['margin']:+.0f}",
                 ", ".join(c["members"]) if c["n"] <= 8 else f"{c['n']} areas"])
R.table(["Cluster", "Areas", "UPV", "Rebound", "HCP (mV)", "Resistivity", "Cover - carbonation (mm)", "Members"], rows,
        "NDT clusters (mean values)", num_cols=(1, 2, 3, 4, 5, 6), small=True)
R.p("""The clustering is useful because it is unsupervised: it does not know about the joints, yet it separates the wet end zones
next to both piers from the rest of the bridge, which tells the engineer that the second joint (P1) also needs attention. The engineering
bands then single out the G1 end zone of S2. The concrete itself (UPV, rebound) is sound almost everywhere, so this is a problem of
water and steel, not of concrete quality.""")
R.h2("AI/ML output and the engineer's reading", "14.3")
R.fig("f23_case_ml", "Year-18 campaign: mechanism probabilities from the fused and the vibration-only models; change of the load-test response per girder since commissioning.")
pf, pv = cm["p_fused"], cm["p_vibration"]
R.p(f"""The fused Mahalanobis distance is {math.sqrt(cm['novelty_score']):.0f} against a threshold of
{math.sqrt(cm['novelty_threshold']):.1f}: the bridge is clearly no longer its healthy self (most of the distance comes from inspection
features that are always zero on the healthy bridge, such as rust stains). The fused models give the highest probabilities to
{', '.join(f'{SH[m].lower()} ({pf[m]:.2f})' for m in sorted(pf, key=pf.get, reverse=True)[:3])}. The vibration-only models give
{', '.join(f'{SH[m].lower()} {pv[m]:.2f}' for m in sorted(pv, key=pv.get, reverse=True)[:2])}: they find the bearings, which stiffen the span,
and nothing else. The localisation model gives G1 a probability of {cm['loc_proba'][0]:.2f}.""")
R.p(f"""The prestress estimate needed more care. My first version of the Gaussian process, which took the largest crack width of any
kind as an input, gave <b>18&nbsp;%</b> for this campaign. It had been misled by the 0.45&nbsp;mm rust-stained splitting cracks at the joint
end: in its training data a visible crack on a girder almost always meant a flexural crack, which appears only after a loss of about
25&nbsp;%. After splitting the crack feature into flexural cracks and cracks along bars or ducts and retraining, the model gives
{cm['pl_gpr_mean']:.1f} &plusmn; {1.96*cm['pl_gpr_sd']:.1f}&nbsp;% (95&nbsp;% band), and its test error also improved. Because the levels and the VW
strain moved mainly in G1, I checked this by inverting the G1 measurements directly with the twin sensitivities (1.04&nbsp;mm and
12.4&nbsp;microstrain per 1&nbsp;% loss), which gives {cm['pl_G1_combined']:.1f} &plusmn; {1.96*cm['pl_G1_sd']:.1f}&nbsp;% for G1 (levelling
{cm['pl_G1_levelling']:.1f}&nbsp;%, VW strain {cm['pl_G1_vw']:.1f}&nbsp;%). The two estimates overlap; I used the direct G1 estimate for the
structural check because it rests on two measurements whose physics is clear, and carried the GPR value as an upper indication.""")
R.p("""Is this physically reasonable? I checked it against the mechanisms rather than accepting it. A leaking joint explains the
bearing distress at both spans next to it, the wet concrete and low resistivity at the girder ends, and, through the anchorages at
that end, a possible loss of prestress in G1; the slight rise of f<sub>1</sub> fits restrained bearings; the absence of any change in
UPV and rebound rules out a concrete problem; the load test changed little, which is consistent with a girder that has lost some
precompression but has not cracked. The ML output is therefore consistent with one cause, the joint, and three consequences.
What it cannot tell is whether the strands inside the ducts at the G1 anchorage are corroding; only an intrusive tendon investigation
can show that, and the maintenance plan has to include one.""")
R.p(f"""<b>Structural check with the estimated loss.</b> Using the G1 estimate ({asm['loss_used_pct']:.1f}&nbsp;%), the soffit margin
under the rare combination becomes {asm['sls_margin_design']:+.2f}&nbsp;MPa on the design basis (0.9 P<sub>m</sub>) and
{asm['sls_margin_mean']:+.2f}&nbsp;MPa with mean prestress, against cracking at -f<sub>ctm</sub>; the quasi-permanent soffit stress is still
{asm['qp_soffit_mean']:+.1f}&nbsp;MPa compression. The ULS flexural capacity at mid-span is not affected by a loss of force (the strands
yield anyway), M<sub>Ed</sub>/M<sub>Rd</sub> = {asm['uls_ratio']:.2f}. At the corroded B end the vertical prestress component falls, and
the net design shear at 2.75&nbsp;m rises from {asm['shear_Vnet_design']:.0f} to {asm['shear_Vnet']:.0f}&nbsp;kN, still below
V<sub>Rd,max</sub> = {asm['shear_VRdmax']:.0f}&nbsp;kN and covered by the adopted links. The girder is therefore safe today, but it is close
to losing its design margin against cracking, and its condition will worsen if water keeps reaching the anchorage.""")
R.h2("Check against the hidden history", "14.4")
tru = cs["truth"]
desc = []
for m in tru:
    if m["type"] == "BR":
        desc.append(f"bearing restraint at end B (stiffness x{np.mean(m['kh_mult'][0][1:2] if False else [row[1] for row in m['kh_mult']]):.0f}) and slight hardening at end A from year {cs['onset']['bearing']:.0f}")
    if m["type"] == "CO":
        desc.append(f"corrosion of G1 at the B end, x = {m['x0']:.1f}-{m['x1']:.1f} m, strand area loss {100*m['loss']:.1f}&nbsp;% from year {cs['onset']['corrosion']:.0f}")
    if m["type"] == "PL":
        desc.append(f"extra prestress loss in G1 of {100*m['loss']:.1f}&nbsp;%")
R.p(f"""The history used to create the data was: joint seal at pier P2 failing in year {cs['onset']['joint']:.0f}; {'; '.join(desc)}. The
assessment found all three mechanisms and their common cause, located them correctly, and estimated the prestress loss of G1 within
its stated uncertainty. It did not, and could not, measure the strand corrosion inside the ducts; it pointed to where that has to be
investigated.""")

# ====================================================================== 15
R.h1("Maintenance scheme", "15")
R.p("""The maintenance decision combines all sources, as the brief asks: visual inspection + NDT + structural response + vibration
monitoring + AI/ML. In my scheme the ML output is one line of evidence among them. An action is triggered by the evidence
itself, a crack, a potential map, a bearing that does not move, and the ML result mainly sets the priority and tells what to check
next. A probability of 0.8 for 'prestress loss' is not a reason to strengthen a girder; it is a reason to measure the force.""")
R.h2("Decision framework", "15.1")
R.raw("""<table style="font-size:8.4pt;text-align:center"><tr>
<td style="width:22%">Condition assessment<br><span class="small">VI + NDT + response + vibration + AI/ML</span></td><td>&rarr;</td>
<td style="width:22%">No alarm, CS &le; 2:<br><b>routine monitoring</b><br><span class="small">pre/post-monsoon inspection, monthly data review</span></td><td>or</td>
<td style="width:22%">Alarm or CS &ge; 3:<br><b>detailed investigation</b><br><span class="small">special inspection, targeted NDT, assessment</span></td><td>&rarr;</td>
<td style="width:18%"><b>Maintenance / repair</b><br><span class="small">remove the cause first, then the damage</span></td><td>&rarr;</td>
<td style="width:16%"><b>Reinspection and monitoring</b><br><span class="small">new baseline after works</span></td></tr></table>""")
R.table(["Evidence pattern", "Interpretation", "Action"], [[a, b_, c] for a, b_, c in MT.DECISION_MATRIX],
        "Decision matrix used for B33", widths=[34, 30, 36], small=True)
R.h2("Maintenance plan for B33 after the year-18 assessment", "15.2")
dec = cs["decision"]
R.p(f"""The assessment places the bridge in the category <b>{dec['category']}</b>. The plan (Table below) first removes the cause, water
at the pier P2 joint, then repairs the damage it has done, then measures what cannot be inferred, and finally re-baselines the
monitoring so that the repaired bridge is again compared with itself.""")
R.table(["Action", "When", "Triggered by", "Notes"], [[a, w_, r_, n_] for a, w_, r_, n_ in dec["actions"]],
        "Actions, time frames and the evidence behind each", widths=[33, 13, 32, 22], small=True)
iv = dec["intervals"]
R.p(f"""<b>Sequence and justification.</b> Drains and the joint come first because every other defect here is a consequence of
water; repairing concrete or bearings while the joint still leaks would waste the repair. The bearings of S2 end B (and S3 end A,
same joint) are replaced by lifting each span on jacks placed under the end diaphragm, which was designed as a jacking beam; the lift
is limited to 10&nbsp;mm and the joint is closed to traffic for one night per span. Concrete repair at the G1 end zone and the pier cap
follows the half-cell map, with an anti-carbonation coating on repaired and low-cover areas and a migrating corrosion inhibitor where
potentials are in the uncertain band. The tendon investigation (GPR to locate the ducts, impact-echo along them, borescope through a
small drilled port at the anchorage and at duct high points) and a lift-off or stress-release test on G1 decide the last step: if the
measured force confirms a loss that removes the margin against cracking, external post-tensioning (two straight 12T15 tendons per
girder with deviators at the mid-span cross girder and anchors on the end diaphragms) restores it; if not, the girder is kept under
closer monitoring.""")
R.p(f"""<b>Inspection and monitoring intervals.</b> Until the works are complete: routine inspection {iv['routine']}, monitoring
{iv['monitoring']}; principal inspection {iv['principal']}. Crack gauges and two extra VW gauges at the G1 end zone and bearing
transducers on all girders of S2 are added at once and kept for at least two years after the repair. After the works, the
commissioning steps of chapter 5 are repeated (ambient test, load test, NDT of the repaired zones) and become the new baseline;
the bridge returns to the routine regime (pre/post-monsoon routine inspections, principal inspection every three years, quarterly
review of the monitoring) when two years of data after repair stay inside the new limits.""")
R.p("""<b>Longer-term improvement.</b> Each of the four expansion joints is a source of water. When the S2-S3 joint is replaced, it
is worth considering a continuity (link) slab over the piers, which would leave only the two abutment joints. The drain spouts of
all spans should be extended at least 100&nbsp;mm below the girder soffit with a drip groove, which costs little and removes the
second source of wetting found in this assessment.""")

# ====================================================================== 16
R.h1("Use of AI tools and verification", "16")
R.p("""As permitted by &sect;1.15 of the brief, I used an AI assistant (Claude, by Anthropic) throughout: to discuss the problem
and the approach, to write and debug most of the Python code, to generate the synthetic data, and to draft and edit parts of this
report. The assumptions, the engineering choices and the interpretation are mine, and every result was checked before it was used.
The brief says that an AI-generated result is not a verified engineering result; the most useful part of the interaction was the
list of things that were wrong at first and were changed after checking:""")
R.ul(["""The first cable layout swung two cables sideways inside the 300&nbsp;mm web, which is impossible; the profiles were redrawn so
the lateral swing happens inside the bottom bulb, and a clash check along the whole girder was added.""",
      """Stressing all four cables at 10 days gave 18.6&nbsp;MPa at the soffit against a limit of 17.3&nbsp;MPa; two-stage stressing was
adopted.""",
      """The first vibration model gave peak accelerations of 170&nbsp;mg because the lorry bounce frequencies overlapped the first mode
and the ambient forcing was too strong; both were recalibrated until the night-time RMS (0.6-0.8&nbsp;mg) and the traffic peaks
(10-25&nbsp;mg) were of the order reported for bridges of this size.""",
      """With 10-minute records and peak picking the scatter of f<sub>1</sub> was &plusmn;0.5&nbsp;%; 30-minute records with a fitted
spectrum reduced it to about 0.1&nbsp;%.""",
      """The AutoMAC check showed that the first sensor layout could not separate two pairs of modes; two accelerometers were added.""",
      """Identifying the commissioning test with the design-model shapes biased f<sub>2</sub> by 3.6&nbsp;%; the commissioning test is now
identified model-free (FDD).""",
      """The first synthetic inspection saw every hardened bearing and every 2&nbsp;% corrosion case, which made inspection look perfect;
detection was made to depend on severity, and leaking joints were also allowed on sound bridges.""",
      """The first prestress-loss model mistook corrosion cracks for flexural cracks and estimated 18&nbsp;% loss for the year-18 campaign;
the crack feature was split by crack type and the result checked against a direct physical inversion (section 14.3).""",
      """The novelty detector failed at first on the inspection features (singular covariance, because a healthy bridge never has rust
stains); a small floor on the covariance was added and its effect on the false-alarm rate checked.""",
      """The impact factor for 70R wheeled vehicles on a 33&nbsp;m span, the IRC:6 footway load for spans over 30&nbsp;m and the
three-lane reduction were checked against the code text rather than taken from the assistant's first answer."""])
n_ok = sum(c["ok"] for c in ver)
R.p(f"""Appendix B lists {len(ver)} independent checks of the computations by simpler methods (hand formulas, a separate influence-line
calculation, Courbon's method, beam-frequency formulas, recovery of the known parameters); {n_ok} of {len(ver)} pass their tolerances.
The span, the configuration, the prestressing system, the deterioration scenarios and values, the synthetic data and the
maintenance scheme were all developed for this report, as the brief requires for an individual problem (&sect;1.16).""")

# ====================================================================== 17
R.h1("Conclusions", "17")
R.ul([f"""A 33&nbsp;m span with six 1.80&nbsp;m post-tensioned M45 girders at 2.70&nbsp;m, four 12T15 cables each and two-stage stressing satisfies
IRC:6 / IRC:112 with the soffit uncracked under the rare combination (margin about 1&nbsp;MPa) and M<sub>Ed</sub>/M<sub>Rd</sub> = {d['uls']['G1']['MEd']/d['uls']['G1']['M']:.2f}.
The girder number was set by the limited construction depth rather than by material quantity.""",
      """A healthy baseline must be measured, not calculated. The design model was 7-10&nbsp;% off in frequency; after updating three
parameters on the commissioning tests the twin reproduced them within 0.2&nbsp;%, and one year of data gave the temperature and traffic
normalisation without which frequencies are not usable here (4&nbsp;% seasonal swing).""",
      """Vibration monitoring alone is poor at detecting the mechanisms that usually decide the life of a post-tensioned bridge,
prestress loss and corrosion. Levelling, embedded VW strain and bearing transducers are cheap and do it much better; NDT and inspection
are indispensable for corrosion.""",
      f"""Data fusion, not the algorithm, made the AI/ML useful: macro-F1 {sup['Vibration only']['macro_F1']:.2f} with vibration, {sup['All sources fused']['macro_F1']:.2f}
fused; the models' feature importances agree with the physics, and the Gaussian process gives prestress loss with an honest uncertainty.""",
      """Applied to the bridge at year 18, the assessment traced the defects to one cause (a leaking joint), located them, estimated the
prestress loss of the affected girder and produced a maintenance plan that removes the cause, repairs the damage, measures what the
monitoring can only infer, and re-baselines the bridge."""])

# ====================================================================== references
R.h1("References")
refs = [
    "IRC:5-2015. Standard Specifications and Code of Practice for Road Bridges, Section I: General Features of Design. Indian Roads Congress.",
    "IRC:6-2017. Standard Specifications and Code of Practice for Road Bridges, Section II: Loads and Load Combinations. Indian Roads Congress.",
    "IRC:112-2020. Code of Practice for Concrete Road Bridges. Indian Roads Congress.",
    "IRC:83 (Part II)-2018. Standard Specifications and Code of Practice for Road Bridges, Section IX: Bearings, Part II: Elastomeric Bearings.",
    "IRC:SP:51-2015. Guidelines for Load Testing of Bridges. Indian Roads Congress.",
    "IRC:SP:35-1990. Guidelines for Inspection and Maintenance of Bridges. Indian Roads Congress.",
    "IRC:SP:52-1999. Bridge Inspector's Reference Manual. Indian Roads Congress.",
    "IRC:SP:40-2019. Guidelines on Techniques for Strengthening and Rehabilitation of Bridges. Indian Roads Congress.",
    "IRC:SP:80-2008. Guidelines for Corrosion Prevention, Monitoring and Remedial Measures for Concrete Bridge Structures.",
    "IS 516 (Part 5/Sec 1):2018 and (Part 5/Sec 4):2020. Hardened Concrete - Methods of Test: Non-destructive testing (UPV; rebound hammer). BIS.",
    "ASTM C876-15. Standard Test Method for Corrosion Potentials of Uncoated Reinforcing Steel in Concrete.",
    "Hambly, E.C. (1991). Bridge Deck Behaviour, 2nd ed. E &amp; FN Spon, London.",
    "Brincker, R., Zhang, L., Andersen, P. (2001). Modal identification of output-only systems using frequency domain decomposition. Smart Materials and Structures, 10(3), 441-445.",
    "Peeters, B., De Roeck, G. (2001). One-year monitoring of the Z24-Bridge: environmental effects versus damage events. Earthquake Engineering and Structural Dynamics, 30(2), 149-171.",
    "Sohn, H. (2007). Effects of environmental and operational variability on structural health monitoring. Philosophical Transactions of the Royal Society A, 365, 539-560.",
    "Worden, K., Manson, G., Fieller, N.R.J. (2000). Damage detection using outlier analysis. Journal of Sound and Vibration, 229(3), 647-667.",
    "Farrar, C.R., Worden, K. (2013). Structural Health Monitoring: A Machine Learning Perspective. Wiley.",
    "Hamed, E., Frostig, Y. (2006). Natural frequencies of bonded and unbonded prestressed beams - prestress force effects. Journal of Sound and Vibration, 295, 28-39.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45, 5-32.",
    "Rasmussen, C.E., Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT Press.",
    "Woodward, R.J., Williams, F.W. (1988). Collapse of Ynys-y-Gwas bridge, West Glamorgan. Proceedings of the ICE, Part 1, 84, 635-669.",
    "Concrete Society (2010). Technical Report 72: Durable post-tensioned concrete structures. Camberley.",
]
R.raw("<ol class='small' style='font-size:9pt'>" + "".join(f"<li>{r}</li>" for r in refs) + "</ol>")

# ====================================================================== appendices
R.h1("Appendix A. Files and how to reproduce the results")
R.table(["File / folder", "Content"], [
    ["psc/inputs.py", "every adopted dimension, material, load and criterion"],
    ["psc/section.py, grillage.py, vehicles.py", "section properties, grillage FE (static, influence surfaces, modal), IRC vehicles and worst-position search"],
    ["psc/prestress.py, design.py", "cable geometry, losses, stage stresses, ULS, shear, bearings, girder-number study"],
    ["psc/twin.py, dynamics.py", "digital twin (state &rarr; responses), vibration simulation, FDD and modal-filter identification"],
    ["psc/condition.py, scenarios.py, features.py", "inspection and NDT models, deterioration scenarios, campaign generator, feature extraction"],
    ["psc/mlkit.py, maintenance.py", "ML methods (numpy/scipy) and the decision rules"],
    ["run_01_design.py ... run_05_case_study.py", "the five steps of the study, in order"],
    ["make_figures.py, make_figures_ml.py, build_report.py", "figures and this report"],
    ["verify.py", "independent checks (Appendix B)"],
    ["inspection_app/index.html", "digital visual inspection application (offline, single file)"],
    ["out/data/*.csv", "baseline year, scenario library, NDT surveys, visual inspection, 18-year monitoring record"],
], widths=[36, 64], small=True)
R.p("""Requirements: Python 3 with numpy, scipy and matplotlib only. Run the five <code>run_*.py</code> scripts in order, then
<code>verify.py</code>, <code>make_figures.py all</code> and <code>build_report.py</code>. The whole chain takes about an hour on a
16-thread laptop, most of it the 3600 vibration records of the library.""")
R.h1("Appendix B. Verification checks")
R.table(["Check", "Computed", "Independent", "Diff.", "Tol.", ""],
        [[c["check"] + (f" <span class='small'>({c['note']})</span>" if c["note"] else ""), f"{c['computed']:.4g} {c['unit']}",
          f"{c['hand']:.4g}", f"{100*c['rel_diff']:.1f} %", f"{100*c['tol']:.0f} %", "pass" if c["ok"] else "<b>fail</b>"] for c in ver],
        "Independent checks of the computed results", num_cols=(1, 2, 3, 4), small=True)
R.h1("Appendix C. Notation", nobreak=True)
R.table(["Symbol", "Meaning"], [
    ["P<sub>j</sub>, P<sub>0</sub>, P<sub>e</sub>", "jacking force, force after immediate losses, effective force after all losses"],
    ["&eta;", "ratio of actual to expected effective prestress"], ["f<sub>1</sub>...f<sub>4</sub>", "first bending, first torsion, first transverse, second bending frequency"],
    ["a<sub>RMS</sub>, a<sub>max</sub>", "RMS and peak acceleration of a record"], ["&delta;, &epsilon;", "deflection, strain"],
    ["MAC", "modal assurance criterion"], ["FDD", "frequency domain decomposition"], ["HCP", "half-cell potential (mV vs copper-copper sulphate)"],
    ["UPV", "ultrasonic pulse velocity"], ["VW", "vibrating-wire (strain gauge)"], ["CS", "condition state 1 (good) to 5 (critical)"],
    ["BCI", "bridge condition index, 0-100"], ["GPR", "Gaussian-process regression"], ["EWMA", "exponentially weighted moving average"],
], widths=[22, 78], small=True)

# ====================================================================== write + print (two passes for the contents page numbers)
import pathlib
out = pathlib.Path("report"); out.mkdir(exist_ok=True)
html_path = out / "B33_PSC_Bridge_Integrated_Study.html"
pdf_path = out / "B33_PSC_Bridge_Integrated_Study.pdf"
html_path.write_text(R.html(), encoding="utf-8")
print_pdf(html_path, pdf_path)
pages = heading_pages(pdf_path, R)
html_path.write_text(R.html(pages), encoding="utf-8")
print_pdf(html_path, pdf_path)
import pymupdf
print(f"report: {pdf_path} ({pymupdf.open(str(pdf_path)).page_count} pages), {R.fig_no} figures, {R.tab_no} tables; headings found {len(pages)}/{len(R.toc)}")
