# Second half of the report (chapters 10-17, references, appendices) and printing.
# Executed from build_report.py, which defines R, d, b, ml, cs, ver, sens and the helpers.
from psc import twin as TW

twn = TW.Twin(E_scale=tw["E_scale"], barrier=tw["barrier"], E_dyn=tw["E_dyn"])
H0 = TW.healthy_state()
f_h, _ = twn.modal(H0)


def resp(st):
    f_, _ = twn.modal(st); d_, e_ = twn.load_test(st); lev, vw = twn.long_term(st); br = twn.bearings(st)
    return f_, d_, e_, lev, vw, br


sup, nov = ml["supervised"], ml["novelty"]
SRC = list(FE.SOURCES.keys())
SH = {"PL": "Loss of prestress", "CD": "Concrete deterioration", "CR": "Flexural cracking", "CO": "Corrosion",
      "BR": "Bearing deterioration", "LS": "Local stiffness loss"}
SRC_NAME = {"Continuous monitoring (no closure)": "Continuous monitoring (no lane closure)"}
pl, brs, cdd = sens["PL"], sens["BR"], sens["CD"]

# ====================================================================== 10
R.h1("Deterioration Scenarios", "10")
R.p("""The type, location and size of the deterioration are left open in the problem, so I selected six mechanisms that are realistic
for a post-tensioned girder bridge in a wet coastal climate. For each mechanism I identified a cause on this particular bridge, chose a
range of values that can be defended, and set up a physical model for its effect on the structure. I kept the ranges wide, from barely detectable to clearly serious, because the AI/ML models have to learn where detection
begins.""")
R.h2("Mechanisms and their ranges", "10.1")
R.table(["Mechanism", "Cause on B33", "Model in the twin", "Range and reason"], [
    ["Loss of prestress (PL)", "creep, shrinkage and relaxation larger than predicted; grout voids and strand corrosion near anchorages; anchorage slip",
     "ratio &eta; of actual to expected <i>P</i><sub>e</sub>, in one girder (60 % of cases) or in all girders (40 %); no change of uncracked stiffness; cracking and reopening of cracks as described in Section 10.2",
     "3 to 35 % of <i>P</i><sub>e</sub>. The uncertainty of the predicted long-term loss alone is about &plusmn;30 % of 12.7 %, i.e. 4 % of <i>P</i><sub>e</sub>; the upper end covers badly corroded tendons"],
    ["Concrete deterioration (CD)", "poorly compacted or chemically attacked concrete in one girder",
     "reduction d<sub>E</sub> of the modulus over a 6 to 15 m zone or the whole girder; strength reduced by 0.9d<sub>E</sub>; voids affecting UPV; faster carbonation",
     "d<sub>E</sub> = 5 to 30 %; beyond 30 % the girder would show obvious distress"],
    ["Flexural cracking (CR)", "a vehicle heavier than the design load",
     "loss d<sub>s</sub> of flexural stiffness over 4 to 10 m near mid-span in a load test; in ambient vibration only 20 to 50 % of it, since cracks in a prestressed girder close again; residual crack width 0.05 to 0.35 mm",
     "d<sub>s</sub> = 15 to 50 % locally, i.e. a few per cent of the stiffness of the girder"],
    ["Corrosion (CO)", "water from a leaking joint or a short drain spout on low-cover concrete of an outer girder; carbonation reaches the links, then strands in poorly grouted ducts",
     "loss c of strand area over 2 to 8 m at a joint end or under a spout; local prestress loss equal to c; stiffness loss 0.25c from spalling; wet concrete, low cover, more negative HCP and lower resistivity; rust and spalling visible only after enough rust has formed",
     "c = 2 to 25 %; 80 % of cases on G1 or G6, which carry the drains"],
    ["Bearing deterioration (BR)", "ageing of the elastomer and debris from a leaking joint packed around the bearing",
     "shear stiffness of the bearing multiplied by 5 to 500 at one or both ends, Eq. ([[eq:bearing]])",
     "a bearing locked by debris behaves as several hundred times stiffer in shear; 60 % of cases have a visibly leaking joint"],
    ["Local stiffness loss (LS)", "a tall vehicle on the service road striking a fascia girder; internal void or delamination",
     "loss of 15 to 60 % of the flexural stiffness over 1 to 3 m, anywhere between 5 and 28 m",
     "local damage that removes a large part of one section but little of the stiffness of the span; visible in 75 % of cases on outer girders and 40 % on inner girders"],
], "Deterioration mechanisms, their models and ranges", widths=[13, 22, 40, 25], small=True)
R.p("""The healthy cases also vary: the age of the bridge (5 to 30 years), the depth of carbonation, which grows with the square root of age,
the temperature, the moisture of the deck and the traffic. A campaign contains one mechanism (60&nbsp;% of campaigns), two mechanisms at the
same time (15&nbsp;%) or none (25&nbsp;%).""")
R.h2("Physical models of the mechanisms", "10.2")
R.p("""<b>Loss of prestress.</b> In an uncracked girder with bonded tendons the axial prestressing force does not reduce the flexural
stiffness (Hamed and Frostig, 2006), so a loss of prestress does not change the natural frequencies or the response to a load test as long
as the girder stays uncracked. It does change the camber and the strain in the concrete. For a loss &Delta;P, the change of level at
mid-span and the change of strain at the soffit are""")
R.eq(r"\Delta\delta \approx \dfrac{\Delta P}{P_e}\,\delta_{P_e}\,(1 + \chi\varphi), \qquad \Delta\varepsilon_b \approx \dfrac{\Delta P}{E_{cm}}\left(\dfrac{1}{A} + \dfrac{e}{Z_b}\right)(1 + \chi\varphi)", "plsens")
R.p(f"""where &delta;<sub>Pe</sub> = {d['deflection']['camber_Pe_elastic_mm']:.0f}&nbsp;mm is the elastic camber due to P<sub>e</sub>
(Section 4.8), and (1 + &chi;&phi;) = 1.8 allows for creep under a slowly developing loss. This gives about 1.0&nbsp;mm of sag and
12&nbsp;&mu;&epsilon; of tensile strain change per 1&nbsp;% loss of prestress. The girder cracks when the soffit stress under heavy (not extreme)
traffic, taken as 0.8 times the design live load together with 0.6 times the reverse temperature difference, exceeds f<sub>ctm</sub>. Once
cracked, a section reopens in a later load test as soon as the soffit is decompressed again, and its stiffness above the decompression moment
M<sub>dec</sub> follows a Branson-type expression,""")
R.eq(r"I_{\mathrm{eff}} = \left(\dfrac{M_{dec}}{M_a}\right)^3 I_g + \left[1 - \left(\dfrac{M_{dec}}{M_a}\right)^3\right] I_{cr}, \qquad \dfrac{I_{cr}}{I_g} = 0.35", "branson")
R.p("""<b>Bearing deterioration.</b> In a two-dimensional grillage a horizontal restraint at the level of the bearings cannot be modelled
directly. If both ends of a girder are restrained horizontally at a distance e below the neutral axis, a rotation of the ends stretches the
soffit and the restraint produces end moments. This was modelled as a rotational spring that couples the two end rotations of each girder,""")
R.eq(r"k_r = k_s\,e^2, \qquad k_s = \left(\dfrac{1}{k_{b,A}} + \dfrac{1}{k_{sub,A}} + \dfrac{L}{E A} + \dfrac{1}{k_{b,B}} + \dfrac{1}{k_{sub,B}}\right)^{-1}", "bearing")
R.p("""where k<sub>b</sub> is the shear stiffness of a bearing and k<sub>sub</sub> the horizontal stiffness of the pier or abutment. The
same series stiffness gives the movement of each bearing per degree of temperature. One consequence is important for monitoring: if only one
end is restrained, k<sub>s</sub> remains small (the other end is still free), the frequencies hardly change, but the free end takes the whole
movement, which the bearing transducers see immediately.""")
R.p("""<b>Concrete deterioration</b> reduces the modulus in a zone of one girder. In the NDT model the pulse velocity changes with the square
root of the dynamic modulus (V &prop; &radic;E<sub>d</sub>), so a 20&nbsp;% loss of modulus lowers the UPV by about 10&nbsp;%, while the rebound
number follows the reduced strength of the surface concrete. <b>Corrosion</b> combines a local loss of strand area, a local loss of prestress
and some spalling with the NDT signature described in Table [[tab:ndt]]. <b>Cracking</b> and <b>local stiffness loss</b> reduce the flexural
stiffness of a few elements of one girder.""")
R.h2("Effect on the measured quantities", "10.3")
R.p("""Before any random data were generated, each mechanism was increased from zero to the upper end of its range on girder G1 (or on all
girders where that is the natural case), with all noise switched off. Figure [[fig:f13_sensitivity]] shows what each measurement can and
cannot see before any machine learning is involved.""")
R.fig("f13_sensitivity", "Response of the twin (without noise) to each mechanism, from healthy (0) to the upper end of its range (1)")
R.ul([f"""<b>Loss of prestress</b> leaves the frequencies and the load-test deflection unchanged until the girders crack, at roughly
25&nbsp;% loss; after that the load-test deflection increases sharply (by {max(pl['d']):.0f}&nbsp;% at 35&nbsp;% loss). The level change
({pl['lev'][3]:.1f}&nbsp;mm at about 18&nbsp;% loss) and the VW strain ({pl['vw'][3]:.0f}&nbsp;&mu;&epsilon;) respond from the start.""",
      f"""<b>Bearing restraint</b> <i>increases</i> the frequencies, by up to {max(brs['f1']):.1f}&nbsp;% for f<sub>1</sub> when both ends are
locked, because a horizontal force below the neutral axis resists the rotation of the ends. A rise in frequency therefore does not mean
that the bridge is healthy.""",
      f"""<b>Concrete deterioration</b> over a whole girder lowers f<sub>1</sub> by up to {abs(min(cdd['f1'])):.1f}&nbsp;% and increases the
load-test deflection by {max(cdd['d']):.0f}&nbsp;%; a mechanism that changes the stiffness of a large volume is the one that global vibration
detects best.""",
      """<b>Cracking, corrosion and local stiffness loss</b> change the global quantities by a fraction of a per cent to a few per cent. They are
local, and the six-girder deck redistributes the load around them."""])
R.p(f"""These changes have to be compared with the effect of temperature: a day 10&nbsp;&deg;C warmer lowers f<sub>1</sub> by about
{abs(10 * nb['f1']['per_degC_pct']):.1f}&nbsp;%, which is as much as a 20&nbsp;% loss of modulus in one girder. As the problem statement also
warns, a change in frequency cannot be read directly as a particular type of damage. Figure [[fig:f13_sensitivity]] shows the reason: four
mechanisms lower the frequency, one raises it, one leaves it unchanged, and temperature can move it either way.""")

# ====================================================================== 11
R.h1("Synthetic Data", "11")
R.h2("Generation of a measurement campaign", "11.1")
R.p("""I produced the synthetic data with the updated twin of Chapter 5 and not with the hidden 'real' model, because in practice only the
model of the bridge is available for simulation. Every synthetic campaign goes through the same steps as a real
measurement campaign (Table [[tab:generation]]).""")
R.table(["Step", "What is generated", "Variability and noise"], [
    ["Scenario", "0, 1 or 2 mechanisms of random size and location (Chapter 10)", "25 % healthy, 60 % one mechanism, 15 % two mechanisms"],
    ["Environment", "effective bridge temperature from a seasonal and daily model of the site; deck moisture in the monsoon", "16 to 41 &deg;C; mass +0.6 % in the monsoon, &plusmn;0.2 % at random"],
    ["Vibration", "response of 8 modes to moving lorries (bounce 1.3 to 2.6 Hz, amplitude 2 to 6 %), cars and broadband ambient forces; 12 channels, 30 minutes, 100 Hz", "3 to 60 vehicles per 10 minutes at night, 150 to 420 by day; damping 1.5 &plusmn; 0.2 %; sensor noise 2 &micro;g/&radic;Hz"],
    ["Identification", "complete processing of Chapter 9 applied to each record", "frequencies, MAC, damping, <i>a</i><sub>RMS</sub> and <i>a</i><sub>max</sub> as identified, not as simulated"],
    ["Load test", "four 300 kN lorries; deflection and strain of all six girders", "lorry weight &plusmn;1 %; 0.05 mm; 3 &mu;&epsilon;"],
    ["Geometry and VW strain", "change beyond the healthy creep prediction", "creep model error 2.5 mm and 25 &mu;&epsilon; (common to all girders); instrument error 0.8 mm and 3 &mu;&epsilon;"],
    ["Bearings", "movement per &deg;C at both ends", "&plusmn;0.012 mm/&deg;C"],
    ["NDT", "30 test areas of span S2: rebound, UPV, cover, HCP, resistivity, carbonation, chloride", "scatter of strength, cover and carbonation rate in the structure; reading errors of each test"],
    ["Visual inspection", "observations with size and condition state", "hairline cracks often missed (probability of detection 1 &minus; e<sup>&minus;w/0.06</sup>); condition state wrong by one for 15 % of entries; early defects often not visible"],
], "Steps in the generation of one synthetic campaign", tag="generation", widths=[15, 47, 38], small=True)
R.p(f"""In total {ml['n']} campaigns were generated: {ml['counts']['H']} healthy, about {int(np.mean([ml['counts'][m] for m in SC.MECHS]))} for each
single mechanism and {ml['counts']['pair']} with two mechanisms. Each campaign contains 52 raw quantities and its true labels, and the data are
stored in <code>out/data/library.csv</code>. Some examples are given below.""")
R.h2("Examples of the synthetic data", "11.2")
st_list = [("Healthy", H0)]
for loss in (0.20, 0.27, 0.33):
    st = TW.healthy_state(); st["eta"][:] = 1 - loss; st_list.append((f"Prestress loss {loss * 100:.0f} %", st))
rows = []
for name, st in st_list:
    f_, d_, e_, lev, vw, brr = resp(st)
    rows.append([name, f"{d_[0]:.2f}", f"{d_[2]:.2f}", f"{e_[0]:.0f}", f"{lev.mean():+.1f}", f"{vw.mean():+.0f}", f"{f_[0]:.3f}"])
R.table(["Condition (all girders)", "Deflection G1 (mm)", "Deflection G3 (mm)", "Strain G1 (&mu;&epsilon;)", "Level change (mm)",
         "VW change (&mu;&epsilon;)", "<i>f</i><sub>1</sub> (Hz)"], rows,
        "Synthetic load-test, geometry and frequency data for increasing loss of prestress (without noise, 25 &deg;C)", num_cols=(1, 2, 3, 4, 5, 6),
        tag="exdisp", small=True)
R.p("""Table [[tab:exdisp]] shows the result of Chapter 10 in numbers: up to 20&nbsp;% loss the load test and the frequency show nothing,
whereas the level and the VW strain have already changed by about 21&nbsp;mm and 250&nbsp;&mu;&epsilon;. Once the girders crack (27&nbsp;%) the
load-test deflection increases sharply.""")
ndt_rows = [r for r in csv.DictReader(open("out/data/case_ndt_survey.csv")) if r["span"] == "S2" and r["girder"] == "G1"]
rows = [[r["location"], f"{float(r['rebound']):.0f}", f"{float(r['upv']):.2f}", f"{float(r['cover']):.0f}", f"{float(r['carbonation']):.0f}",
         f"{float(r['hcp']):.0f}", f"{float(r['resistivity']):.0f}", r["cluster"]] for r in ndt_rows]
R.table(["Location", "Rebound number", "UPV (km/s)", "Cover (mm)", "Carbonation (mm)", "HCP (mV)", "Resistivity (k&Omega;cm)", "Cluster (Ch. 14)"], rows,
        "Synthetic NDT data along girder G1 of span S2 at an age of 18 years", num_cols=(1, 2, 3, 4, 5, 6), tag="exndt", small=True)
R.p("""Along G1 the readings change towards the joint end (Z4 and Z5): the resistivity drops and the potentials become more negative where
the concrete is wet, and the cover is lower than specified with the carbonation front close to it. UPV and rebound are hardly affected,
because the concrete itself is sound; corrosion is a problem of the steel and its environment, not of the quality of the concrete.""")
st = TW.healthy_state(); st["kh_mult"][:, :] = 100
f_br, *_ = resp(st)
st = TW.healthy_state(); st["conc"].append((0, 0, P.L, 0.2))
f_cd, *_ = resp(st)
R.table(["Mode", "Healthy (Hz)", "Concrete deterioration, G1, &minus;20 % E (Hz)", "Bearings restrained at both ends, &times;100 (Hz)", "Temperature +10 &deg;C (Hz)"],
        [[n, f"{f_h[k]:.3f}", f"{f_cd[k]:.3f} ({100 * (f_cd[k] / f_h[k] - 1):+.1f} %)", f"{f_br[k]:.3f} ({100 * (f_br[k] / f_h[k] - 1):+.1f} %)",
          f"{f_h[k] * (1 + 10 * nb['f1']['per_degC_pct'] / 100):.3f} ({10 * nb['f1']['per_degC_pct']:+.1f} %)"]
         for n, k in (("<i>f</i><sub>1</sub>, bending", 0), ("<i>f</i><sub>2</sub>, torsion", 1), ("<i>f</i><sub>3</sub>, transverse", 2))],
        "Synthetic vibration data for two deterioration cases and a change of temperature", num_cols=(1, 2, 3, 4), small=True)
R.p("""The example in the problem statement changes f<sub>1</sub> from 3.20 to 3.05&nbsp;Hz, i.e. by &minus;4.7&nbsp;%. Since the frequency
varies with &radic;EI, a change of that size on B33 would need a loss of about 9&nbsp;% of the bending stiffness of the whole deck, more than any
single mechanism in the chosen ranges produces, or simply a temperature rise of about 30&nbsp;&deg;C. The frequency alone cannot tell these
apart. The pattern across the modes carries more information: deterioration of G1 lowers f<sub>1</sub> more than f<sub>2</sub> and
f<sub>3</sub>, bearing restraint raises f<sub>1</sub> and f<sub>2</sub> but hardly f<sub>3</sub>, and temperature changes all three by nearly
the same percentage. For this reason all four tracked modes are kept as features.""")

# ====================================================================== 12
R.h1("Feature Extraction", "12")
R.p("""A useful feature should have a physical meaning, respond to at least one mechanism and be stable on the healthy bridge. Each
feature used here is expressed as a change from the structure's <i>own</i> healthy baseline: the residual of a frequency after the
temperature and traffic model of Eq. ([[eq:norm]]), the load-test deflection relative to the commissioning test after correction to
25&nbsp;&deg;C, the levels and VW strains relative to the creep prediction, and the bearing movement relative to &alpha;L/2. This is
essential because the models are trained on twin data and then applied to the real bridge; an error of 0.5&nbsp;% in the twin's absolute
frequency would otherwise look like damage.""")
mean = {"VIB": "global stiffness, mass and boundary conditions; MAC and edge asymmetry show which side of the deck has changed",
        "LT": "static stiffness of each girder under a known load; the asymmetry locates a soft girder",
        "GEO": "permanent deformation: loss of precompression (camber, soffit strain) and creep of a softened girder",
        "BRG": "freedom of the bearings; restraint at one end doubles the movement at the other",
        "NDT": "condition of the concrete (UPV, rebound) and of the steel (HCP, resistivity, cover against carbonation)",
        "VI": "what the inspector sees: cracks, spalling, rust, bearing and joint distress"}
gname = {"VIB": "Vibration", "LT": "Load test", "GEO": "Geometry, VW strain", "BRG": "Bearings", "NDT": "NDT", "VI": "Visual inspection"}
rows = [[gname[g], ", ".join(FE.LABELS[n] for n in names), mean[g]] for g, names in FE.GROUPS.items()]
R.table(["Group", "Features", "Physical meaning"], rows, f"The {sum(len(v) for v in FE.GROUPS.values())} features used by the AI/ML models",
        widths=[13, 50, 37], small=True)
R.p("""I started from the feature vector suggested in the problem statement, x = [f<sub>1</sub>, f<sub>2</sub>, a<sub>RMS</sub>, &delta;<sub>max</sub>,
&epsilon;<sub>max</sub>, UPV, rebound number], and changed two of its entries after looking at the data.
a<sub>RMS</sub> and a<sub>max</sub> depend much more on traffic than on condition (the ratio between day and night is about 2), so they are used
to <i>normalise</i> the frequencies rather than as condition features. &delta;<sub>max</sub> was replaced by the change of the mean deflection
together with the G1&ndash;G6 asymmetry, which separate a global from a local change. Figure [[fig:f14_effect_size]] shows, before any
classifier is trained, how strongly each feature reacts to each mechanism, measured as the shift of its mean in units of the healthy standard
deviation.""")
R.fig("f14_effect_size", "Effect of each mechanism on each feature: shift of the mean from the healthy mean, in healthy standard deviations (campaigns in the upper half of each severity range). Blue: decrease, red: increase.", scale=0.92)
R.p("""Reading Figure [[fig:f14_effect_size]] column by column already gives most of the rules an engineer would write: loss of prestress moves
the levels and the VW strain; bearing restraint changes the bearing movement ratio, raises f<sub>1</sub> and often comes with a leaking joint;
corrosion changes the potentials, the resistivity and the cover relative to carbonation; concrete deterioration lowers UPV and rebound and,
over a whole girder, the frequencies; cracking and local damage appear in the load test and the inspection. The machine-learning models are
useful because they combine these weak and overlapping signals consistently, attach a measure of confidence and estimate the size of the
problem. Figure [[fig:f15_distributions]] shows the distributions of six of the features.""")
R.fig("f15_distributions", "Distributions of six features for healthy (H) and single-mechanism campaigns (boxes: quartiles; whiskers: 1.5 times the interquartile range)")

# ====================================================================== 13
R.h1("AI/ML-Assisted Condition Assessment", "13")
R.p(f"""The library was split into a training set ({ml['n_train']} campaigns) and a test set ({ml['n_test']} campaigns), stratified by class;
all results below are for the test set. Each method is introduced together with the question it answers.""")
R.h2("Is the bridge different from its healthy self? (novelty detection)", "13.1")
R.p("""The novelty detector is trained on healthy campaigns only. This matches the situation on a real bridge, which has a healthy baseline
but no labelled examples of its own damage. For a standardised feature vector x the squared Mahalanobis distance from the healthy data is""")
R.eq(r"D^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu}_H)^{\mathrm{T}}\,\mathbf{\Sigma}_H^{-1}\,(\mathbf{x} - \boldsymbol{\mu}_H)", "maha")
R.p("""where &mu;<sub>H</sub> and &Sigma;<sub>H</sub> are the mean and the covariance of the healthy training campaigns. The covariance was shrunk
slightly towards its diagonal, and a small value was added to the diagonal so that features which are always zero on a healthy bridge (rust
stains, for example) do not make the matrix singular. A campaign is flagged when D&sup2; exceeds the 99th percentile of the healthy training
values, i.e. the designed false-alarm rate is 1&nbsp;%.""")
R.table(["Data source", "Healthy (false alarm)"] + [SH[m] for m in SC.MECHS],
        [[SRC_NAME.get(s, s), pct(nov[s]["false alarm"])] + [pct(nov[s][m]) for m in SC.MECHS] for s in SRC],
        "Share of test campaigns flagged by the novelty detector", num_cols=tuple(range(1, 8)), tag="novelty", small=True)
R.fig("f16_novelty", "Novelty detection: detection rate by data source, and probability of detection against severity")
R.p(f"""With vibration data alone the detector finds most cases of concrete deterioration ({pct(nov['Vibration only']['CD'])}) and local
stiffness loss ({pct(nov['Vibration only']['LS'])}), fewer bearing cases ({pct(nov['Vibration only']['BR'])}, essentially those restrained at both
ends), and almost no cases of prestress loss ({pct(nov['Vibration only']['PL'])}) or corrosion ({pct(nov['Vibration only']['CO'])}). This confirms
Figure [[fig:f13_sensitivity]] in the presence of noise and environmental variation: vibration monitoring alone is not suitable for the two
mechanisms that most often govern the life of a post-tensioned bridge. Adding levelling, VW strain and bearing movement, which can all be
measured without closing the bridge, raises the detection of prestress loss to {pct(nov['Continuous monitoring (no closure)']['PL'])} and of
bearing problems to {pct(nov['Continuous monitoring (no closure)']['BR'])}. With all sources combined, every mechanism is detected in
{pct(min(nov['All sources fused'][m] for m in SC.MECHS))} or more of the cases at a false-alarm rate of {pct(nov['All sources fused']['false alarm'])};
the cases that are missed are the smallest ones.""")
R.h2("Which mechanism? (supervised classification and data fusion)", "13.2")
R.p("""For each mechanism a random forest (Breiman, 2001) of 100 classification trees was trained to answer the question 'is this mechanism
present?', so that two mechanisms can be reported at the same time. Each tree is grown on a bootstrap sample of the training data; at each node it
tries &radic;p randomly chosen features and selects the split that reduces the Gini impurity the most,""")
R.eq(r"G = 1 - \sum_{k} p_k^2", "gini")
R.p("""where p<sub>k</sub> is the share of class k in the node. The probability given by the forest is the average over its trees. The same
model was trained on each data source separately and on all of them together; this is the data-fusion study. The models are compared with the
area under the ROC curve (AUC) and with the F1 score at a threshold of 0.5,""")
R.eq(r"F_1 = \dfrac{2\,\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}", "f1")
R.table(["Data source"] + [SH[m] for m in SC.MECHS] + ["Macro-F1"],
        [[SRC_NAME.get(s, s)] + [f"{sup[s][m]['AUC']:.2f} / {sup[s][m]['F1']:.2f}" for m in SC.MECHS] + [f"<b>{sup[s]['macro_F1']:.2f}</b>"] for s in SRC],
        "ROC-AUC / F1 for each mechanism and data source (test set)", num_cols=tuple(range(1, 8)), tag="supervised", small=True)
R.fig("f17_supervised", "ROC-AUC for each mechanism and data source, and macro-averaged F1 score")
lr = ml["logistic_fused"]
R.p(f"""The macro-F1 increases from {sup['Vibration only']['macro_F1']:.2f} with vibration alone to {sup['Static + geometry + bearings']['macro_F1']:.2f}
with static, geometric and bearing data, {sup['Inspection + NDT']['macro_F1']:.2f} with inspection and NDT, and
{sup['All sources fused']['macro_F1']:.2f} with all sources combined. No single source is sufficient, and the sources complement each other in a
way that follows the physics: NDT and inspection see corrosion and surface damage, levelling and VW strain see the prestress, the bearing
transducers see the bearings, and the load test sees cracking. A logistic regression on the combined features reaches AUC values of
{min(v['AUC'] for v in lr.values()):.2f} to {max(v['AUC'] for v in lr.values()):.2f}, close to those of the random forest, so the choice of features
mattered far more than the choice of algorithm.""")
R.fig("f18_confusion", "Most likely mechanism for healthy and single-mechanism test campaigns: vibration data only and all sources combined")
R.p("""The confusion matrices of Figure [[fig:f18_confusion]] show where the errors are. With vibration alone, prestress loss and corrosion are
mostly classified as healthy, and cracking, local stiffness loss and concrete deterioration are confused with each other because all three
lower the frequencies slightly and differ mainly in where and by how much. With all sources, the remaining errors are early prestress loss and
the smallest cases of cracking and concrete deterioration, which are hard to see by any means.""")
R.h2("How much prestress has been lost? (regression with uncertainty)", "13.3")
g = ml["pl_regression"]
R.p("""The size of a loss of prestress decides between continued monitoring and strengthening, so an estimate without its uncertainty is of
little use. Gaussian-process regression (Rasmussen and Williams, 2006) gives both. With a squared-exponential covariance function with one length
scale per input, the prediction at a new point x<sub>*</sub> has the mean and variance""")
R.eq(r"\mu_* = \mathbf{k}_*^{\mathrm{T}}\left(\mathbf{K} + \sigma_n^2\mathbf{I}\right)^{-1}\mathbf{y}, \qquad \sigma_*^2 = k_{**} - \mathbf{k}_*^{\mathrm{T}}\left(\mathbf{K} + \sigma_n^2\mathbf{I}\right)^{-1}\mathbf{k}_*", "gpr")
R.p(f"""where K is the covariance matrix of the training inputs, k<sub>*</sub> the covariances between the new point and the training points,
and &sigma;<sub>n</sub> the noise. The hyper-parameters were found by maximising the marginal likelihood. The inputs are the largest sag of a
girder, the mean level change, the largest and mean VW strain changes, the change of load-test deflection and the width of flexural cracks. On
the test set the mean absolute error is <b>{g['gpr']['MAE']:.1f}&nbsp;%</b> of P<sub>e</sub> (R&sup2; = {g['gpr']['R2']:.2f}) and the 95&nbsp;% band
contains the true value in {pct(g['coverage95'])} of the cases, so the stated uncertainty is realistic (Figure [[fig:f20_pl_regression]]). A
random forest gives a similar error ({g['rf']['MAE']:.1f}&nbsp;%). A purely physical inversion with the sensitivities of Eq. ([[eq:plsens]]) on the
worst instrumented girder is less accurate (MAE {g['levelling_only']['MAE']:.1f}&nbsp;%), mainly because it cannot see losses in G2 and G5, which
have no gauges, and because it is disturbed by a second mechanism in the two-mechanism cases. The shortest length scales of the fitted process
belong to the {' and the '.join(FE.LABELS[k] for k in sorted(g['length_scales'], key=g['length_scales'].get)[:2])}.""")
R.fig("f20_pl_regression", "Estimated against true extra loss of prestress for the test campaigns: Gaussian-process regression and physical inversion")
loc = ml["localisation"]
R.h2("Where? (localisation)", "13.4")
R.p(f"""A random forest using the deflections and strains of each girder in the load test, the levels, the VW strains and the asymmetry of the
first mode shape identifies the affected girder in <b>{pct(loc['accuracy'])}</b> of the single-mechanism test cases (prestress loss
{pct(loc['by_mechanism']['PL'])}, concrete deterioration {pct(loc['by_mechanism']['CD'])}, cracking {pct(loc['by_mechanism']['CR'])}, corrosion
{pct(loc['by_mechanism']['CO'])}, local stiffness loss {pct(loc['by_mechanism']['LS'])}). Corrosion is located worst because its effect on the global
response is small and often near a support, where the load test is not sensitive; for corrosion the location comes from the NDT grid instead
(Chapter 14).""")
R.h2("Do the models use the physics?", "13.5")
R.p("""The importance of a feature was measured by permutation: the values of the feature in the test set were shuffled and the resulting drop in
AUC was recorded. The importances (Figure [[fig:f19_importance]]) agree with the effect sizes of Figure [[fig:f14_effect_size]] and with the
mechanisms of Chapter 10. The model for prestress loss relies on the VW strain and the levels, the bearing model on the bearing movement ratio,
the corrosion model on the resistivity and the potentials, and the model for concrete deterioration on the UPV. I treated this agreement as a
necessary condition for trusting the models: a model that detected corrosion from the damping of the first mode, for example, would have learnt a
peculiarity of the simulation rather than anything about the bridge.""")
R.fig("f19_importance", "Permutation importance of the five most important features for each mechanism (all sources combined)")
R.h2("Limitations", "13.6")
R.ul(["""All data are synthetic. The scores show how well the method works <i>if the twin and the noise models are right</i>. On a real bridge they
would be lower, and the models would have to be recalibrated with the bridge's own data.""",
      """The twin is a linear grillage. Loss of bond, behaviour of crack faces, friction in the bearings and soil-structure interaction are
represented only by simple rules.""",
      """The labels in the library are exact. In practice the condition at the time of a past campaign is often unknown, which is why novelty
detection against the healthy baseline, and not the supervised models, is the first step of the assessment.""",
      """Only six mechanisms were considered. A different mechanism, such as fatigue of a strand or scour of a pier, would appear as 'novel but
not classified', which is the appropriate result for a case the models were not trained on."""])

# ====================================================================== 14
R.h1("Assessment of B33 at an Age of 18 Years", "14")
cm = cs["ml"]; asm = cs["assessment"]; tr = cs["trends"]
R.p("""This chapter applies everything above to the bridge itself. The 'real' bridge (the hidden model of Chapter 5) was aged for 18 years
with a deterioration history that was not used by the assessment. Monthly monitoring records (night-time vibration, bearing movement, VW strain)
and yearly levels were generated for the whole period, and in year 18 a principal inspection with NDT of all spans and a load test were carried
out. The assessment uses only these measurements, the baseline of Chapter 5 and the models of Chapter 13.""")
R.h2("Monitoring record", "14.1")
R.p("""Each monitored quantity x<sub>t</sub> was followed with an exponentially weighted moving average,""")
R.eq(r"z_t = \lambda\,x_t + (1 - \lambda)\,z_{t-1}, \qquad \mathrm{limits:}\ \ \mu_0 \pm L\,\sigma_0\sqrt{\dfrac{\lambda}{2 - \lambda}}", "ewma")
R.p("""with &lambda; = 0.2 and L = 3, where &mu;<sub>0</sub> and &sigma;<sub>0</sub> are the mean and standard deviation of the first two years. The
long-term slope over the last three years was estimated with the Theil&ndash;Sen estimator (the median of all pairwise slopes), which is not affected
by a few outliers.""")
R.fig("f21_case_trends", "Eighteen years of monitoring of span S2: monthly values and EWMA with &plusmn;3&sigma; limits from the first two years. The grey bands mark the hidden onset of the changes and the dash-dot lines the first alarm.")
rows = [[k, (f"{v['first_alarm_year']:.1f}" if v['first_alarm_year'] is not None else "no alarm"), f"{v['slope_last3y_per_year']:+.3f}"]
        for k, v in tr.items()]
R.table(["Monitored quantity", "First EWMA alarm (year)", "Theil&ndash;Sen slope over the last 3 years (per year)"], rows,
        "Trend identification on the monitoring record", num_cols=(1, 2), full=False)
R.p(f"""The bearing transducers gave the first and clearest signal, in year {tr['bearing B movement']['first_alarm_year']:.1f}: the movement at end B
fell towards zero while that at end A increased, which is the signature of a bearing being locked at one end while the other end takes over the
movement. The f<sub>1</sub> residual also raised an alarm, in year {tr['f1 residual']['first_alarm_year']:.1f}, but <i>upwards</i>: the span had become
stiffer. On its own this could be mistaken for a sign of good health or for an error in the temperature model; next to the bearing record it is
the restraint described in Chapter 10. The level of G1 dropped (alarm in year {tr['level G1']['first_alarm_year']:.1f}) and its VW gauge drifted towards
tension (year {tr['VW strain G1']['first_alarm_year']:.1f}), which is the pattern of lost precompression in one girder. All alarms came one to two years
after the hidden onsets, which is about what EWMA charts with these limits can achieve with monthly data. With vibration data alone the record
would only have said that the bridge had become stiffer.""")
R.h2("Principal inspection and NDT at year 18", "14.2")
obs = json.load(open("inspection_app/seed_observations.json"))
key = sorted([o for o in obs if o["cs"] >= 3], key=lambda o: -o["cs"])[:14]
R.table(["Span", "Component", "Location", "Observation", "Size", "CS", "Note"],
        [[o["span"], o["component"], o["location"], o["observation"], o["size"], o["cs"], o["note"]] for o in key],
        "Main observations of the year-18 inspection (condition state 3 and worse)", num_cols=(5,), small=True)
gr = cs["decision"]["groups"]
names_g = {"G": "girders", "EJ": "expansion joints", "BRG": "bearings", "DR": "drainage", "PC": "pier caps", "WC": "wearing coat",
           "DS": "deck slab", "CB": "crash barriers", "AB": "abutments", "FP": "footpaths", "PR": "piers", "CG": "cross girders"}
R.p(f"""The bridge condition index is <b>{cs['decision']['bci']:.0f}</b> out of 100. The worst component groups are the
{', '.join(f'{names_g[k]} (CS{v})' for k, v in sorted(gr.items(), key=lambda kv: -kv[1])[:4])}. All serious findings lie within a few metres of
the expansion joint over pier P2: the torn seal, silt around the S2 bearings at end B and the S3 bearings at end A, leakage and spalling on the pier
cap, a short drain spout, and rust-stained splitting cracks and spalling at the anchorage end of G1. The inspector rated two of these cracks, 0.39
and 0.45&nbsp;mm wide along the duct line in zones Z4 and Z5, as CS5, although by the width rule they would be CS4. The rating was kept in the
record, because rust coming out of a crack along a tendon duct next to an anchorage is exactly the warning sign that preceded the collapse of the
Ynys-y-Gwas bridge (Woodward and Williams, 1988), but the category of the bridge was decided only after the structural check of Section 14.4.""")
cl = cs["ndt_clusters"]
hot = [r["location"] for r in csv.DictReader(open("out/data/case_ndt_survey.csv")) if float(r["hcp"]) < -350]
risk = [c for c in cl["clusters"] if c["name"] != "sound / dry"]
sound = [c for c in cl["clusters"] if c["name"] == "sound / dry"]
R.p(f"""The 90 NDT test areas of the three spans were grouped with a Gaussian mixture model fitted by the EM algorithm to the standardised UPV,
rebound number, half-cell potential, logarithm of resistivity and cover minus carbonation depth. The Bayesian information criterion (BIC) was
lowest for {cl['k_best']} components, so two groups were kept: {sound[0]['n']} areas of sound, dry concrete and {risk[0]['n']} areas of wet concrete
with a risk of corrosion (Figure [[fig:f22_case_ndt]]). All {risk[0]['n']} of the latter lie at girder ends next to the joints over the two piers
({', '.join(risk[0]['members'])}). Within this group the ASTM C876 limits separate the areas further: only {' and '.join(hot)} have potentials below
&minus;350&nbsp;mV together with a resistivity near 10&nbsp;k&Omega;cm, i.e. probable active corrosion, while the others are wet but still passive.
k-means clustering with the same number of groups agrees with the mixture model for {pct(cl['kmeans_agreement'])} of the pairs of test areas.""")
R.fig("f22_case_ndt", "Year-18 NDT survey of all girders grouped by a Gaussian mixture model, and half-cell potential against resistivity with the limits of ASTM C876")
R.p("""The clustering is useful because it is unsupervised: it knows nothing about the joints, yet it separates the wet end zones next to both piers
from the rest of the bridge, which tells the engineer that the joint over pier P1 also needs attention. The engineering limits then single out the
G1 end zone of S2. The UPV and rebound values are normal almost everywhere, so the problem is one of water and steel, not of concrete quality.""")
R.h2("AI/ML output and its interpretation", "14.3")
R.fig("f23_case_ml", "Year-18 campaign: probability of each mechanism from the models using vibration only and all sources (* marks the mechanisms actually present), and change of the load-test response of each girder since year 0")
pf, pv = cm["p_fused"], cm["p_vibration"]
R.p(f"""The Mahalanobis distance of the year-18 campaign is {math.sqrt(cm['novelty_score']):.0f}, against a threshold of
{math.sqrt(cm['novelty_threshold']):.1f}, so the bridge is clearly no longer in its healthy state. Most of the distance comes from inspection
features that are always zero on the healthy bridge, such as rust stains. The models using all sources give the highest probabilities to
{', '.join(f'{SH[m].lower()} ({pf[m]:.2f})' for m in sorted(pf, key=pf.get, reverse=True)[:3])}. The models using vibration only give
{', '.join(f'{SH[m].lower()} {pv[m]:.2f}' for m in sorted(pv, key=pv.get, reverse=True)[:2])}: they find the bearings, which stiffen the span, and
nothing else. The localisation model gives G1 a probability of {cm['loc_proba'][0]:.2f}.""")
R.p(f"""The estimate of the prestress loss needed more care. My first version of the Gaussian-process model, which used the largest crack width
of any type as an input, gave <b>18&nbsp;%</b> for this campaign. It had been misled by the 0.45&nbsp;mm rust-stained splitting cracks at the joint
end: in its training data a visible crack on a girder almost always meant a flexural crack, and flexural cracks appear only after a loss of about
25&nbsp;%. The crack feature was then split into flexural cracks and cracks along bars or ducts, and the model was retrained. It now gives
{cm['pl_gpr_mean']:.1f} &plusmn; {1.96 * cm['pl_gpr_sd']:.1f}&nbsp;% (95&nbsp;% band), and its error on the test set also became smaller. Because
the levels and the VW strain changed mainly in G1, the G1 measurements were also inverted directly with the sensitivities of Eq. ([[eq:plsens]]),
which gives {cm['pl_G1_combined']:.1f} &plusmn; {1.96 * cm['pl_G1_sd']:.1f}&nbsp;% for G1 (levelling {cm['pl_G1_levelling']:.1f}&nbsp;%, VW strain
{cm['pl_G1_vw']:.1f}&nbsp;%). The two estimates overlap. The direct estimate for G1 was used in the structural check, because it rests on two
measurements whose physics is clear, and the Gaussian-process value was kept as an upper indication.""")
R.p("""The result also makes physical sense. A leaking joint explains the bearing distress in both spans next to it, the wet concrete and low
resistivity at the girder ends, and, through the anchorages at that end, a possible loss of prestress in G1. The slight rise of f<sub>1</sub> fits
restrained bearings. The absence of any change in UPV and rebound rules out a problem with the concrete. The load test changed little, which is
consistent with a girder that has lost some precompression but has not cracked. The output of the models is therefore consistent with one cause,
the joint, and three consequences. What it cannot show is whether the strands inside the ducts near the G1 anchorage are corroding; only an
investigation of the tendons can show that, so one has to be included in the maintenance scheme.""")
R.h2("Structural check with the estimated loss", "14.4")
R.p(f"""With the estimated loss in G1 ({asm['loss_used_pct']:.1f}&nbsp;%), the margin of the soffit stress against cracking under the rare
combination becomes {asm['sls_margin_design']:+.2f}&nbsp;MPa on the design basis (0.9P<sub>m</sub>) and {asm['sls_margin_mean']:+.2f}&nbsp;MPa with the
mean prestress, and the soffit stress under the quasi-permanent combination is still {asm['qp_soffit_mean']:+.1f}&nbsp;MPa compression. The flexural
capacity at mid-span is not affected by a loss of force, since the strands yield at failure anyway, and M<sub>Ed</sub>/M<sub>Rd</sub> remains
{asm['uls_ratio']:.2f}. At the corroded end B the vertical component of the prestress decreases, and the net design shear 2.75&nbsp;m from the bearing
rises from {asm['shear_Vnet_design']:.0f} to {asm['shear_Vnet']:.0f}&nbsp;kN, still below V<sub>Rd,max</sub> = {asm['shear_VRdmax']:.0f}&nbsp;kN and
covered by the links provided. The girder is therefore safe at present, but it is close to losing its design margin against cracking, and its
condition will become worse if water continues to reach the anchorage.""")
R.h2("Comparison with the hidden history", "14.5")
tru = cs["truth"]
desc = []
for m in tru:
    if m["type"] == "BR":
        desc.append(f"restraint of the bearings at end B (shear stiffness &times;{np.mean([row[1] for row in m['kh_mult']]):.0f}) with slight hardening at end A, from year {cs['onset']['bearing']:.0f}")
    if m["type"] == "CO":
        desc.append(f"corrosion of G1 at end B between {m['x0']:.1f} and {m['x1']:.1f} m with a strand area loss of {100 * m['loss']:.1f}&nbsp;%, from year {cs['onset']['corrosion']:.0f}")
    if m["type"] == "PL":
        desc.append(f"an extra loss of prestress of {100 * m['loss']:.1f}&nbsp;% in G1")
R.p(f"""The history used to generate the data was: failure of the joint seal over pier P2 in year {cs['onset']['joint']:.0f}; {'; '.join(desc)}. The
assessment found all three mechanisms and their common cause, located them correctly, and estimated the loss of prestress in G1 within its
stated uncertainty. It could not measure the corrosion of the strands inside the ducts, but it showed where this has to be
investigated.""")

# ====================================================================== 15
R.h1("Maintenance Scheme", "15")
R.p("""The maintenance decision combines all the available information: visual inspection, NDT, structural response measurement,
vibration monitoring and the AI/ML assessment. In this scheme the AI/ML output is one line of evidence among several. An action is triggered
by the evidence itself (a crack, a potential map, a bearing that does not move), and the AI/ML result mainly sets the priority and indicates
what should be checked next. For example, a probability of 0.8 for 'loss of prestress' leads to a measurement of the remaining force, and
not directly to strengthening of the girder.""")
R.h2("Decision framework", "15.1")
R.p("""The decision follows the sequence condition assessment &rarr; routine monitoring or detailed
investigation &rarr; maintenance or repair &rarr; reinspection and monitoring. If no monitoring alarm is raised and no defect is worse than CS2,
the bridge stays under routine monitoring. Otherwise a detailed investigation follows (special inspection, targeted NDT, structural assessment),
then maintenance or repair, in which the cause is removed before the damage is repaired, and finally reinspection and a new baseline. The
evidence patterns used to choose the actions are summarised in Table [[tab:matrix]].""")
R.table(["Evidence", "Interpretation", "Action"], [[a, b_, c] for a, b_, c in MT.DECISION_MATRIX],
        "Decision matrix used for B33", tag="matrix", widths=[34, 31, 35], small=True)
R.h2("Maintenance plan for B33 after the year-18 assessment", "15.2")
dec = cs["decision"]
cat_main, cat_note = (dec["category"].split(" - ") + [""])[:2]
R.p(f"""The assessment places the bridge in the category <b>{cat_main.lower()}</b> ({cat_note}). The plan in Table [[tab:plan]] first removes the
cause, i.e. the water at the joint over pier P2, then repairs the damage the water has done, then measures what the monitoring can only infer, and
finally establishes a new baseline so that the repaired bridge is again compared with itself.""")
R.table(["Action", "When", "Reason"], [[a, w_, r_] for a, w_, r_, n_ in dec["actions"]],
        "Maintenance actions, time frames and the evidence behind each", tag="plan", widths=[42, 16, 42], small=True)
R.h2("Sequence and justification", "15.3")
iv = dec["intervals"]
R.p("""The drains and the joint come first, because every other defect found is a consequence of water; repairing the concrete or the bearings
while the joint still leaks would waste the repair. The bearings of S2 at end B, and of S3 at end A under the same joint, are replaced by lifting
each span on jacks placed under the end diaphragm, which was designed as a jacking beam; the lift is limited to 10&nbsp;mm and the joint is closed to
traffic for one night per span. Concrete repair at the G1 end zone and on the pier cap follows the half-cell map, with an anti-carbonation coating on
the repaired and low-cover areas and a migrating corrosion inhibitor where the potentials are in the uncertain range.""")
R.p("""The investigation of the tendons (GPR to locate the ducts, impact-echo along them, and a borescope through a small hole drilled at the
anchorage and at the high points of the ducts) and a lift-off or stress-release test on G1 decide the last step. If the measured force confirms a
loss that removes the margin against cracking, external post-tensioning restores it, with two straight 12T15 tendons per girder, deviators at the
mid-span cross girder and anchorages on the end diaphragms (IRC:SP:40). If not, the girder is kept under closer monitoring.""")
R.h2("Inspection and monitoring after the works", "15.4")
R.p(f"""Until the works are complete, routine inspection is carried out {iv['routine']}, the monitoring data are reviewed weekly, and a principal
inspection follows the works and is repeated after one year. Crack gauges, two additional VW gauges at the end zone of G1 and bearing transducers on
all girders of S2 are installed at once and kept for at least two years after the repair. After the works the commissioning tests of Chapter 5 are
repeated (ambient vibration test, load test, NDT of the repaired zones) and become the new baseline. The bridge returns to the routine regime
(routine inspection before and after the monsoon, principal inspection every three years and quarterly review of the monitoring data) when two years
of data after the repair remain within the new limits.""")
R.h2("Longer-term improvements", "15.5")
R.p("""Each of the four expansion joints is a possible source of water. When the joint between S2 and S3 is replaced, a continuity (link) slab over
the piers should be considered, which would leave only the two joints at the abutments. The drain spouts of all spans should be extended at least
100&nbsp;mm below the soffit of the girders and given a drip groove; this costs little and removes the second source of wetting found in this
assessment.""")

# ====================================================================== 16
R.h1("Use of AI Tools and Verification of Results", "16")
R.p("""I used three AI tools during this work (Table [[tab:aitools]]): Claude for the Python scripts and for cross-checking the results,
ChatGPT for ideas on writing the report, and Gemini for parts of the HTML. The assumptions, models, equations, units, results,
interpretation and decisions in this report are my own responsibility.""")
R.parts.append('<div class="keep">')
R.table(["AI tool", "Used for"], [[a, u_] for a, u_ in AI_TOOLS], "AI tools used in this work", tag="aitools", widths=[24, 76])
R.parts.append('</div>')
R.p("""I used these tools only as assistants. I ran and checked everything they suggested or helped me to write, and changed it where it was
wrong. The problems I found and corrected in this way were:""")
R.ul(["""My first cable layout moved two cables sideways inside the 300&nbsp;mm web, which is impossible. I redrew the profiles so that the
lateral movement takes place inside the bottom bulb, and added a check of the duct positions along the whole girder.""",
      """Stressing all four cables at 10 days gave 18.6&nbsp;MPa at the soffit against an allowable 17.3&nbsp;MPa, so I changed to two-stage stressing.""",
      """The first vibration model gave peak accelerations of 170&nbsp;mg, because the bounce frequencies of the lorries coincided with the first mode
and the ambient forcing was too strong. I recalibrated both until the night-time RMS (0.6 to 0.8&nbsp;mg) and the traffic peaks (10 to
25&nbsp;mg) were of the order reported for bridges of this size.""",
      """With 10-minute records and simple peak picking the scatter of f<sub>1</sub> was &plusmn;0.5&nbsp;%; 30-minute records with a fitted spectrum
reduced it to about 0.1&nbsp;%.""",
      """The AutoMAC check showed that my first sensor layout could not separate two pairs of modes, so I added two accelerometers.""",
      """Identifying the commissioning test with the mode shapes of the design model biased f<sub>2</sub> by 3.6&nbsp;%; I now identify the
commissioning test without model shapes (FDD).""",
      """The first synthetic inspection detected every hardened bearing and every case of 2&nbsp;% corrosion, which made inspection look perfect.
I made detection depend on severity, and also allowed leaking joints on sound bridges.""",
      """The novelty detector failed at first on the inspection features, because their covariance matrix was singular; I added a small value to
its diagonal and checked the effect on the false-alarm rate.""",
      """My first prestress-loss model mistook corrosion cracks for flexural cracks and estimated an 18&nbsp;% loss for the year-18 campaign. I split
the crack feature by type and checked the estimate against a direct physical inversion (Section 14.3).""",
      """I checked the impact factor for the 70R wheeled vehicle on a 33&nbsp;m span, the IRC:6 footway load for spans over 30&nbsp;m and the
reduction for three lanes against the code."""])
n_ok = sum(c["ok"] for c in ver)
R.p(f"""In addition, I made {len(ver)} independent checks of the computed results with simpler methods: hand formulas, a separate influence-line
calculation, Courbon's method, the frequency formula for a simply supported beam, and recovery of the known parameters from synthetic measurements.
{n_ok} of the {len(ver)} checks are within their tolerances (Appendix B).""")

# ====================================================================== 17
R.h1("Conclusions", "17")
R.table(["Item", "Result"], [
    ["Configuration", f"33 m span; six 1.80 m deep M45 girders at 2.70 m; 225 mm deck slab; end diaphragms and one cross girder at mid-span"],
    ["Live load on outer girder", f"{f(MLL_G1, 0)} kN m (grillage), {f(d['courbon']['M'][0], 0)} kN m (Courbon)"],
    ["Prestress", f"four 12T15 cables per girder, two-stage stressing; <i>P</i><sub>e</sub> = {f(tot['Pe_mid'] / 1e3, 2)} MN, total loss {pct(1 - tot['Pe_mid'] / tot['Pj'], 1)}"],
    ["Serviceability and strength", f"rare-combination soffit stress {ss['rare_min_bot']:+.2f} MPa (uncracked); <i>M</i><sub>Ed</sub>/<i>M</i><sub>Rd</sub> = {d['uls']['G1']['MEd'] / d['uls']['G1']['M']:.2f}; live-load deflection {d['deflection']['LL_mm']['G1']:.1f} mm"],
    ["Healthy baseline", f"<i>f</i><sub>1</sub> = {b['f_ref'][0]:.2f} Hz, <i>f</i><sub>2</sub> = {b['f_ref'][1]:.2f} Hz at 25 &deg;C; {nb['f1']['per_degC_pct']:+.2f} %/&deg;C; load-test deflection {b['load_test_ref']['d'][2]:.2f} mm"],
    ["AI/ML (test set)", f"macro-F1 {sup['Vibration only']['macro_F1']:.2f} with vibration only, {sup['All sources fused']['macro_F1']:.2f} with all sources; prestress loss estimated with MAE {g['gpr']['MAE']:.1f} %"],
    ["Year-18 assessment", f"BCI {cs['decision']['bci']:.0f}; category {cat_main.lower()} ({cat_note}); leaking joint over pier P2 causing restrained bearings, corrosion near the G1 anchorage and a loss of prestress of about {cm['pl_G1_combined']:.0f} % in G1"],
], "Main results", widths=[24, 76], small=True)
R.p("""The main conclusions of the study are:""")
R.ul([f"""A 33&nbsp;m span with six 1.80&nbsp;m deep post-tensioned M45 girders at 2.70&nbsp;m, four 12T15 cables per girder and two-stage stressing
satisfies IRC:6 and IRC:112, with the soffit uncracked under the rare combination (margin about 1&nbsp;MPa) and a moment utilisation of
{d['uls']['G1']['MEd'] / d['uls']['G1']['M']:.2f}. The number of girders was governed by the limited construction depth rather than by the quantity
of material.""",
      """The healthy baseline has to come from measurements on the bridge; the design model alone is not good enough. It was 7 to 10&nbsp;% off in frequency; after three parameters had
been updated on the commissioning tests, the twin reproduced them within 0.2&nbsp;%. One year of data provided the temperature and traffic
normalisation without which the frequencies of this bridge are not usable, since they vary by about 4&nbsp;% over a year.""",
      """Vibration monitoring alone is poor at detecting the two mechanisms that usually decide the life of a post-tensioned bridge, loss of
prestress and corrosion. Levelling, embedded VW gauges and bearing displacement transducers are inexpensive and detect them much better, and
NDT and visual inspection remain essential for corrosion.""",
      f"""The AI/ML assessment became useful mainly through combining the data sources, much more than through the choice of algorithm: the macro-F1 rose from
{sup['Vibration only']['macro_F1']:.2f} with vibration to {sup['All sources fused']['macro_F1']:.2f} with all sources. The importance of the features
agrees with the physics, and the Gaussian process gives the loss of prestress with a realistic uncertainty.""",
      """Applied to the bridge at 18 years, the assessment traced the defects to one cause, a leaking joint, located them, estimated the loss of
prestress in the affected girder, and led to a maintenance plan that removes the cause, repairs the damage, measures what the monitoring can only
infer, and re-establishes the baseline. My first version of the prestress model was misled by corrosion cracks, which showed me that every ML
output has to be checked against the physics before it is used for a decision."""])
R.p("""The work could be extended by calibrating the noise models and the detection thresholds with data from an instrumented bridge, by
replacing the grillage with a finite-element model that represents bond and crack behaviour, and by adding the cost of the actions so that
the timing of maintenance can be optimised over the life of the bridge.""")

# ====================================================================== references
R.h1("References", newpage=True)
refs = [
    "Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45, 5&ndash;32.",
    "Brincker, R., Zhang, L. and Andersen, P. (2001). Modal identification of output-only systems using frequency domain decomposition. <i>Smart Materials and Structures</i>, 10(3), 441&ndash;445.",
    "Concrete Society (2010). <i>Durable Post-Tensioned Concrete Structures</i>, Technical Report 72. Camberley.",
    "Farrar, C.R. and Worden, K. (2013). <i>Structural Health Monitoring: A Machine Learning Perspective</i>. Wiley.",
    "Hambly, E.C. (1991). <i>Bridge Deck Behaviour</i>, 2nd ed. E &amp; FN Spon, London.",
    "Hamed, E. and Frostig, Y. (2006). Natural frequencies of bonded and unbonded prestressed beams &ndash; prestress force effects. <i>Journal of Sound and Vibration</i>, 295, 28&ndash;39.",
    "IRC:5-2015. <i>Standard Specifications and Code of Practice for Road Bridges, Section I: General Features of Design</i>. Indian Roads Congress, New Delhi.",
    "IRC:6-2017. <i>Standard Specifications and Code of Practice for Road Bridges, Section II: Loads and Load Combinations</i>. Indian Roads Congress, New Delhi.",
    "IRC:83 (Part II)-2018. <i>Standard Specifications and Code of Practice for Road Bridges, Section IX: Bearings, Part II: Elastomeric Bearings</i>. Indian Roads Congress, New Delhi.",
    "IRC:112-2020. <i>Code of Practice for Concrete Road Bridges</i>. Indian Roads Congress, New Delhi.",
    "IRC:SP:35-1990. <i>Guidelines for Inspection and Maintenance of Bridges</i>. Indian Roads Congress, New Delhi.",
    "IRC:SP:40-2019. <i>Guidelines on Techniques for Strengthening and Rehabilitation of Bridges</i>. Indian Roads Congress, New Delhi.",
    "IRC:SP:51-2015. <i>Guidelines for Load Testing of Bridges</i>. Indian Roads Congress, New Delhi.",
    "IRC:SP:52-1999. <i>Bridge Inspector's Reference Manual</i>. Indian Roads Congress, New Delhi.",
    "IRC:SP:80-2008. <i>Guidelines for Corrosion Prevention, Monitoring and Remedial Measures for Concrete Bridge Structures</i>. Indian Roads Congress, New Delhi.",
    "IS 516 (Part 5/Sec 1):2018 and IS 516 (Part 5/Sec 4):2020. <i>Hardened Concrete &ndash; Methods of Test, Non-Destructive Testing: Ultrasonic Pulse Velocity; Rebound Hammer</i>. Bureau of Indian Standards, New Delhi.",
    "ASTM C876-15. <i>Standard Test Method for Corrosion Potentials of Uncoated Reinforcing Steel in Concrete</i>. ASTM International.",
    "Peeters, B. and De Roeck, G. (2001). One-year monitoring of the Z24-Bridge: environmental effects versus damage events. <i>Earthquake Engineering and Structural Dynamics</i>, 30(2), 149&ndash;171.",
    "Rasmussen, C.E. and Williams, C.K.I. (2006). <i>Gaussian Processes for Machine Learning</i>. MIT Press.",
    "Sohn, H. (2007). Effects of environmental and operational variability on structural health monitoring. <i>Philosophical Transactions of the Royal Society A</i>, 365, 539&ndash;560.",
    "Woodward, R.J. and Williams, F.W. (1988). Collapse of Ynys-y-Gwas bridge, West Glamorgan. <i>Proceedings of the Institution of Civil Engineers, Part 1</i>, 84, 635&ndash;669.",
    "Worden, K., Manson, G. and Fieller, N.R.J. (2000). Damage detection using outlier analysis. <i>Journal of Sound and Vibration</i>, 229(3), 647&ndash;667.",
]
R.raw("<ol style='font-size:11pt; line-height:1.3'>" + "".join(f"<li style='margin-bottom:2.5pt'>{r}</li>" for r in refs) + "</ol>")

# ====================================================================== appendices
R.h1("Appendix A: Program Files", newpage=True)
R.table(["File", "Content"], [
    ["psc/inputs.py", "all adopted dimensions, materials, loads and criteria"],
    ["psc/section.py, grillage.py, vehicles.py", "section properties; grillage model (static analysis, influence surfaces, modal analysis); IRC vehicles and the search for the worst position"],
    ["psc/prestress.py, design.py", "cable geometry, losses, stage stresses, ULS, shear, bearings, comparison of girder arrangements"],
    ["psc/twin.py, dynamics.py", "digital twin (deterioration state to responses); vibration simulation; FDD and modal-filter identification"],
    ["psc/condition.py, scenarios.py, features.py", "inspection and NDT models; deterioration scenarios; generation of campaigns; feature extraction"],
    ["psc/mlkit.py, maintenance.py", "machine-learning methods written with numpy and scipy; decision rules"],
    ["run_01_design.py to run_05_case_study.py", "the five steps of the study, run in this order"],
    ["verify.py", "independent checks (Appendix B)"],
    ["make_figures.py, make_figures_ml.py, build_report.py", "figures and this report"],
    ["inspection_app/index.html", "digital visual inspection application (single file, works offline)"],
    ["out/data/*.csv", "baseline year, scenario library, NDT surveys, visual inspection, 18-year monitoring record"],
], "Program and data files", widths=[38, 62], small=True)
R.p("""The programs need Python 3 with numpy, scipy and matplotlib. The five <code>run_*.py</code> scripts are run in order, followed by
<code>verify.py</code>, <code>make_figures.py</code> and <code>build_report.py</code>. The complete chain takes about half an hour on my laptop
(16 threads), most of it for the 3600 vibration records of the scenario library.""")
R.h1("Appendix B: Verification Checks")
R.table(["Check", "Computed", "Independent", "Difference", "Tolerance", "Result"],
        [[c["check"] + (f" ({c['note']})" if c["note"] else ""), f"{c['computed']:.4g} {c['unit']}".replace("microstrain", "&mu;&epsilon;").replace("degC", "&deg;C"),
          f"{c['hand']:.4g}", f"{100 * c['rel_diff']:.1f} %", f"{100 * c['tol']:.0f} %", "pass" if c["ok"] else "<b>fail</b>"] for c in ver],
        "Independent checks of the computed results", num_cols=(1, 2, 3, 4), small=True)

# ====================================================================== write and print (two passes for the page numbers)
import pathlib
out = pathlib.Path("report"); out.mkdir(exist_ok=True)
html_path = out / "B33_PSC_Bridge_Integrated_Study.html"
pdf_path = out / "B33_PSC_Bridge_Integrated_Study.pdf"
html_path.write_text(R.html(), encoding="utf-8")
print_pdf(html_path, pdf_path)
pages = find_pages(pdf_path, R)
html_path.write_text(R.html(pages), encoding="utf-8")
print_pdf(html_path, pdf_path)
pages2 = find_pages(pdf_path, R)
if pages2 != pages:
    html_path.write_text(R.html(pages2), encoding="utf-8")
    print_pdf(html_path, pdf_path)
import pymupdf, os
doc = pymupdf.open(str(pdf_path))
doc.set_metadata({"title": "Post-Tensioned Prestressed Concrete I-Girder Bridge of 33 m Span", "author": f"{STUDENT} ({ROLL})",
                  "subject": "CE60258 One-Month Integrated Big Problem", "creator": "", "producer": ""})
doc.save(str(pdf_path) + ".tmp", garbage=3, deflate=True)
doc.close()
os.replace(str(pdf_path) + ".tmp", pdf_path)
n_found = sum(1 for _, _, _, a in R.toc if a in pages2)
print(f"report: {pdf_path} ({pymupdf.open(str(pdf_path)).page_count} pages), {R.fig_no} figures, {R.tab_no} tables, "
      f"{R.eq_no} equations; headings located {n_found}/{len(R.toc)}")
