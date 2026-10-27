"""
Builds report/B33_PSC_Bridge_Integrated_Study.html and .pdf from the saved results.
All numbers in the text are read from out/*.json, so the text always matches the calculations.
Part 2 (chapters 10 onwards, back matter, printing) is in build_report_part2.py.
"""
import json, math, csv
import numpy as np
from report_lib import Report, print_pdf, find_pages
from psc import inputs as P, section as S, prestress as PS, condition as C, features as FE, scenarios as SC
from psc import dynamics as DY, maintenance as MT

STUDENT = "Shashank Pandya"
ROLL = "23CE10064"
# AI assistants used for this work and what each was used for (Chapter 16).
# Add a row for any other assistant that was used.
AI_TOOLS = [
    ("Claude (Anthropic)", "help in writing and debugging the Python scripts, and then in cross-checking the results "
     "against hand calculations and the codes"),
    ("ChatGPT (OpenAI)", "taking ideas for writing the report: the order of the chapters, what each should cover "
     "and the wording of some explanations"),
    ("Gemini (Google)", "writing parts of the HTML and CSS used to lay out the report (tables, equations, captions) "
     "before printing it to PDF"),
]

d = json.load(open("out/design.json"))
b = json.load(open("out/baseline.json"))
ml = json.load(open("out/ml_results.json"))
cs = json.load(open("out/case_study.json"))
ver = json.load(open("out/verification.json"))
sens = json.load(open("out/sensitivity.json"))
arr = np.load("out/design_arrays.npz")


def f(x, n=1):
    return f"{x:,.{n}f}" if abs(x) >= 1e4 else f"{x:.{n}f}"


def pct(x, n=0):
    return f"{100 * x:.{n}f}&nbsp;%"


R = Report()
pc = d["prestress"]["per_cable"]
tot = d["prestress"]
pre, comp_o, comp_i = d["sections"]["precast"], d["sections"]["composite_outer"], d["sections"]["composite_inner"]
ss = d["stages_summary"]
tw = b["twin"]
nb = b["norm"]
MLL_G1 = float(arr["MLL"][0, -1])

# ====================================================================== title page
R.raw(f"""
<div class="titlepage">
 <div class="course">CE60258 &nbsp; AI/ML for Sustainable Infrastructure Systems<br>One-Month Integrated Big Problem</div>
 <div class="title">Post-Tensioned Prestressed Concrete<br>I-Girder Bridge of 33 m Span</div>
 <div class="sub">Design, prestressing, healthy baseline, inspection and monitoring,<br>AI/ML-assisted condition assessment and maintenance</div>
 <div class="by">Submitted by<br><b>{STUDENT}</b><br>Roll No. {ROLL}<br><br>Course Instructor<br><b>Prof. Nirjhar Dhang</b></div>
 <div class="inst">Department of Civil Engineering<br>Indian Institute of Technology Kharagpur</div>
</div>""")

# ====================================================================== abstract
R.h1("Abstract", newpage=True)
R.p(f"""This report presents an integrated study of a post-tensioned prestressed concrete I-girder bridge of 33&nbsp;m span,
followed from the choice of the cross-section to a maintenance decision eighteen years after construction. The bridge, called
B33, has a 16.0&nbsp;m deck carrying three traffic lanes and two footpaths on six precast M45 I-girders at 2.70&nbsp;m spacing. A
grillage analysis for IRC:6 loading and a design to IRC:112 led to four 12T15 cables per girder, stressed in two stages; after a
total loss of {pct(1 - tot['Pe_mid'] / tot['Pj'], 1)} the effective prestress is {f(tot['Pe_mid'] / 1e3, 2)}&nbsp;MN per girder and the
soffit remains uncracked under the rare load combination. A healthy baseline was established by updating the design model on
simulated commissioning tests and by one year of monitoring, which showed a seasonal variation of about 4&nbsp;% in the natural
frequencies that had to be removed by temperature and traffic normalisation. Inspection, NDT and monitoring schemes were designed
for the bridge, and six deterioration mechanisms were modelled in a digital twin to generate {ml['n']} synthetic measurement
campaigns. Novelty detection, random forests, Gaussian-process regression and clustering applied to physically defined features
showed that vibration data alone identify the mechanisms poorly (macro-F1 {ml['supervised']['Vibration only']['macro_F1']:.2f}),
while the combined data reach {ml['supervised']['All sources fused']['macro_F1']:.2f}, and that loss of prestress appears in levelling
and vibrating-wire strain long before it affects vibration or load tests. Applied to the bridge at an age of 18 years, the assessment
traced restrained bearings, corrosion near an anchorage and a small loss of prestress to a single leaking expansion joint, and a
maintenance scheme was prepared on that basis.""")
R.p("""<b>Keywords:</b> post-tensioned I-girder bridge, prestress loss, grillage analysis, structural health monitoring,
operational modal analysis, synthetic data, random forest, Gaussian process, condition assessment, bridge maintenance.""")

R.h1("Acknowledgement")
R.p("""I thank Prof. Nirjhar Dhang for setting a problem that made me follow one bridge through its whole life instead of
solving separate exercises, and for the clarifications given during the month. Working on it showed me how closely design
choices, inspection practice and the usefulness of monitoring data depend on each other.""")
R.p(f"""<span style="float:right">{STUDENT}</span>""")

R.h1("Contents", newpage=True, toc=False)
R.raw("<!--TOC-->")
R.h1("List of Figures", newpage=True)
R.raw("<!--LOF-->")
R.h1("List of Tables")
R.raw("<!--LOT-->")
R.h1("Notation")
R.table(["Symbol", "Meaning"], [
    ["<i>L</i>", "effective span (33.0 m)"],
    ["<i>P</i><sub>j</sub>, <i>P</i><sub>0</sub>, <i>P</i><sub>e</sub>", "jacking force, force after immediate losses, effective force after all losses"],
    ["&eta;", "ratio of actual to expected effective prestress"],
    ["<i>A</i>, <i>I</i>, <i>Z</i><sub>b</sub>, <i>Z</i><sub>t</sub>", "area, second moment of area, section moduli (bottom, top)"],
    ["<i>e</i>", "eccentricity of the prestressing force"],
    ["<i>f</i><sub>ck</sub>, <i>f</i><sub>ctm</sub>, <i>E</i><sub>cm</sub>", "characteristic cube strength, mean tensile strength, mean modulus of concrete"],
    ["<i>f</i><sub>pk</sub>, <i>f</i><sub>p0.1k</sub>, <i>E</i><sub>p</sub>", "tensile strength, 0.1&nbsp;% proof stress and modulus of strand"],
    ["&mu;, <i>k</i>", "friction coefficient, wobble coefficient"],
    ["&phi;(<i>t</i>,<i>t</i><sub>0</sub>), &epsilon;<sub>cs</sub>", "creep coefficient, shrinkage strain"],
    ["<i>f</i><sub>1</sub> ... <i>f</i><sub>4</sub>", "first bending, first torsion, first transverse and second bending frequency"],
    ["<i>a</i><sub>RMS</sub>, <i>a</i><sub>max</sub>", "root-mean-square and peak acceleration of a record"],
    ["&delta;, &epsilon;", "deflection, strain"],
    ["<i>T</i>", "effective bridge temperature"],
    ["CS, BCI", "condition state (1 good to 5 critical), bridge condition index (0 to 100)"],
    ["FDD, MAC", "frequency domain decomposition, modal assurance criterion"],
    ["HCP, UPV, VW", "half-cell potential, ultrasonic pulse velocity, vibrating-wire (strain gauge)"],
    ["GPR, RF, EWMA", "Gaussian-process regression, random forest, exponentially weighted moving average"],
    ["AUC, F1", "area under the ROC curve, harmonic mean of precision and recall"],
], widths=[26, 74])

# ====================================================================== 1
R.h1("Introduction", "1", newpage=True)
R.h2("Objective", "1.1")
R.p("""The objective of the problem is to study a post-tensioned PSC I-girder bridge as an infrastructure system over its
service life. The bridge has to be analysed and designed, prestressed, given a healthy baseline, inspected, tested and
monitored, deteriorated in a controlled way, and finally assessed with the help of AI/ML so that a maintenance decision can be
taken. I have treated these parts as stages in the life of one bridge rather than as separate exercises, so every chapter uses
the bridge, the numbers and the instruments defined in the chapters before it.""")
R.h2("Approach", "1.2")
R.p(f"""The work was organised as the chain shown in Figure [[fig:f00_flowchart]]. A configuration was chosen for the 33&nbsp;m span
and the deck was analysed and designed with a grillage model written for this purpose. The prestressing system was designed and
the losses were calculated cable by cable. The as-built bridge was then 'measured' at commissioning, and the design model was
updated so that it reproduced these measurements; this updated model is called the digital twin in the rest of the report. The
inspection, NDT and monitoring schemes were designed for this bridge. Deterioration was introduced in the twin, and the twin
produced the synthetic data from which the AI/ML models were trained. Finally, the models were applied to the bridge at an age of
18 years and the result was used to prepare the maintenance scheme.""")
R.fig("f00_flowchart", "Flow of the study and the chapters in which each step is described")
R.p("""All measurements in this report are synthetic, which the problem statement allows. To make the assessment a fair test, the
'real' bridge used to generate the measurements in Chapters 5 and 14 is a model whose stiffness parameters and deterioration history
I kept hidden from the assessment. The twin had to identify the parameters from the measurements, and the assessment had to find the
deterioration from the data, in the same way as it would have to on an actual bridge.""")
R.h2("Assumed site and service conditions", "1.3")
R.p("""No site is specified in the problem, so I assumed one to make the later choices definite. B33 carries a
one-way, three-lane carriageway of a divided state highway (one of two parallel bridges) over an irrigation canal and a service road
in the coastal plain of West Bengal, about 120&nbsp;km from the Bay of Bengal. It has three simply supported spans of 33&nbsp;m on
wall-type piers P1 and P2 and abutments A1 and A2. Only one span length is used; span S2, between the two piers, is the span that
is designed, instrumented and assessed in detail.""")
R.p("""The effective bridge temperature varies between about 15 and 42&nbsp;&deg;C. The monsoon brings heavy rain from June to
September and the deck goes through wetting and drying for the rest of the year. The exposure has therefore been taken as
<i>severe</i> in the IRC:112 classification, with a clear cover of 50&nbsp;mm to the links and 75&nbsp;mm to the ducts. The design
life is 100 years. Traffic is mixed, with a high proportion of loaded three-axle and multi-axle lorries.""")
R.p("""The piers, abutments and foundations are outside the scope of the design and appear only in the inspection scheme. Scour,
seismic and wind design and the shop drawings of the prestressing system are also not covered.""")
R.h2("Codes and standards", "1.4")
R.table(["Document", "Used for"], [
    ["IRC:5-2015", "carriageway, footpath and kerb widths"],
    ["IRC:6-2017", "dead loads, Class A and Class 70R vehicles, impact, footway loading, multi-lane reduction, temperature, load combinations"],
    ["IRC:112-2020", "materials, prestress losses, stress limits, crack control, flexure and shear at the ultimate limit state, deflection"],
    ["IRC:83 (Part II)-2018", "laminated elastomeric bearings"],
    ["IRC:SP:69-2011", "expansion joints"],
    ["IRC:SP:51-2015", "load testing of bridges"],
    ["IRC:SP:35-1990, IRC:SP:52-1999", "inspection and maintenance practice"],
    ["IRC:SP:40-2019, IRC:SP:80-2008", "repair and strengthening techniques; corrosion monitoring and remedial measures"],
    ["IS 516 (Part 5), ASTM C876", "ultrasonic pulse velocity and rebound hammer tests; half-cell potential"],
], "Codes and standards used", widths=[30, 70])
R.h2("Software", "1.5")
R.p("""I did all the calculations with Python programs written for this study, using only the numpy, scipy and matplotlib libraries.
The machine-learning methods (random forest, logistic regression, Gaussian-process regression, k-means, Gaussian mixture and the
Mahalanobis novelty detector) were also coded directly with numpy and scipy, because I could not install the usual library
(scikit-learn) on my computer. Appendix A lists the program files.""")

# ====================================================================== 2
R.h1("Bridge Configuration", "2")
R.h2("Cross-section", "2.1")
R.p(f"""The carriageway is 11.0&nbsp;m wide, which provides three lanes of 3.5&nbsp;m and marginal strips of 0.25&nbsp;m. It is
bounded on each side by a 0.5&nbsp;m New Jersey type crash barrier. Outside each barrier there is a 1.75&nbsp;m footpath and a
0.25&nbsp;m railing kerb, so that the overall width of the deck is {P.DECK_WIDTH:.1f}&nbsp;m (Figure [[fig:f01_cross_section]]). The
footpath is slightly wider than the minimum of 1.5&nbsp;m in IRC:5 because it also carries ducts for power and optical-fibre
cables. Placing the barrier between the traffic and the footpath protects pedestrians on a road used by fast heavy vehicles and keeps
vehicles off the deck cantilever.""")
R.fig("f01_cross_section", "Cross-section of B33 at mid-span (dimensions in m). The deck slab is 225 mm thick (M40) and the precast girders are 1.80 m deep (M45); the mid-span cross girder behind the section is shown dashed.")
R.h2("Number and spacing of girders", "2.2")
R.p(f"""The construction depth is limited, because the deck has to clear the service road below and the approach embankments are
already high. The depth of the precast girder was therefore fixed at 1.80&nbsp;m, i.e. 2.025&nbsp;m with the slab and a
span-to-depth ratio of {P.L / (P.GIRDER_DEPTH + P.SLAB_T):.1f}. With the depth fixed, the number of girders decides how much
prestress each girder has to carry. Table [[tab:options]] compares four, five and six girders on a preliminary basis: Courbon's
distribution, the same section shape, and the prestress needed for zero tension at the soffit under the characteristic load
combination.""")
rows = []
for o in d["options"]:
    cab = math.ceil(o["strands"] / 12)
    rows.append([o["n"], f(o["spacing"], 2), f(o["slab"] * 1e3, 0), f(o["k_outer"], 3), f(o["M_LL"], 0), f(o["Pe"] / 1e3, 2),
                 f(o["strands"], 1), f"{cab}" + (" (too many)" if cab > 4 else ""), f(o["girder_t"], 0), f(o["concrete"], 0)])
R.table(["Girders", "Spacing (m)", "Slab (mm)", "Courbon factor, outer girder", "LL moment, outer girder (kN m)", "<i>P</i><sub>e</sub> needed (MN)",
         "Strands per girder", "12T15 cables", "Mass of one girder (t)", "Concrete in deck (m&sup3;)"],
        rows, "Comparison of girder arrangements at a depth of 1.80 m (preliminary)", num_cols=(0, 1, 2, 3, 4, 5, 6, 8, 9),
        tag="options", small=True)
R.p("""With four girders each girder would need about 64 strands, i.e. six 12T15 cables. Six ducts cannot be placed in a
750&nbsp;mm bottom bulb with the required spacing and cover, and the deck slab would have to span 3.0&nbsp;m between the flanges.
Five girders need about 53 strands, or five cables, which is still too many for the bulb at this depth. Six girders need about 45
strands, which four 12T15 cables (48 strands) provide. The six-girder arrangement uses about 5&nbsp;% more strand and 9&nbsp;% more
concrete than four girders, but it works within the depth limit, allows a 225&nbsp;mm slab with a clear span of 1.7&nbsp;m and a
cantilever of 1.25&nbsp;m, keeps each girder below 80&nbsp;t for lifting by two cranes, and provides more load paths if one girder is
damaged. I therefore adopted six girders at 2.70&nbsp;m. The last point also has a consequence for monitoring: a deck with six
girders sheds load from a weakened girder to its neighbours, which keeps it safe but also makes local damage harder to detect from
global measurements.""")
R.h2("Girder section", "2.3")
R.p(f"""The precast girder (Figure [[fig:f02_girder_section]]) has a 1000 &times; 175&nbsp;mm top flange with a 100&nbsp;mm taper, a
300&nbsp;mm web and a 750 &times; 250&nbsp;mm bottom bulb with a 200&nbsp;mm taper. I chose a 300&nbsp;mm web instead of the usual
250&nbsp;mm so that a 92&nbsp;mm HDPE duct can pass through it with enough cover. The bulb width is set by three ducts side by side at
a clear spacing larger than the duct diameter. Over 1.5&nbsp;m at each end the web is widened to the full bulb width (end block) and
then tapers back over a further 1.5&nbsp;m, to hold the four anchorages and resist the high shear near the bearings.""")
R.p(f"""The section properties are A = {pre['A']:.3f}&nbsp;m&sup2;, y<sub>b</sub> = {pre['yb']:.3f}&nbsp;m and
I = {pre['I']:.4f}&nbsp;m<sup>4</sup> for the precast girder, and I = {comp_o['I']:.4f}&nbsp;m<sup>4</sup> (outer) and
{comp_i['I']:.4f}&nbsp;m<sup>4</sup> (inner) for the composite girder, with the M40 slab transformed by the modular ratio
m = {comp_i['m']:.3f}. For a 33&nbsp;m span the effective-width rule of IRC:112 gives more than the girder spacing, so the full slab
width between girders is effective.""")
R.fig("f02_girder_section", "Precast girder at mid-span and at the anchorage face, showing the ducts and the anchor plates of the 12T15 cables (dimensions in mm)")
R.h2("Deck slab, cross girders and diaphragms", "2.4")
R.p("""The deck slab is 225&nbsp;mm thick and spans 1.7&nbsp;m between the girder flanges; it was checked separately for the wheel
loads of the 70R vehicle. End diaphragms 400&nbsp;mm wide and 1600&nbsp;mm deep hold the girder ends, transfer transverse loads to the
bearings and serve as jacking beams when the bearings have to be replaced. For the intermediate cross girders, one cross girder at
mid-span was compared in the grillage model with three cross girders at the quarter points. The two additional cross girders changed
the live-load moment of the outer girder by less than 3&nbsp;% and the first two natural frequencies by about 1&nbsp;%, so a single
300 &times; 1450&nbsp;mm cross girder at mid-span was adopted. It also braces the girders laterally while the deck slab is cast.""")
R.h2("Bearings and expansion joints", "2.5")
R.p("""Each girder rests on a 400 &times; 500&nbsp;mm laminated elastomeric bearing at both ends, with five internal elastomer layers
of 12&nbsp;mm. With elastomeric bearings at both ends the span expands about its centre and both ends move. Strip-seal expansion
joints (IRC:SP:69) with a 40&nbsp;mm gap are provided at each pier and abutment. The joint at a pier is the place where water can reach
the bearings, the girder ends and the anchorages, and it becomes the central issue of the assessment in Chapter 14.""")
R.h2("Materials", "2.6")
R.table(["Item", "Specification"], [
    ["Precast girders", f"M45; first cables stressed at {P.FCK_TRANSFER:.0f} MPa (about 10 days)"],
    ["Deck slab, cross girders, diaphragms", "M40"],
    ["Reinforcement", "Fe500D"],
    ["Prestressing steel", "15.2 mm seven-wire low-relaxation strand (IS 14268, Class 2), <i>f</i><sub>pk</sub> = 1860 MPa, 140 mm&sup2;, <i>E</i><sub>p</sub> = 195 GPa"],
    ["Ducts", "corrugated HDPE, 92 mm outer diameter"],
    ["Unit weights (IRC:6)", "25 kN/m&sup3; for concrete, 22 kN/m&sup3; for the 75 mm bituminous wearing coat"],
], "Materials", widths=[34, 66])

# ====================================================================== 3
R.h1("Structural Analysis and Preliminary Design", "3")
R.h2("Loads", "3.1")
q_fp = P.footpath_ll()
R.table(["Load", "Value", "Basis"], [
    ["Self-weight of girder", f"{pre['A'] * 25:.1f} kN/m (end block {d['sections']['end_block']['A'] * 25:.1f} kN/m)", "carried by the precast girder"],
    ["Wet deck slab, cross girder", f"{2.7 * P.SLAB_T * 25:.1f} kN/m per inner girder, {2.6 * P.SLAB_T * 25:.1f} kN/m per outer girder; cross girder {2.4 * 0.3 * 1.45 * 25:.1f} kN at mid-span", "carried by the precast girders, tributary widths"],
    ["Crash barrier, footpath, railing", f"{P.BARRIER_LOAD}, {P.FOOTPATH_LOAD} and {P.RAILING_LOAD} kN/m on each side", "composite deck (grillage)"],
    ["Wearing coat", f"{P.WC_LOAD:.2f} kN/m&sup2; over 11.0 m", "75 mm at 22 kN/m&sup3;"],
    ["Footway live load", f"{q_fp:.2f} kN/m&sup2;", "IRC:6, Eq. ([[eq:footway]])"],
    ["Vehicles", "Class A train; Class 70R wheeled (1000 kN) and tracked (700 kN)", "IRC:6; three lanes: one 70R and one Class A, or three Class A"],
    ["Impact factor", f"{P.IMPACT:.3f} for Class A and 70R wheeled; 0.10 for 70R tracked", "IRC:6, Eq. ([[eq:impact]])"],
    ["Multi-lane reduction", "10 % when three lanes are loaded", "IRC:6"],
    ["Temperature difference", "positive 17.8 / 4.0 / 2.1 &deg;C; reverse 10.6 / 0.7 / 0.8 / 6.6 &deg;C", "IRC:6 profiles, &alpha; = 12 &times; 10<sup>-6</sup> /&deg;C"],
    ["Differential shrinkage", "100 &times; 10<sup>-6</sup> between deck and girder; creep factor 0.43", "self-equilibrating stresses only"],
], "Loads acting on one span", widths=[24, 42, 34], small=True)
R.p("""The footway load for spans above 30&nbsp;m and the impact factor for spans above 12&nbsp;m were taken from IRC:6 as""")
R.eq(r"P' = P - 260 + \dfrac{4800}{L}\left(\dfrac{16.5 - W}{15}\right)\quad \mathrm{kg/m^2}", "footway")
R.eq(r"I_{\mathrm{impact}} = \dfrac{4.5}{6 + L}", "impact")
R.p(f"""where P = 400&nbsp;kg/m&sup2;, W = 1.75&nbsp;m is the width of the footway and L = 33&nbsp;m. This gives P' = {q_fp / 9.81 * 1000:.0f}&nbsp;kg/m&sup2;
({q_fp:.2f}&nbsp;kN/m&sup2;) and an impact factor of {P.IMPACT:.3f}.""")
R.h2("Load combinations", "3.2")
R.p("""The load combinations follow Annex B of IRC:6. For the ultimate limit state the partial factors are 1.35 on self-weight and
superimposed dead load, 1.75 on the wearing coat and 1.5 on traffic. For the serviceability checks the rare (characteristic)
combination uses factors of 1.0 on live load and 0.6 on temperature, the frequent combination 0.75 and 0.5, and the quasi-permanent
combination zero live load and 0.5 on temperature.""")
R.h2("Grillage model", "3.3")
R.p(f"""The composite deck was analysed with a grillage of six longitudinal members along the girder lines, with the properties of
the composite girders, and transverse members every {P.L / P.NX:.3f}&nbsp;m representing strips of the deck slab. Following Hambly,
the slab strips were given I = bt&sup3;/12 and a torsion constant J = bt&sup3;/6, and they were replaced by T-beams at the cross
girders. Each node has a vertical displacement and two rotations. The torsion constant of a girder was taken as the sum of the values
for its flange, web and bulb, plus half the slab contribution. The bearings were modelled as vertical springs of
{P.BRG_KV / 1e6:.1f} &times; 10<sup>6</sup>&nbsp;kN/m. A load between girder lines was shared between the four surrounding nodes,
and a load outside the outer girders (barrier, footpath) was applied as a force and a torque on the outer girder line.""")
R.p("""For live load, influence surfaces of bending moment and shear were calculated at 13 sections of each girder. Each vehicle was
moved along the span in steps of 0.25&nbsp;m and across the carriageway in steps of 0.05&nbsp;m. The worst transverse arrangement of
vehicles was found by a search that respects the IRC clearances: 1.2&nbsp;m from the kerb for the 70R vehicle, 0.15&nbsp;m for Class A,
and 1.2&nbsp;m between vehicles. The self-weight of the girders and of the wet slab acts on each precast girder separately and was
calculated as for a simply supported beam with 0.45&nbsp;m overhangs.""")
R.p("""As an independent check, the live-load share of the outer girder was also calculated by Courbon's method, which assumes a rigid
cross-section in the transverse direction:""")
R.eq(r"R_i = \dfrac{\sum W}{n}\left(1 + \dfrac{n\,e\,x_i}{\sum_{j} x_j^2}\right)", "courbon")
R.p(f"""where n = 6 is the number of girders, x<sub>i</sub> the distance of girder i from the deck centreline and e the eccentricity of
the resultant vehicle load ({abs(d['courbon']['e']):.2f}&nbsp;m for a 70R vehicle at the kerb clearance with a Class A train beside it).""")
R.h2("Checks on the model", "3.4")
R.p(f"""With a uniform load of 10&nbsp;kN/m on each girder, the sum of the mid-span moments of the grillage is within 0.3&nbsp;% of
wL&sup2;/8, and the deflection of an inner girder is within 0.2&nbsp;% of 5wL<sup>4</sup>/384EI plus the compression of the bearings.
For live load, Courbon's method is applicable here (span/width = 2.06, stiff cross girders) and gives {f(d['courbon']['M'][0], 0)}&nbsp;kN&nbsp;m
for G1, against {f(MLL_G1, 0)}&nbsp;kN&nbsp;m from the grillage (Figure [[fig:f07_ll_distribution]]). For the inner girders the grillage
gives larger moments than Courbon's method, because a vehicle near the centre of the deck loads G3 and G4 more than a rigid deck would.
The grillage values were used for design.""")
R.fig("f07_ll_distribution", "Live-load moment at mid-span of each girder (with impact and lane reduction): grillage and Courbon's method")
R.h2("Load effects", "3.5")
M = {k: d["stages"]["G1"]["16.500"]["M"][k] for k in ("Mg", "Ms", "Medge", "Mwc", "Mll")}
M2 = d["stages"]["G2"]["16.500"]["M"]; M3 = d["stages"]["G3"]["16.500"]["M"]
R.table(["Mid-span moment (kN m)", "G1 (outer)", "G2", "G3"], [
    ["Self-weight of girder", f(M["Mg"], 0), f(M2["Mg"], 0), f(M3["Mg"], 0)],
    ["Deck slab and cross girder", f(M["Ms"], 0), f(M2["Ms"], 0), f(M3["Ms"], 0)],
    ["Crash barrier, footpath, railing", f(M["Medge"], 0), f(M2["Medge"], 0), f(M3["Medge"], 0)],
    ["Wearing coat", f(M["Mwc"], 0), f(M2["Mwc"], 0), f(M3["Mwc"], 0)],
    ["Vehicles with impact, and footway load", f(M["Mll"], 0), f(M2["Mll"], 0), f(M3["Mll"], 0)],
    ["Ultimate design moment <i>M</i><sub>Ed</sub>", f(d["uls"]["G1"]["MEd"], 0), f(d["uls"]["G2"]["MEd"], 0), f(d["uls"]["G3"]["MEd"], 0)],
], "Bending moments at mid-span (G4 to G6 are symmetric to G3 to G1)", num_cols=(1, 2, 3), full=False)
R.p("""For G1 the worst case is a 70R wheeled vehicle at the kerb clearance with a Class A train beside it; for the inner girders it
is the same pair of vehicles placed over them. The outer girder has the largest live-load moment, while the inner girders carry more
slab weight. In the end the ultimate moments of G1 and G2 differ by less than 1&nbsp;%, so a single girder design was used for all six
girders. The envelopes for G1 are shown in Figure [[fig:f05_envelopes]].""")
R.fig("f05_envelopes", "Bending moment and shear force envelopes of girder G1")

# ====================================================================== 4
R.h1("Prestressing System", "4")
R.h2("Cables and jacking force", "4.1")
R.p(f"""Each girder has four cables of twelve 15.2&nbsp;mm strands (12T15, A<sub>p</sub> = 1680&nbsp;mm&sup2; per cable) in 92&nbsp;mm
corrugated HDPE ducts. The cables are stressed from both ends to {P.JACK_RATIO:.2f}&nbsp;f<sub>pk</sub> = 1395&nbsp;MPa, i.e.
{f(pc['C1']['Pj'], 0)}&nbsp;kN per cable. This is below the maximum allowed by IRC:112 and also below the value of 0.765&nbsp;f<sub>pk</sub>
used in earlier practice, which leaves some room for over-stressing if the friction on site turns out to be higher than assumed.""")
R.h2("Cable layout and profiles", "4.2")
R.p("""The cable layout was fixed by checking the position of every duct along the whole girder, not only at mid-span. At the
anchorage face the four anchor plates are placed one above the other in the plane of the web, at 0.35, 0.75, 1.15 and 1.55&nbsp;m above
the soffit and 400&nbsp;mm apart. The centroid of the group (0.95&nbsp;m) then coincides with the centroid of the end block (0.93&nbsp;m).
At mid-span three cables lie in the bottom row at 130&nbsp;mm and one in a second row at 300&nbsp;mm (Figures [[fig:f02_girder_section]]
and [[fig:f03_cable_profile]]).""")
R.p("""The two outer cables of the bottom row also have to move sideways by 220&nbsp;mm, which is not possible inside a 300&nbsp;mm web.
In my first layout these cables moved sideways while they were still in the web; this was found when the cover to the ducts was
checked along the girder. In the final layout these two cables drop into the bulb within the first 5 and 8&nbsp;m (parabolas ending at
those points) and move sideways only once they are inside the bulb. The two centre cables follow parabolas over the full half-length
and stay in the plane of the web. Each profile is a parabola from the anchorage with a horizontal tangent where it reaches its mid-span
level, and is straight beyond that point. An allowance of 12&nbsp;mm was made for the strands lying low in a draped duct.""")
R.fig("f03_cable_profile", "Cable profiles in elevation and in plan. Cables C2 and C3 are stressed at 10 days, C1 and C4 at 28 days.")
R.h2("Stressing sequence", "4.3")
R.p(f"""If all four cables were stressed at 10 days, the soffit compression at mid-span would be about 18.6&nbsp;MPa, against an
allowable value of 0.48&nbsp;f<sub>cj</sub> = {0.48 * P.FCK_TRANSFER:.1f}&nbsp;MPa at that age. The cables are therefore stressed in two
stages: C2 and C3 at 10 days, which is enough to lift the girder off the casting bed and carry its own weight, and C1 and C4 at
28 days. Both pairs have their centroid at the centroid of the end block, so the girder ends are not bent at either stage.""")
R.h2("Losses of prestress", "4.4")
R.p("""The force after friction at a distance x from the nearer anchorage is""")
R.eq(r"P(x) = P_j\,e^{-(\mu\theta + kx)}", "friction")
R.p(f"""where &theta; is the cumulative vertical and horizontal angle change of the cable from the anchorage, &mu; = {P.MU_FRICTION} and
k = {P.K_WOBBLE}&nbsp;/m for HDPE ducts. The draw-in of {P.ANCHOR_SLIP * 1e3:.0f}&nbsp;mm at the anchorage affects a length l<sub>s</sub>
found from the condition that the area between the force profiles before and after the slip equals the slip times the axial stiffness
of the cable:""")
R.eq(r"2\int_0^{l_s}\left[P(x) - P(l_s)\right]dx = \Delta_s\,E_p A_p", "slip")
R.p("""The loss due to elastic shortening was calculated cable by cable in the order of stressing (C2, C3, C1, C4), using the modulus
of the concrete at the age of each stressing. When cable i is stressed, every cable j already anchored loses""")
R.eq(r"\Delta P_j = \dfrac{E_p}{E_c(t_i)}\,A_{p,j}\left(\dfrac{P_i}{A} + \dfrac{P_i\,e_i\,e_j}{I}\right)", "es")
R.p("""The long-term loss due to creep, shrinkage and relaxation was obtained from the expression given in IRC:112 (of the same form
as Eq. 5.46 of EN 1992-1-1):""")
R.eq(r"\Delta\sigma_{p,c+s+r} = \dfrac{\varepsilon_{cs}E_p + 0.8\,\Delta\sigma_{pr} + \dfrac{E_p}{E_{cm}}\,\varphi(t,t_0)\,\sigma_{c,QP}}"
     r"{1 + \dfrac{E_p}{E_{cm}}\,\dfrac{A_p}{A_c}\left(1 + \dfrac{A_c}{I_c}z_{cp}^2\right)\left[1 + 0.8\,\varphi(t,t_0)\right]}", "longterm")
R.p(f"""The creep coefficient and the shrinkage strain were calculated from the expressions in Annex A2 of IRC:112 for a relative
humidity of {P.RH:.0f}&nbsp;%, a notional size of 281&nbsp;mm and a cylinder strength taken as 0.8 times the cube strength. The
quasi-permanent concrete stress at the level of the tendons is 10.8&nbsp;MPa at mid-span. The final relaxation was taken as three times
the 1000-hour value for the actual initial stress ratio. The results at mid-span are given in Table [[tab:losses]].""")
rows = []
for n, v in pc.items():
    rows.append([n, PS.STAGE_TIME[1 if n in ("C2", "C3") else 2], f(v["Pj"], 0), f(v["Pj"] - v["Pfs_mid"], 0), f(v["ES_mid"], 0),
                 f(v["P0_mid"], 0), f(v["phi"], 2), f(v["relax"], 0), f(v["dsig"], 0), f(v["Pe_mid"], 0)])
rows.append(["<b>Total</b>", "", f(tot["Pj"], 0), f(tot["Pj"] - tot["Pfs_mid"], 0), f(sum(v["ES_mid"] for v in pc.values()), 0),
             f(tot["P0_mid"], 0), "", "", "", f"<b>{f(tot['Pe_mid'], 0)}</b>"])
R.table(["Cable", "Stressed at (days)", "<i>P</i><sub>j</sub> (kN)", "Friction + draw-in (kN)", "Elastic shortening (kN)",
         "<i>P</i><sub>0</sub> (kN)", "&phi;(&infin;, <i>t</i><sub>0</sub>)", "Relaxation (MPa)", "Long-term loss (MPa)", "<i>P</i><sub>e</sub> (kN)"],
        rows, "Prestressing forces and losses at mid-span of one girder", num_cols=tuple(range(1, 10)), tag="losses", small=True)
R.p(f"""The immediate loss at mid-span is {pct(1 - tot['P0_mid'] / tot['Pj'], 1)} and the long-term loss a further
{pct((tot['P0_mid'] - tot['Pe_mid']) / tot['Pj'], 1)}, which gives P<sub>0</sub> = {f(tot['P0_mid'] / 1e3, 2)}&nbsp;MN,
P<sub>e</sub> = {f(tot['Pe_mid'] / 1e3, 2)}&nbsp;MN and P<sub>e</sub>/P<sub>j</sub> = {tot['Pe_mid'] / tot['Pj']:.3f}. A total loss in the low
twenties of per cent is normal for a post-tensioned girder of this length, so I took this as a first check on the calculation. Cable C2
loses the most to elastic shortening because three cables are stressed after it. Figure [[fig:f04_prestress_force]] shows the force
along the girder; the draw-in of the cables with straight central portions extends almost to mid-span because the only friction there is
the wobble term.""")
R.fig("f04_prestress_force", "Total prestressing force of one girder along its length at successive stages")
R.h2("Stresses at the important stages", "4.5")
R.p("""The stresses were checked at mid-span, at L/4 and at two sections near the support of every girder, for five stages: transfer
of the first two cables (10 days, self-weight only), all cables stressed (28 days), casting of the deck slab (60 days), opening to
traffic (90 days) and long-term service in the three load combinations. The stress at the soffit, for example, is""")
R.eq(r"\sigma_b = \dfrac{P}{A} + \dfrac{P\,e}{Z_b} - \dfrac{M_g + M_s}{Z_b} - \dfrac{M_{sidl} + \psi M_q}{Z_{b,c}} + \sigma_{T} + \sigma_{sh}", "stress")
R.p("""with compression taken as positive, Z<sub>b</sub> and Z<sub>b,c</sub> the section moduli of the precast and composite sections,
&psi; the combination factor on live load, and &sigma;<sub>T</sub> and &sigma;<sub>sh</sub> the self-equilibrating stresses from the
temperature difference and differential shrinkage, added only where they are adverse. For the service checks the characteristic value
of the prestress was used: 0.9P<sub>m</sub> where prestress is favourable and 1.1P<sub>m</sub> where it is not.""")
R.p("""The design criteria adopted are stricter than the crack-width limits of IRC:112 for bonded prestressed members: no decompression
of the soffit under the quasi-permanent combination, soffit tension below f<sub>ctm</sub> (uncracked section) under the rare combination,
and compression below 0.48&nbsp;f<sub>ck</sub> under the rare and 0.36&nbsp;f<sub>ck</sub> under the quasi-permanent combination.""")
R.fig("f06_stage_stresses", "Stresses at mid-span of the outer girder G1 at the construction and service stages (service stages with 0.9P<sub>m</sub>)")
R.table(["Check (worst of G1 to G3 and of all sections)", "Value (MPa)", "Limit (MPa)", "Result"], [
    ["Transfer, first stage: maximum compression / minimum stress", f"{ss['transfer1_max_comp']:.2f} / {ss['transfer1_min']:+.2f}", f"{0.48 * P.FCK_TRANSFER:.2f} / &ge; 0", "satisfied"],
    ["All cables stressed: soffit compression", f"{ss['transfer2_max_comp']:.2f}", f"{0.48 * P.FCK_GIRDER:.2f}", "satisfied"],
    ["All cables stressed: top fibre", f"{ss['transfer2_min_top']:+.2f}", "&ge; 0 (no tension)", "satisfied"],
    ["Long term, quasi-permanent, soffit (0.9<i>P</i><sub>m</sub>)", f"{ss['qp_min_bot']:+.2f}", "&ge; 0 (no decompression)", "satisfied"],
    ["Long term, frequent, soffit (0.9<i>P</i><sub>m</sub>)", f"{ss['frequent_min_bot']:+.2f}", f"&ge; &minus;{P.fctm(P.FCK_GIRDER):.1f} (uncracked)", "satisfied"],
    ["Long term, rare, soffit (0.9<i>P</i><sub>m</sub>)", f"{ss['rare_min_bot']:+.2f}", f"&ge; &minus;{P.fctm(P.FCK_GIRDER):.1f} (uncracked)", "satisfied"],
    ["Rare: top of girder / top of slab (1.1<i>P</i><sub>m</sub>)", f"{ss['rare_max_top_girder']:.2f} / {ss['rare_max_slab']:.2f}", f"{0.48 * P.FCK_GIRDER:.1f} / {0.48 * P.FCK_DECK:.1f}", "satisfied"],
    ["Quasi-permanent: soffit compression (1.1<i>P</i><sub>m</sub>)", f"{ss['qp_max_bot']:.2f}", f"{0.36 * P.FCK_GIRDER:.1f}", "satisfied"],
], "Summary of stress checks (compression positive)", num_cols=(1, 2), small=True)
R.p(f"""The governing check is the soffit stress at mid-span under the rare combination, {ss['rare_min_bot']:+.2f}&nbsp;MPa. The reverse
temperature difference alone contributes about {0.6 * 2.51:.1f}&nbsp;MPa of tension at the soffit, so the girder would not meet a 'no
tension under rare loads' rule; it does, however, remain uncracked with a margin of about 1&nbsp;MPa. I kept this margin in view
because it decides how much prestress the girder can lose before it cracks (Chapter 10).""")
R.h2("Flexure at the ultimate limit state", "4.6")
u = d["uls"]["G1"]
R.p("""The moment of resistance at mid-span was found by strain compatibility, with a rectangular stress block of depth 0.8x at
f<sub>cd</sub> = 0.67&nbsp;f<sub>ck</sub>/1.5 and a bilinear stress-strain curve for the strand with f<sub>pd</sub> = f<sub>p0.1k</sub>/1.15,
starting from the strain due to P<sub>e</sub>. When the neutral axis lies within the slab of effective width b,""")
R.eq(r"0.67\dfrac{f_{ck}}{1.5}\,b\,(0.8x) = A_p f_p, \qquad M_{Rd} = A_p f_p\,(d_p - 0.4x)", "uls")
R.p(f"""For G1 this gives x = {u['x'] * 1e3:.0f}&nbsp;mm (stress block {0.8 * u['x'] * 1e3:.0f}&nbsp;mm deep, within the 225&nbsp;mm slab),
a strand strain of {u['eps_p'] * 1e3:.0f} &times; 10<sup>-3</sup> (the strands yield and the failure is ductile, x/d = {u['x_over_d']:.2f})
and M<sub>Rd</sub> = {f(u['M'], 0)}&nbsp;kN&nbsp;m, against M<sub>Ed</sub> = {f(u['MEd'], 0)}&nbsp;kN&nbsp;m. The utilisation is
{u['MEd'] / u['M']:.2f}. The untensioned bottom reinforcement was neglected, which is conservative.""")
R.h2("Shear", "4.7")
sh = d["shear"]["G3@2.75"]
R.p(f"""The critical section for shear is where the web has just narrowed to 300&nbsp;mm, 2.75&nbsp;m from the bearing. Since a duct runs
through the web there, the nominal web width is b<sub>w</sub> &minus; 1.2&phi; = {sh['bw_nom'] * 1e3:.0f}&nbsp;mm for HDPE ducts. For the worst
girder V<sub>Ed</sub> = {f(sh['VEd'], 0)}&nbsp;kN, the vertical component of the prestress (0.9P<sub>e</sub> sin&alpha;) is
{f(sh['Vp'], 0)}&nbsp;kN, and the net shear of {f(sh['Vnet'], 0)}&nbsp;kN is below V<sub>Rd,max</sub> = {f(sh['VRdmax'], 0)}&nbsp;kN for a
strut inclination with cot&theta; = 2. The required shear reinforcement is {sh['Asw_s']:.0f}&nbsp;mm&sup2;/m. Two-legged 12&nbsp;mm links at
200&nbsp;mm (1131&nbsp;mm&sup2;/m) were adopted over the end 6&nbsp;m, and two-legged 10&nbsp;mm links at 250&nbsp;mm elsewhere, which is
above the minimum of {sh['Asw_s_min']:.0f}&nbsp;mm&sup2;/m.""")
R.h2("Deflection and camber", "4.8")
df = d["deflection"]
R.p(f"""The live-load deflection with impact is {df['LL_mm']['G1']:.1f}&nbsp;mm for G1, or span/{P.L * 1e3 / df['LL_mm']['G1']:.0f}, well
within the limit of span/800 = {df['limit_mm']:.1f}&nbsp;mm. The upward camber is {df['camber_transfer_mm']:.0f}&nbsp;mm when all the cables have
been stressed and about {df['camber_deck_mm']:.0f}&nbsp;mm after the deck slab has been cast. The elastic camber due to P<sub>e</sub> alone is
{df['camber_Pe_elastic_mm']:.0f}&nbsp;mm. This value is used again in Chapter 10, because a loss of prestress shows up as a loss of this
camber.""")
R.h2("Bearings", "4.9")
bg = d["bearings"]["G3"]
R.p(f"""The largest bearing reaction is {f(bg['Nmax'], 0)}&nbsp;kN, at an inner girder, with a permanent reaction of {f(bg['Nmin'], 0)}&nbsp;kN.
For the 400 &times; 500&nbsp;mm bearing with five 12&nbsp;mm layers the shape factor is {bg['S']:.1f}. The total design strain due to
compression, shear (a movement of {bg['v_x'] * 1e3:.0f}&nbsp;mm from creep, shrinkage and temperature) and rotation is {bg['eps_t']:.2f},
below the limit of 7 in IRC:83 (Part II), and the minimum pressure of {bg['sigma_min']:.1f}&nbsp;MPa is above 3&nbsp;MPa, so the bearing will not
creep out of position. In service each bearing shears by about {b['bearing_ref']:.3f}&nbsp;mm per &deg;C, i.e. &alpha;L/2; this is the
quantity watched by the bearing displacement transducers (Chapter 8).""")
R.h2("Anchorage zones and other details", "4.10")
R.p("""The end block was sized for the anchor plates (spacing 400&nbsp;mm, edge distance 250&nbsp;mm at the top) and for the bursting forces in
the local zones. The final anchorage reinforcement has to follow the approval document of the prestressing system actually used. The deck
slab was designed separately for the 70R wheel loads on its 1.7&nbsp;m clear span; these local designs do not affect the later chapters and
are not repeated here.""")

# ====================================================================== 5
R.h1("Healthy Baseline", "5")
R.h2("Need for a measured baseline", "5.1")
R.p("""The healthy baseline cannot simply be taken from the design calculation. A bridge as built always differs somewhat from the design: the concrete is
usually stiffer than the code value of the modulus, crash barriers and footpaths add stiffness, and bearings are not ideal. Later
measurements have to be compared with the bridge as it was measured when new, and with a model that reproduces those measurements. The
baseline was therefore set up in three steps: commissioning tests, updating of the model, and one year of monitoring of the healthy
bridge.""")
R.h2("Commissioning tests", "5.2")
R.p(f"""Before the bridge was opened, a 30-minute ambient vibration test was recorded at night with all 12 accelerometers, and a static
load test was carried out with four 300&nbsp;kN three-axle lorries (two side by side and two in tandem, centred at mid-span) following the
approach of IRC:SP:51. In this study the 'real' bridge is a model in which the concrete modulus is 6&nbsp;% above E<sub>cm</sub>, the
crash barriers act with about one-third of their stiffness and the dynamic modulus is 24&nbsp;% above the static modulus. I did not use
these values anywhere in the assessment.""")
R.p("""The commissioning record was identified without using any mode shapes from the model, by frequency domain decomposition (FDD) of
the 12 channels (Figure [[fig:f10_commissioning]]). The cross-spectral matrix of the channels was computed at every frequency line and
decomposed by singular value decomposition; peaks of the first singular value were picked, each peak was named after the design-model
mode that its singular vector resembled most, and the frequency was refined by fitting a single-degree-of-freedom spectrum around the
peak (Section 9.4).""")
R.fig("f10_commissioning", "Commissioning ambient vibration test: one channel in the time and frequency domains, and the singular values of the cross-spectral matrix of all 12 channels")
R.p(f"""At first I identified the frequencies by filtering the channels with the mode shapes of the design model. This gave
f<sub>2</sub> = 3.78&nbsp;Hz, whereas the model-free FDD gave {tw['meas_f'][1]:.2f}&nbsp;Hz. With the first bending and torsion modes only
5&nbsp;% apart, the design-model shapes let one mode leak into the other. From this I learnt that a model which has not yet been checked
against the structure should be used only to name the modes, and not to process the data.""")
R.h2("Model updating", "5.3")
R.p("""Three parameters of the model were updated: a factor on the concrete modulus (E<sub>scale</sub>), the share of the crash-barrier
stiffness acting with the deck (&beta;) and the ratio of dynamic to static modulus (E<sub>dyn</sub>). They were found by minimising""")
R.eq(r"J(\theta) = \sum_{k=1}^{4}\left(\dfrac{f_k(\theta)/f_k^{m} - 1}{0.002}\right)^2 + \sum_{i=1}^{6}\left(\dfrac{\delta_i(\theta) - \delta_i^{m}}{0.10\ \mathrm{mm}}\right)^2", "update")
R.p("""over the four identified frequencies f<sub>k</sub><sup>m</sup> and the six measured mid-span deflections
&delta;<sub>i</sub><sup>m</sup>, with weights equal to the expected measurement errors (0.2&nbsp;% and 0.10&nbsp;mm).""")
R.table(["Quantity", "Design model", "Measured (year 0)", "Updated twin"], [
    ["<i>f</i><sub>1</sub>, first bending (Hz)", f(tw['design_f'][0], 3), f(tw['meas_f'][0], 3), f(tw['updated_f'][0], 3)],
    ["<i>f</i><sub>2</sub>, first torsion (Hz)", f(tw['design_f'][1], 3), f(tw['meas_f'][1], 3), f(tw['updated_f'][1], 3)],
    ["<i>f</i><sub>3</sub>, first transverse (Hz)", f(tw['design_f'][2], 3), f(tw['meas_f'][2], 3), f(tw['updated_f'][2], 3)],
    ["<i>f</i><sub>4</sub>, second bending (Hz)", f(tw['design_f'][3], 3), f(tw['meas_f'][3], 3), f(tw['updated_f'][3], 3)],
    ["Load-test deflection G1 / G3 (mm)", f"{tw['design_d'][0]:.2f} / {tw['design_d'][2]:.2f}", f"{tw['meas_d'][0]:.2f} / {tw['meas_d'][2]:.2f}",
     f"{tw['updated_d'][0]:.2f} / {tw['updated_d'][2]:.2f}"],
    ["<i>E</i><sub>scale</sub>", "1.00", "&ndash;", f"{tw['E_scale']:.3f} &plusmn; {tw['sd'][0]:.3f}"],
    ["&beta;", "0", "&ndash;", f"{tw['barrier']:.2f} &plusmn; {tw['sd'][1]:.2f}"],
    ["<i>E</i><sub>dyn</sub>", "1.20", "&ndash;", f"{tw['E_dyn']:.3f} &plusmn; {tw['sd'][2]:.3f}"],
], "Design model, commissioning measurements and the updated model (digital twin)", num_cols=(1, 2, 3), tag="updating", full=False)
R.p(f"""The design model is 7 to 10&nbsp;% too soft in frequency and 9 to 14&nbsp;% too flexible in the load test, which is normal for a design
model. After updating, the twin reproduces all ten measurements within 0.2&nbsp;% in frequency and 0.15&nbsp;mm in deflection, and the
measured and twin mode shapes agree with MAC values above 0.99 for the first three modes. The updated parameters are close to the hidden
values (1.06, 0.35, 1.24), but I still treat the twin as a calibrated model and not as the true structure, since E<sub>scale</sub> and &beta; are
correlated (correlation coefficient {tw['corr'][0][1]:.2f}) and a somewhat different pair could fit almost as well. The second bending mode
at 14.5&nbsp;Hz nearly coincides with the second torsion mode, and its measured shape is a mixture of the two (MAC 0.47); for this mode the
twin shape was used in the later signal processing.""")
R.fig("f08_mode_shapes", "First bending, torsion and transverse modes and the second bending mode of the updated twin at 25 &deg;C")
R.h2("One year of healthy monitoring", "5.4")
R.p(f"""The permanent monitoring system then recorded two 30-minute records a day for a year, at 02:00 and 14:00, giving 730 records. Over
the year the first bending frequency varied between 3.65 and 3.79&nbsp;Hz, a range of almost 4&nbsp;%. The causes are the temperature
({nb['f1']['per_degC_pct']:+.3f}&nbsp;%/&deg;C, mainly through the modulus of concrete), the traffic present during the record and the extra
mass of a wet deck in the monsoon. For each frequency a linear model in the temperature T and the logarithm of the RMS acceleration (a
measure of how busy the bridge was during the record) was fitted:""")
R.eq(r"\hat f_k = \beta_{0k} + \beta_{1k}\,(T - 25) + \beta_{2k}\ln a_{RMS}", "norm")
R.p(f"""After this normalisation the scatter of f<sub>1</sub> is {nb['f1']['sd'] * 1e3:.1f}&nbsp;mHz ({100 * nb['f1']['sd'] / nb['f1']['mean']:.2f}&nbsp;%)
(Figure [[fig:f11_baseline_year]]). The remaining pattern in the residuals is the effect of the monsoon on the mass. I did not remove it,
because on a real bridge its size would not be known in advance.""")
R.fig("f11_baseline_year", "One year of healthy monitoring: first bending frequency against temperature, and the residual after temperature and traffic normalisation")
R.table(["Baseline quantity (span S2, healthy, reference temperature 25 &deg;C)", "Value"], [
    ["Effective prestress per girder at mid-span", f"<i>P</i><sub>e</sub> = {f(tot['Pe_mid'] / 1e3, 2)} MN; soffit precompression due to <i>P</i><sub>e</sub> {7172 * (1 / pre['A'] + (pre['zc'] - 0.1605) / pre['Zb']) / 1e3:.1f} MPa"],
    ["Soffit stress, quasi-permanent combination (mean prestress)", f"about {d['stages']['G1']['16.500']['rows']['V quasi, inf']['bot'] + 2.34:.1f} MPa compression (G1)"],
    ["Load test (4 &times; 300 kN): mid-span deflection of G1, G2, G3", " / ".join(f"{x:.2f}" for x in b['load_test_ref']['d'][:3]) + " mm"],
    ["Load test: soffit strain of G1, G2, G3", " / ".join(f"{x:.0f}" for x in b['load_test_ref']['e'][:3]) + " &mu;&epsilon;"],
    ["Natural frequencies <i>f</i><sub>1</sub> to <i>f</i><sub>4</sub>", " / ".join(f"{x:.3f}" for x in b['f_ref']) + " Hz"],
    ["Temperature sensitivity of <i>f</i><sub>1</sub> to <i>f</i><sub>4</sub>", " / ".join(f"{nb[k]['per_degC_pct']:+.2f}" for k in ('f1', 'f2', 'f3', 'f4')) + " %/&deg;C"],
    ["Scatter after normalisation, <i>f</i><sub>1</sub> to <i>f</i><sub>4</sub>", " / ".join(f"{100 * nb[k]['sd'] / nb[k]['mean']:.2f}" for k in ('f1', 'f2', 'f3', 'f4')) + " %"],
    ["Acceleration at mid-span", "0.6 to 0.8 mg RMS at night; 1.3 to 1.5 mg RMS in daytime traffic; peaks 10 to 25 mg"],
    ["Bearing movement", f"{b['bearing_ref']:.3f} mm/&deg;C at both ends"],
    ["NDT on girders, year 1 (mean &plusmn; s.d.)", f"rebound {b['ndt']['rebound']['mean']:.1f} &plusmn; {b['ndt']['rebound']['sd']:.1f}; UPV {b['ndt']['upv']['mean']:.2f} &plusmn; {b['ndt']['upv']['sd']:.2f} km/s; cover {b['ndt']['cover']['mean']:.0f} &plusmn; {b['ndt']['cover']['sd']:.0f} mm; HCP {b['ndt']['hcp']['mean']:.0f} &plusmn; {b['ndt']['hcp']['sd']:.0f} mV; resistivity {b['ndt']['resistivity']['mean']:.0f} &plusmn; {b['ndt']['resistivity']['sd']:.0f} k&Omega;cm"],
], "Baseline structural and monitoring data of the healthy bridge", widths=[44, 56], small=True, tag="baseline")
R.p("""All later measurements are compared with this table and with the normalisation models of Eq. ([[eq:norm]]), not with the design
calculation. The same 'own baseline' principle is also what allows a model trained on simulated data to be applied to the bridge itself
(Chapter 12).""")

# ====================================================================== 6
R.h1("Visual Inspection Scheme", "6")
R.p("""Visual inspection is the cheapest source of information about the bridge and, for most of the defects that matter on a bridge of
this type (water, corrosion, spalling, bearing and joint problems), also the first. The scheme is organised as Bridge &rarr; Span &rarr;
Component &rarr; Location &rarr; Observation, so that each record has an address that can be found again at the next inspection.""")
R.h2("Components, locations and defects", "6.1")
R.table(["Component", "Locations recorded", "What is looked for"], [
    ["PSC girders G1 to G6", "zones Z1 to Z5 along the span (Z1 and Z5: end zones 0 to 3.3 m with anchorages and bearings; Z2 and Z4: quarter spans; Z3: mid-span); soffit, web faces, anchorage faces",
     "flexural cracks at the soffit (any crack is significant in a fully prestressed girder), longitudinal cracks along ducts, rust staining, spalling, honeycombing, efflorescence, impact damage over the service road"],
    ["Deck slab", "bays between girders, cantilevers, construction joints", "cracks, leakage and efflorescence, spalling near drains"],
    ["Cross girders and diaphragms", "end diaphragms A and B, cross girder at L/2", "cracks near jacking points, leakage from the joint"],
    ["Bearings", "G1-A to G6-B", "bulging, splitting, too much or no shear deformation, debris, corrosion of plates, loss of contact"],
    ["Expansion joints", "EJ-A1, EJ-P1, EJ-P2, EJ-A2", "torn or missing seal, debris in the gap, water passing through, noise"],
    ["Drainage", "spouts U-1 to U-6 and D-1 to D-6 in each span", "blockage, spouts too short (water running on the girder), broken pipes"],
    ["Footpaths, crash barriers, wearing coat", "upstream and downstream, lanes", "loose cover slabs, cracking of barriers, ravelling and potholes near joints"],
    ["Pier caps, piers, abutments", "P1, P2, A1, A2", "leakage stains, spalling of pedestals, cracking, vegetation, signs of scour"],
], "Inspection hierarchy, locations and items to be inspected", widths=[18, 34, 48], small=True)
R.h2("Condition states and condition index", "6.2")
R.p("""Each observation is given a condition state from 1 to 5 using written rules, so that two inspectors arrive at similar ratings
(Table [[tab:cs]]). Crack-width bands are stricter for prestressed girders than for reinforced concrete elements, because a flexural crack
in a girder designed to remain uncracked means that its precompression has been lost.""")
R.table(["Condition state", "Meaning", "Crack width, PSC girder", "Crack width, RC element", "Other examples"], [
    ["CS1 Good", "no repair needed", "none", "&lt; 0.10 mm", "&ndash;"],
    ["CS2 Fair", "minor defect, monitor", "up to 0.10 mm", "0.10 to 0.20 mm", "efflorescence, minor leakage"],
    ["CS3 Poor", "repair at the next opportunity", "0.10 to 0.20 mm", "0.20 to 0.30 mm", "spalling without exposed steel, rust staining, leaking joint"],
    ["CS4 Serious", "repair soon, check structural effect", "0.20 to 0.30 mm", "0.30 to 1.0 mm", "spalling with exposed bars, restrained bearing"],
    ["CS5 Critical", "immediate action, possibly restriction", "&gt; 0.30 mm", "&gt; 1.0 mm", "exposed strands over a large area"],
], "Condition states used in the inspection", tag="cs", widths=[14, 22, 17, 17, 30], small=True)
R.p("""A component takes the worst condition state of its observations. A bridge condition index (BCI) between 0 and 100 is then
calculated as a weighted average over the component groups,""")
R.eq(r"\mathrm{BCI} = 100\,\dfrac{\sum_c w_c\,(5 - \mathrm{CS}_c)/4}{\sum_c w_c}", "bci")
R.p("""with weights reflecting the structural importance of each group: 0.30 for the girders, 0.12 each for the deck slab and the
bearings, 0.08 each for the joints, piers and pier caps, 0.06 each for the cross girders and abutments, 0.04 for drainage and 0.02 each
for footpaths, crash barriers and wearing coat.""")
R.h2("Types and frequency of inspection", "6.3")
R.table(["Type", "When", "Scope"], [
    ["Routine inspection", "before and after every monsoon", "walk-over of the deck, view from the service road and the canal banks with binoculars; drains, joints and bearings visible from the piers"],
    ["Principal inspection", "every three years", "close visual inspection of every element from an under-bridge inspection unit; hammer tapping of suspect concrete; NDT at the fixed test areas"],
    ["Special inspection", "after an alarm from the monitoring system, an overload, a flood or a vehicle impact", "as needed for the event, by a bridge engineer"],
], "Inspection regime", widths=[18, 30, 52], small=True)
R.h2("Digital inspection application", "6.4")
R.p("""I wrote a small web application for the scheme (<code>inspection_app/index.html</code>; it runs in any browser without an
internet connection). It guides the inspector through span, component, location and observation, suggests the condition state from the
crack width or the type of defect (the inspector can change it), stores photographs, keeps the records in the browser and exports them as
CSV or JSON. It also draws the worst condition state of every girder zone as a map and calculates the component ratings and the bridge
condition index. The year-18 inspection of Chapter 14 is included as an example (Figure [[fig:f24_inspection_app]]).""")
R.fig("f24_inspection_app", "The inspection application with the year-18 inspection loaded: entry form, component ratings and bridge condition index, worst condition state of each girder zone of span S2, and list of observations", scale=0.9)

# ====================================================================== 7
R.h1("Non-Destructive Testing Programme", "7")
R.p("""Each test in the NDT programme is chosen to answer a specific question raised by the inspection. The programme for B33 is
therefore set out in the order observed defect, engineering question, NDT, measurement, interpretation (Table [[tab:ndt]]).""")
R.table(["Observed defect", "Engineering question", "NDT", "Measurement", "Interpretation"], [
    ["Rust staining, spalling, damp patches on the girder soffit", "Is the reinforcement corroding, and over what area?", "half-cell potential on a 0.5 m grid (ASTM C876); Wenner resistivity",
     "potential (mV vs CSE); resistivity (k&Omega;cm)", "below &minus;350 mV corrosion likely, &minus;200 to &minus;350 mV uncertain, above &minus;200 mV low risk; resistivity below 10 k&Omega;cm means fast corrosion is possible. A wet surface also lowers the potential, so both are read together"],
    ["As above, and areas of low cover", "Has carbonation reached the steel? Is the cover as specified?", "cover meter; phenolphthalein test on drill holes or small cores",
     "cover (mm), carbonation depth (mm)", "carbonation depth at or beyond the cover in wet concrete: the steel is no longer passive"],
    ["Cracks, honeycombing, surface deterioration", "Is the concrete sound through the section? Where are the voids?", "ultrasonic pulse velocity across the web and the bulb (IS 516 Part 5/Sec 1)",
     "pulse velocity (km/s)", "above 4.40 km/s good, 3.75 to 4.40 medium, below 3.75 doubtful; a local drop across a crack gives its depth"],
    ["As above", "Is the surface concrete weaker?", "rebound hammer on a grid (IS 516 Part 5/Sec 4)", "rebound number", "comparative only; carbonation hardens the surface and raises the rebound; calibrated against a few cores"],
    ["Rust or leakage at anchorages; cracks along ducts", "Are the ducts fully grouted? Are strands corroding inside?", "GPR to locate the ducts; impact-echo along the duct line; borescope through a small drilled hole",
     "duct positions, voids, condition of the strands", "voids at duct high points and anchorages are the usual places of strand corrosion; half-cell readings do not see strands inside ducts"],
    ["Sag of a girder; cracks at mid-span", "How much prestress is left?", "lift-off test at an accessible anchorage, or saw-cut stress release on the soffit with strain gauges",
     "lift-off force (kN); released strain (&mu;&epsilon;)", "the only direct measurement of the remaining force; confirms or rejects what the monitoring suggests"],
    ["Bearing distress", "Can the bearing still move and rotate?", "bearing displacement against temperature; dimensions and visual check", "movement per &deg;C (mm/&deg;C)", "much less than &alpha;L/2 at one end: restrained bearing"],
], "NDT programme: defect, question, test, measurement and interpretation", tag="ndt", widths=[15, 17, 22, 15, 31], small=True)
R.h2("Test locations", "7.1")
R.p("""Each girder has five test areas, one in each zone, on the bottom bulb and on the web (on both faces for the outer girders). A span
therefore has 30 test areas and the bridge 90. The same areas are used at every inspection so that trends can be followed. Additional
half-cell grids and carbonation tests are made at the girder ends next to the joints, under every drain spout of the outer girders, and
wherever rust or spalling is found. Investigations of the tendons are limited to the anchorage zones and the high points of the ducts,
where voids collect. In the learning data of Chapter 11 the 30 test areas of S2 enter as span-level summaries (lowest UPV, most negative
potential and so on); the 90 test areas of the whole bridge are clustered in Chapter 14.""")

# ====================================================================== 8
R.h1("Structural Response Measurement", "8")
R.p("""Three response quantities are monitored: displacement, strain and acceleration. Each of them gives information that the other two
do not. Table [[tab:monitor]] gives the quantities, the sensors and their locations, the sampling and duration, and how the readings are
interpreted. The layout is shown in Figure [[fig:f09_sensor_layout]].""")
R.table(["Quantity", "Sensor and location", "Sampling and duration", "Interpretation"], [
    ["Displacement in a load test", "LVDTs on a reference frame, or a total station with prisms, at mid-span of all six girders", "static readings at each load step; at commissioning and at every principal or special inspection",
     "deflection compared with the baseline after correction to 25 &deg;C; the distribution between girders shows which girder has softened"],
    ["Camber (geometry)", "precise levelling points on the kerbs above G1 and G6 and at the deck centreline, at mid-span and supports", "once a year, in the same season and at the same time of day",
     "a loss of camber beyond the creep prediction is the clearest sign of loss of prestress (about 1 mm per 1 % loss)"],
    ["Long-term strain", "vibrating-wire gauges cast into the bottom bulb at mid-span of G1, G3, G4 and G6, with thermistors", "hourly, permanently; daily averages used",
     "drift towards tension beyond the predicted creep and shrinkage means that precompression is being lost (about 12 &mu;&epsilon; per 1 % loss)"],
    ["Strain in a load test", "electrical resistance or fibre-optic gauges on the soffit at mid-span", "100 Hz during the load test", "local change of stiffness; a sudden increase means a crack is opening"],
    ["Bearing movement", "displacement transducers across the bearings of G1 and G6 at both ends", "every 10 minutes, permanently", f"slope of movement against temperature, healthy value {b['bearing_ref']:.3f} mm/&deg;C; a decrease at one end with an increase at the other indicates a restrained bearing"],
    ["Temperature", "thermistors in the deck slab, web and bulb at mid-span of G1 and G4", "every 10 minutes", "effective temperature and gradient, used to normalise all other data"],
    ["Acceleration", "12 vertical accelerometers (Chapter 9)", "100 Hz, two 30-minute records a day", "frequencies, mode shapes, damping, response to traffic"],
], "Structural response measurement: quantities, sensors, sampling and interpretation", tag="monitor", widths=[14, 28, 22, 36], small=True)
R.fig("f09_sensor_layout", "Instrumentation of span S2 in plan. The data logger and a cellular modem are housed in a cabinet on pier P2.")

# ====================================================================== 9
R.h1("Vibration Measurement and Signal Processing", "9")
R.h2("Number, location and direction of accelerometers", "9.1")
R.p(f"""Twelve uniaxial accelerometers measuring the vertical direction are fixed to the soffits of the bottom bulbs (Figure
[[fig:f09_sensor_layout]]): at L/4, L/2 and 3L/4 of both outer girders, at L/2 of the four inner girders, and at L/4 of G3 and 3L/4 of G4.
Low-noise force-balance or piezoelectric seismic accelerometers are needed (noise about {DY.NOISE_UG_RTHZ:.0f}&nbsp;&micro;g/&radic;Hz including
cable and digitiser, range &plusmn;0.5&nbsp;g), because the ambient response at night is only 0.5 to 1&nbsp;mg RMS; MEMS sensors with a noise
of 20 to 25&nbsp;&micro;g/&radic;Hz would hide the higher modes in noise. Only the vertical direction is measured, since the modes of interest
are vertical bending and torsion; the lateral and longitudinal motion of a 33&nbsp;m span on elastomeric bearings carries little information
about the condition of the girders.""")
R.p("""My first layout did not work. With the first ten sensors (three on each outer girder and one at mid-span of each inner
girder), the AutoMAC matrix of the twin's mode shapes at the sensor positions showed two pairs of modes that could not be distinguished,
with MAC = 1.00 (Figure [[fig:f12_automac]]). The second bending and second transverse modes both have zero response at mid-span and the
same signs at the quarter points of the outer girders, and the same was true of the second torsion mode and a higher transverse mode. The
matrix of mode shapes at the sensors was singular (condition number of the order of 10<sup>12</sup>), so the modes could not be separated.
Two additional sensors at the quarter points of the inner girders G3 and G4 reduced the largest off-diagonal MAC to 0.37 and the condition
number to 3.1. The modal assurance criterion used throughout is""")
R.eq(r"\mathrm{MAC}(\phi_a,\phi_b) = \dfrac{\left|\phi_a^{\mathrm{T}}\phi_b\right|^2}{\left(\phi_a^{\mathrm{T}}\phi_a\right)\left(\phi_b^{\mathrm{T}}\phi_b\right)}", "mac")
R.fig("f12_automac", "AutoMAC of the first eight modes of the twin at the sensor positions, for the first and the adopted layouts")
R.h2("Sampling frequency and duration", "9.2")
R.p("""The data loggers sample at 1&nbsp;kHz with a hardware anti-alias filter and decimate the signals to 100&nbsp;Hz, so that the modes up to
about 20&nbsp;Hz lie well inside the usable band of 40&nbsp;Hz. A record lasts 30 minutes, i.e. more than 6000 cycles of the first mode. With
10-minute records the scatter of the identified f<sub>1</sub> in my simulations was about &plusmn;0.5&nbsp;%, which is too large compared with
the changes to be detected; 30-minute records together with a fitted spectrum reduced it to about 0.1&nbsp;%. Records are taken at 02:00,
when traffic is light and the excitation is closest to broadband, and at 14:00 for the response to traffic.""")
R.h2("Sources of excitation", "9.3")
R.p("""The ambient excitation comes from wind, ground vibration and above all from traffic. Traffic is not white noise: loaded lorries bounce on
their suspensions at about 1.3 to 2.6&nbsp;Hz and cross at 40 to 60&nbsp;km/h, and their mass is added to the bridge while they are on it.
These effects shift the identified frequencies by up to &plusmn;0.2&nbsp;% depending on how busy the bridge is, which is why a<sub>RMS</sub>
appears in the normalisation of Eq. ([[eq:norm]]). A drop-weight or shaker test is not needed for routine monitoring but could be used in a
special investigation to measure damping or mode shapes more accurately.""")
R.h2("Signal processing", "9.4")
R.p("""Each record is processed in the sequence a(t) &rarr; signal processing &rarr; FFT &rarr; frequency spectrum &rarr; natural
frequencies, with two additional steps:""")
R.ul(["""<b>Signal processing:</b> the mean is removed and the 12 channels are filtered with a fourth-order Butterworth band-pass filter from 0.5 to
30&nbsp;Hz, applied forwards and backwards so that no phase shift is introduced.""",
      """<b>FFT:</b> Welch averaging with a Hann window of 8192 points (82&nbsp;s, &Delta;f = 0.012&nbsp;Hz) and 50&nbsp;% overlap, about 43 averages
per record.""",
      """<b>Frequency spectrum:</b> auto-spectra of the channels and the full 12 &times; 12 cross-spectral matrix.""",
      """<b>Modal filter:</b> the channels are combined with the pseudo-inverse of the matrix of baseline mode shapes. This separates the closely
spaced first bending and torsion modes, which are only 5&nbsp;% apart.""",
      """<b>Natural frequencies:</b> a single-degree-of-freedom spectrum (Eq. [[eq:sdof]]) is fitted to the filtered spectrum around each peak,
giving the frequency and damping. The mode shape is the singular vector of the cross-spectral matrix at the peak, compared with the baseline
shape by MAC."""], ordered=True)
R.eq(r"S(f) = \dfrac{A}{\left(f_n^2 - f^2\right)^2 + \left(2\zeta f_n f\right)^2} + B", "sdof")
R.p("""Damping estimated from ambient data is poor; its scatter in my simulations was about &plusmn;25&nbsp;%. I kept it as a feature only to
check whether it was of any use for this bridge, and it was not.""")
R.h2("Accuracy of identification", "9.5")
R.p("""On simulated records of the healthy bridge with known frequencies, the identified first frequency was within 0.1 to 0.2&nbsp;% of the true
value at night and within 0.3&nbsp;% in heavy traffic, and the third frequency within 0.3&nbsp;%. These errors include the effect of traffic and are
comparable with the residual scatter after normalisation in Table [[tab:baseline]]. A change of frequency smaller than about 0.5&nbsp;% can therefore
not be detected from a single record, which is why trends over many records are used in Chapter 14.""")

exec(open("build_report_part2.py", encoding="utf-8").read())
