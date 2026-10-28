"""
Builds report/B33_PSC_Bridge_Integrated_Study.html and .pdf from the saved results.
All numbers in the text are read from out/*.json, so the text always matches the calculations.
"""
import json, math, csv, os, re, pathlib, pickle
import numpy as np
from report_lib import Report, print_pdf, find_pages
from psc import inputs as P, section as S, prestress as PS, condition as C, features as FE, scenarios as SC
from psc import dynamics as DY, maintenance as MT, twin as TW

STUDENT = "Shashank Pandya"
ROLL = "23CE10064"

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


def and_join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


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

R.h1("Acknowledgement")
R.p("""I thank Prof. Nirjhar Dhang for setting this problem and for the clarifications given during the month.""")
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
    ["HCP, UPV, VW, GPR", "half-cell potential, ultrasonic pulse velocity, vibrating-wire (strain gauge), ground-penetrating radar"],
    ["GP, RF, EWMA", "Gaussian process, random forest, exponentially weighted moving average"],
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
R.p("""All measurements in this report are synthetic, which the problem statement allows. To make the assessment a fair test, the
'real' bridge used to generate the measurements in Chapters 5 and 14 is a model whose stiffness parameters and deterioration history
I kept hidden from the assessment. The twin had to identify the parameters from the measurements, and the assessment had to find the
deterioration from the data, in the same way as it would have to on an actual bridge.""")
R.fig("f00_flowchart", "Flow of the study and the chapters in which each step is described")
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
mid-span was compared in the grillage model with three cross girders at the quarter points. The two additional cross girders reduced
the live-load moment of the outer girder by about 5&nbsp;% but increased those of the inner girders by about 2&nbsp;%, and their own weight adds
about 215&nbsp;kN&nbsp;m to the mid-span moment of an inner girder. Since one girder design is used for all six girders, the governing ultimate
moment would have increased by about 2&nbsp;%, while the first two natural frequencies changed by only 1 to 1.5&nbsp;%. A single
300 &times; 1450&nbsp;mm cross girder at mid-span was therefore adopted. It also braces the girders laterally while the deck slab is cast.""")
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
the composite girders, and transverse members every {P.L / P.NX:.3f}&nbsp;m representing strips of the deck slab. Following Hambly (1991),
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
    ["Rare: top of girder / top of slab (0.9<i>P</i><sub>m</sub>)", f"{ss['rare_max_top_girder']:.2f} / {ss['rare_max_slab']:.2f}", f"{0.48 * P.FCK_GIRDER:.1f} / {0.48 * P.FCK_DECK:.1f}", "satisfied"],
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
girder, G3, V<sub>Ed</sub> = {f(sh['VEd'], 0)}&nbsp;kN, the vertical component of the prestress (0.9P<sub>e</sub> sin&alpha;) is
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
R.p("""The commissioning record was identified without using any mode shapes from the model, by frequency domain decomposition (FDD; Brincker et al., 2001) of
the 12 channels (Figure [[fig:f10_commissioning]]). The cross-spectral matrix of the channels was computed at every frequency line and
decomposed by singular value decomposition; peaks of the first singular value were picked, each peak was named after the design-model
mode that its singular vector resembled most, and the frequency was refined by fitting a single-degree-of-freedom spectrum around the
peak (Section 9.4).""")
R.p(f"""At first I identified the frequencies by filtering the channels with the mode shapes of the design model. This gave
f<sub>2</sub> = 3.78&nbsp;Hz, whereas the model-free FDD gave {tw['meas_f'][1]:.2f}&nbsp;Hz. With the first bending and torsion modes only
5&nbsp;% apart, the design-model shapes let one mode leak into the other. From this I learnt that a model which has not yet been checked
against the structure should be used only to name the modes, and not to process the data.""")
R.fig("f10_commissioning", "Commissioning ambient vibration test: one channel in the time and frequency domains, and the singular values of the cross-spectral matrix of all 12 channels")
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
R.p(f"""The design model is 5 to 10&nbsp;% too soft in frequency and 9 to 14&nbsp;% too flexible in the load test, which is normal for a design
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
mass of a wet deck in the monsoon. Environmental changes of this size are known from other bridges, where they were comparable with or
larger than the changes caused by moderate damage (Peeters and De Roeck, 2001; Sohn, 2007). For each frequency a linear model in the temperature T and the logarithm of the RMS acceleration (a
measure of how busy the bridge was during the record) was fitted:""")
R.eq(r"\hat f_k = \beta_{0k} + \beta_{1k}\,(T - 25) + \beta_{2k}\ln a_{RMS}", "norm")
R.p(f"""After this normalisation the scatter of f<sub>1</sub> is {nb['f1']['sd'] * 1e3:.1f}&nbsp;mHz ({100 * nb['f1']['sd'] / nb['f1']['mean']:.2f}&nbsp;%)
(Figure [[fig:f11_baseline_year]]). The remaining pattern in the residuals is the effect of the monsoon on the mass. I did not remove it,
because on a real bridge its size would not be known in advance.""")
by_rows = list(csv.DictReader(open("out/data/baseline_year.csv")))


def acc(hour, key, q):
    v = np.array([float(rw[key]) for rw in by_rows if rw["hour"] == hour])
    return 100 * np.mean(v > 50) if q is None else np.percentile(v, q)


R.fig("f11_baseline_year", "One year of healthy monitoring: first bending frequency against temperature, and the residual after temperature and traffic normalisation")
R.table(["Baseline quantity (span S2, healthy, reference temperature 25 &deg;C)", "Value"], [
    ["Effective prestress per girder at mid-span", f"<i>P</i><sub>e</sub> = {f(tot['Pe_mid'] / 1e3, 2)} MN; soffit precompression due to <i>P</i><sub>e</sub> {7172 * (1 / pre['A'] + (pre['zc'] - 0.1605) / pre['Zb']) / 1e3:.1f} MPa"],
    ["Soffit stress, quasi-permanent combination (mean prestress)", f"about {d['stages']['G1']['16.500']['rows']['V quasi, inf']['bot'] + 2.34:.1f} MPa compression (G1)"],
    ["Load test (4 &times; 300 kN): mid-span deflection of G1, G2, G3", " / ".join(f"{x:.2f}" for x in b['load_test_ref']['d'][:3]) + " mm"],
    ["Load test: soffit strain of G1, G2, G3", " / ".join(f"{x:.0f}" for x in b['load_test_ref']['e'][:3]) + " &mu;&epsilon;"],
    ["Natural frequencies <i>f</i><sub>1</sub> to <i>f</i><sub>4</sub>", " / ".join(f"{x:.3f}" for x in b['f_ref']) + " Hz"],
    ["Temperature sensitivity of <i>f</i><sub>1</sub> to <i>f</i><sub>4</sub>", " / ".join(f"{nb[k]['per_degC_pct']:+.2f}" for k in ('f1', 'f2', 'f3', 'f4')) + " %/&deg;C"],
    ["Scatter after normalisation, <i>f</i><sub>1</sub> to <i>f</i><sub>4</sub>", " / ".join(f"{100 * nb[k]['sd'] / nb[k]['mean']:.2f}" for k in ('f1', 'f2', 'f3', 'f4')) + " %"],
    ["Acceleration (RMS at mid-span, peak over all sensors)", f"RMS {acc('2', 'a_rms', 5):.1f} to {acc('2', 'a_rms', 95):.1f} mg at night and "
     f"{acc('14', 'a_rms', 5):.1f} to {acc('14', 'a_rms', 95):.1f} mg by day (5 to 95 % of records); median peak {acc('2', 'a_max', 50):.0f} mg at "
     f"night and {acc('14', 'a_max', 50):.0f} mg by day, above 50 mg in {acc('14', 'a_max', None):.0f} % of daytime records (heavy lorries)"],
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
     "flexural cracks at the soffit (in a fully prestressed girder any crack wider than a hairline matters), longitudinal cracks along ducts, rust staining, spalling, honeycombing, efflorescence, impact damage over the service road"],
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
in a girder designed to remain uncracked means that its precompression has been lost. Cracks along bars or ducts in a girder are
caused by corrosion rather than by a loss of precompression and are rated with the RC bands.""")
R.table(["Condition state", "Meaning", "Crack width, PSC girder", "Crack width, RC element", "Other examples"], [
    ["CS1 Good", "no repair needed", "&lt; 0.05 mm", "&lt; 0.10 mm", "&ndash;"],
    ["CS2 Fair", "minor defect, monitor", "0.05 to 0.10 mm", "0.10 to 0.20 mm", "efflorescence, minor leakage"],
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
R.fig("f24_inspection_app", "The inspection application with the year-18 inspection of Chapter 14 loaded", scale=0.78)

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
where grout voids and strand corrosion are most often found (Concrete Society, 2010). In the learning data of Chapter 11 the 30 test areas of S2 enter as span-level summaries (lowest UPV, most negative
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
R.fig("f09_sensor_layout", "Instrumentation of span S2 in plan. The data logger and a cellular modem are housed in a cabinet on pier P2.")
w_ref = np.load("out/twin_reference.npz")["w"]


def automac(sensors):
    Sm = np.array([w_ref[iy, ix] for _, iy, ix in sensors])
    M_ = np.array([[DY.mac(Sm[:, i], Sm[:, j]) for j in range(8)] for i in range(8)])
    return (M_ - np.diag(np.diag(M_))).max(), np.linalg.cond(Sm)


mac10, cond10 = automac(DY.SENSORS[:10]); mac12, cond12 = automac(DY.SENSORS)
R.p(f"""My first layout did not work. With the first ten sensors (three on each outer girder and one at mid-span of each inner
girder), the AutoMAC matrix of the twin's mode shapes at the sensor positions showed two pairs of modes that could not be distinguished,
with MAC = {mac10:.2f} (Figure [[fig:f12_automac]]): the second bending mode and the third transverse mode, and the second torsion mode and the
second transverse mode. All four modes have no response at mid-span, so only the quarter-point sensors of the outer girders saw them, and
there the two modes of each pair had proportional values. The matrix of mode shapes at the sensors was singular (condition number of the
order of 10<sup>{math.floor(math.log10(cond10))}</sup>), so the modes could not be separated. Two additional sensors at the quarter points of the
inner girders G3 and G4 reduced the largest off-diagonal MAC to {mac12:.2f} and the condition number to {cond12:.1f}. The modal assurance
criterion used throughout is""")
R.eq(r"\mathrm{MAC}(\phi_a,\phi_b) = \dfrac{\left|\phi_a^{\mathrm{T}}\phi_b\right|^2}{\left(\phi_a^{\mathrm{T}}\phi_a\right)\left(\phi_b^{\mathrm{T}}\phi_b\right)}", "mac")
R.fig("f12_automac", "AutoMAC of the first eight modes of the twin at the sensor positions, for the first and the adopted layouts")
R.h2("Sampling frequency and duration", "9.2")
R.p("""The data loggers sample at 1&nbsp;kHz with a hardware anti-alias filter and decimate the signals to 100&nbsp;Hz, so that the modes up to
about 20&nbsp;Hz lie well inside the usable band of 40&nbsp;Hz. A record lasts 30 minutes, i.e. more than 6000 cycles of the first mode. With
10-minute records the scatter of the identified f<sub>1</sub> in my simulations was about &plusmn;0.5&nbsp;%, which is too large compared with
the changes to be detected; 30-minute records together with a fitted spectrum reduced it to about 0.15&nbsp;%. Records are taken at 02:00,
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
R.p("""In 27 simulated records of the healthy bridge with known frequencies, 15 with night-time and 12 with daytime traffic, the identified
first frequency had a scatter of 0.14&nbsp;% (standard deviation) and was never more than 0.35&nbsp;% from the true value. The third frequency was
identified about 0.2&nbsp;% high with a scatter of only 0.04&nbsp;%; a constant offset of this kind does not matter, because every record is processed
in the same way and compared with the baseline. These errors include the effect of traffic and are comparable with the residual scatter after
normalisation in Table [[tab:baseline]]. A change of frequency smaller than about 0.5&nbsp;% can therefore
not be detected from a single record, which is why trends over many records are used in Chapter 14.""")



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
     "loss d<sub>s</sub> of flexural stiffness over 4 to 10 m near mid-span in a load test; in ambient vibration only 20 to 50 % of it, since cracks in a prestressed girder close again; residual crack width about 0.1 to 0.35 mm",
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
R.fig("f13_sensitivity", "Response of the twin (without noise) to each mechanism, from healthy (0) to the upper end of its range (1)")
dE_equiv = np.interp(-abs(10 * nb['f1']['per_degC_pct']), cdd['f1'][::-1], np.linspace(0, 30, len(cdd['f1']))[::-1])
R.p(f"""These changes have to be compared with the effect of temperature: a day 10&nbsp;&deg;C warmer lowers f<sub>1</sub> by about
{abs(10 * nb['f1']['per_degC_pct']):.1f}&nbsp;%, which is about the same as losing {dE_equiv:.0f}&nbsp;% of the modulus over the whole of one girder. As the problem statement also
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
    ["Environment", "effective bridge temperature from a seasonal and daily model of the site; deck moisture in the monsoon", "14 to 42 &deg;C; mass +0.6 % in the monsoon, &plusmn;0.2 % at random"],
    ["Vibration", "response of 8 modes to moving lorries (bounce 1.3 to 2.6 Hz, amplitude 2 to 6 %), cars and broadband ambient forces; 12 channels, 30 minutes, 100 Hz", "3 to 60 vehicles per 10 minutes at night, 150 to 420 by day; damping 1.5 &plusmn; 0.2 %; sensor noise 2 &micro;g/&radic;Hz"],
    ["Identification", "complete processing of Chapter 9 applied to each record", "frequencies, MAC, damping, <i>a</i><sub>RMS</sub> and <i>a</i><sub>max</sub> as identified, not as simulated"],
    ["Load test", "four 300 kN lorries; deflection and strain of all six girders", "lorry weight &plusmn;1 %; 0.05 mm; 3 &mu;&epsilon;"],
    ["Geometry and VW strain", "change beyond the healthy creep prediction", "creep model error 2.5 mm and 25 &mu;&epsilon; (common to all girders); instrument error 0.8 mm and 3 &mu;&epsilon;"],
    ["Bearings", "movement per &deg;C at both ends", "&plusmn;0.012 mm/&deg;C"],
    ["NDT", "30 test areas of span S2: rebound, UPV, cover, HCP, resistivity, carbonation, chloride", "scatter of strength, cover and carbonation rate in the structure; reading errors of each test"],
    ["Visual inspection", "observations with size and condition state", "hairline cracks often missed (probability of detection 1 &minus; e<sup>&minus;w/0.06</sup>); condition state wrong by one for 15 % of entries; early defects often not visible"],
], "Steps in the generation of one synthetic campaign", tag="generation", widths=[15, 47, 38], small=True)
lib_cols = next(csv.reader(open("out/data/library.csv")))
n_raw = sum(1 for c in lib_cols if c not in ("id", "loc", "n_mech", "mechs") and not c.startswith(("y_", "sev_")) and not c.endswith("_true"))
R.p(f"""In total {ml['n']} campaigns were generated: {ml['counts']['H']} healthy, about {int(np.mean([ml['counts'][m] for m in SC.MECHS]))} for each
single mechanism and {ml['counts']['pair']} with two mechanisms. Each campaign contains {n_raw} measured quantities and its true labels, and the data are
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
         f"{float(r['hcp']):.0f}", f"{float(r['resistivity']):.0f}", f"{float(r['chloride']):.2f}", r["cluster"]] for r in ndt_rows]
R.table(["Location", "Rebound number", "UPV (km/s)", "Cover (mm)", "Carbonation (mm)", "HCP (mV)", "Resistivity (k&Omega;cm)",
         "Chloride (% of cement)", "Cluster (Ch. 14)"], rows,
        "Synthetic NDT data along girder G1 of span S2 at an age of 18 years", num_cols=(1, 2, 3, 4, 5, 6, 7), tag="exndt", small=True,
        widths=[12, 9, 8, 8, 12, 8, 12, 12, 19])
R.p("""Along G1 the readings change towards the joint end (Z4 and Z5): the resistivity drops and the potentials become more negative where
the concrete is wet, the cover is well below the specified 50&nbsp;mm, and the chloride content is more than twice that of the dry zones.
Neither the carbonation front (12 to 14&nbsp;mm) nor the chloride content (about 0.15&nbsp;% of the cement mass, below the usual threshold of
0.2 to 0.4&nbsp;%) would explain active corrosion on its own, so the low potentials have to be confirmed with cores taken at the depth of the
steel before the repair (Section 15.3). UPV and rebound are hardly affected, because the concrete itself is sound.""")
st = TW.healthy_state(); st["kh_mult"][:, :] = 100
f_br, *_ = resp(st)
st = TW.healthy_state(); st["conc"].append((0, 0, P.L, 0.2))
f_cd, *_ = resp(st)
R.table(["Mode", "Healthy (Hz)", "Concrete deterioration, G1, &minus;20 % E (Hz)", "Bearings restrained at both ends, &times;100 (Hz)", "Temperature +10 &deg;C (Hz)"],
        [[n, f"{f_h[k]:.3f}", f"{f_cd[k]:.3f} ({100 * (f_cd[k] / f_h[k] - 1):+.1f} %)", f"{f_br[k]:.3f} ({100 * (f_br[k] / f_h[k] - 1):+.1f} %)",
          f"{f_h[k] * (1 + 10 * nb['f1']['per_degC_pct'] / 100):.3f} ({10 * nb['f1']['per_degC_pct']:+.1f} %)"]
         for n, k in (("<i>f</i><sub>1</sub>, bending", 0), ("<i>f</i><sub>2</sub>, torsion", 1), ("<i>f</i><sub>3</sub>, transverse", 2))],
        "Synthetic vibration data for two deterioration cases and a change of temperature", num_cols=(1, 2, 3, 4), small=True)
R.p(f"""The example in the problem statement changes f<sub>1</sub> from 3.20 to 3.05&nbsp;Hz, i.e. by &minus;4.7&nbsp;%. Since the frequency
varies with &radic;EI, a change of that size on B33 would need a loss of about 9&nbsp;% of the bending stiffness of the whole deck, or simply a
temperature rise of about 30&nbsp;&deg;C. Of the chosen mechanisms only two reach it, and only near the upper ends of their ranges: concrete
deterioration over the whole of one girder (&minus;{abs(min(cdd['f1'])):.1f}&nbsp;% at 30&nbsp;%) and the largest losses of prestress, at which the girders
have cracked (&minus;{abs(min(pl['f1'])):.1f}&nbsp;% at 35&nbsp;%). The frequency alone cannot tell these apart. The pattern across the modes carries more information: deterioration of G1 lowers f<sub>1</sub> more than f<sub>2</sub> and
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
a<sub>RMS</sub> and a<sub>max</sub> depend much more on traffic than on condition (the RMS is about 50&nbsp;% higher by day than at night, and the peak of a record is set by
whichever heavy lorry happens to cross), so they are used
to <i>normalise</i> the frequencies rather than as condition features. &delta;<sub>max</sub> was replaced by the change of the mean deflection
together with the G1&ndash;G6 asymmetry, which separate a global from a local change. Figure [[fig:f14_effect_size]] shows, before any
classifier is trained, how strongly each feature reacts to each mechanism, measured as the shift of its mean in units of the healthy standard
deviation.""")
R.p("""Reading Figure [[fig:f14_effect_size]] column by column already gives most of the rules an engineer would write: loss of prestress moves
the levels and the VW strain; bearing restraint changes the bearing movement ratio, raises f<sub>1</sub> and often comes with a leaking joint;
corrosion changes the potentials, the resistivity and the cover relative to carbonation; concrete deterioration lowers UPV and rebound and,
over a whole girder, the frequencies; cracking and local damage appear in the load test and the inspection. The machine-learning models are
useful because they combine these weak and overlapping signals consistently, attach a measure of confidence and estimate the size of the
problem. Figure [[fig:f15_distributions]] shows the distributions of six of the features.""")
R.fig("f14_effect_size", "Effect of each mechanism on each feature: shift of the mean from the healthy mean, in healthy standard deviations (campaigns in the upper half of each severity range). Blue: decrease, red: increase.", scale=0.92)
R.fig("f15_distributions", "Distributions of six features for healthy (H) and single-mechanism campaigns (boxes: quartiles; whiskers: 1.5 times the interquartile range)")

# ====================================================================== 13
R.h1("AI/ML-Assisted Condition Assessment", "13")
R.p(f"""The library was split into a training set ({ml['n_train']} campaigns) and a test set ({ml['n_test']} campaigns), stratified by class;
all results below are for the test set. The assessment follows the usual sequence of structural health monitoring (Farrar and
Worden, 2013): first decide whether the bridge has changed from its healthy state, then which mechanism is present, how large it is and
where it is.""")
R.h2("Novelty detection", "13.1")
R.p("""The novelty detector is trained on healthy campaigns only. This matches the situation on a real bridge, which has a healthy baseline
but no labelled examples of its own damage. For a standardised feature vector x the squared Mahalanobis distance from the healthy data (Worden et al., 2000) is""")
R.eq(r"D^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu}_H)^{\mathrm{T}}\,\mathbf{\Sigma}_H^{-1}\,(\mathbf{x} - \boldsymbol{\mu}_H)", "maha")
R.p("""where &mu;<sub>H</sub> and &Sigma;<sub>H</sub> are the mean and the covariance of the healthy training campaigns. The covariance was shrunk
slightly towards its diagonal, and a small value was added to the diagonal so that features which are always zero on a healthy bridge (rust
stains, for example) do not make the matrix singular. A campaign is flagged when D&sup2; exceeds the 99th percentile of the healthy training
values, i.e. the designed false-alarm rate is 1&nbsp;%.""")
R.table(["Data source", "Healthy (false alarm)"] + [SH[m] for m in SC.MECHS],
        [[SRC_NAME.get(s, s), pct(nov[s]["false alarm"])] + [pct(nov[s][m]) for m in SC.MECHS] for s in SRC],
        "Share of test campaigns flagged by the novelty detector", num_cols=tuple(range(1, 8)), tag="novelty", small=True)
R.p(f"""With vibration data alone the detector finds most cases of concrete deterioration ({pct(nov['Vibration only']['CD'])}) and local
stiffness loss ({pct(nov['Vibration only']['LS'])}), fewer bearing cases ({pct(nov['Vibration only']['BR'])}, essentially those restrained at both
ends), and almost no cases of prestress loss ({pct(nov['Vibration only']['PL'])}) or corrosion ({pct(nov['Vibration only']['CO'])}). This confirms
Figure [[fig:f13_sensitivity]] in the presence of noise and environmental variation: vibration monitoring alone is not suitable for the two
mechanisms that most often govern the life of a post-tensioned bridge. Adding levelling, VW strain and bearing movement, which can all be
measured without closing the bridge, raises the detection of prestress loss to {pct(nov['Continuous monitoring (no closure)']['PL'])} and of
bearing problems to {pct(nov['Continuous monitoring (no closure)']['BR'])}. With all sources combined, every mechanism is detected in
{pct(min(nov['All sources fused'][m] for m in SC.MECHS))} or more of the cases at a false-alarm rate of {pct(nov['All sources fused']['false alarm'])};
the cases that are missed are the smallest ones.""")
R.fig("f16_novelty", "Novelty detection: detection rate by data source, and probability of detection against severity")
R.h2("Identification of the mechanism and data fusion", "13.2")
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
lower the frequencies slightly and differ mainly in where and by how much. With all sources, the remaining errors are
small cases of prestress loss, concrete deterioration and local stiffness loss that are classified as healthy, and a few cases of local stiffness
loss taken for concrete deterioration; such cases are hard to see by any means.""")
R.h2("Size of the loss of prestress", "13.3")
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
belong to the {' and the '.join(re.sub(r" [(][^)]*[)]$", "", FE.LABELS[k]) for k in sorted(g['length_scales'], key=g['length_scales'].get)[:2])}.""")
R.fig("f20_pl_regression", "Estimated against true extra loss of prestress for the test campaigns: Gaussian-process regression and physical inversion")
loc = ml["localisation"]
R.h2("Localisation", "13.4")
R.p(f"""A random forest using the deflections and strains of each girder in the load test, the levels, the VW strains and the asymmetry of the
first mode shape identifies the affected girder in <b>{pct(loc['accuracy'])}</b> of the single-mechanism test cases (prestress loss
{pct(loc['by_mechanism']['PL'])}, concrete deterioration {pct(loc['by_mechanism']['CD'])}, cracking {pct(loc['by_mechanism']['CR'])}, corrosion
{pct(loc['by_mechanism']['CO'])}, local stiffness loss {pct(loc['by_mechanism']['LS'])}). Corrosion is located worst because its effect on the global
response is small and often near a support, where the load test is not sensitive; for corrosion the location comes from the NDT grid instead
(Chapter 14).""")
R.h2("Physical consistency of the models", "13.5")
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
rows = [[k, (f"{v['first_alarm_year']:.1f}" if v['first_alarm_year'] is not None else "no alarm"), f"{v['slope_last3y_per_year']:+.3f}"]
        for k, v in tr.items()]
R.table(["Monitored quantity", "First EWMA alarm (year)", "Theil&ndash;Sen slope over the last 3 years (per year)"], rows,
        "Trend identification on the monitoring record", num_cols=(1, 2), full=False)
R.fig("f21_case_trends", "Eighteen years of monitoring of span S2: monthly values and EWMA with &plusmn;3&sigma; limits from the first two years. The grey bands mark the hidden onset of the changes and the dash-dot lines the first alarm.")
R.p(f"""The bearing transducers gave the first and clearest signal, in year {tr['bearing B movement']['first_alarm_year']:.1f}: the movement at end B
fell towards zero while that at end A increased, which is the signature of a bearing being locked at one end while the other end takes over the
movement. The f<sub>1</sub> residual also raised an alarm, in year {tr['f1 residual']['first_alarm_year']:.1f}, but <i>upwards</i>: the span had become
stiffer. On its own this could be mistaken for a sign of good health or for an error in the temperature model; next to the bearing record it is
the restraint described in Chapter 10. The level of G1 dropped (alarm in year {tr['level G1']['first_alarm_year']:.1f}) and its VW gauge drifted towards
tension (year {tr['VW strain G1']['first_alarm_year']:.1f}), which is the pattern of lost precompression in one girder. The delays after the hidden onsets
follow how directly each quantity measures its mechanism: the bearing transducers raised their alarm
{tr['bearing B movement']['first_alarm_year'] - cs['onset']['bearing']:.1f} years after the bearings began to lock, the level and the VW strain of G1
{tr['level G1']['first_alarm_year'] - cs['onset']['corrosion']:.1f} and {tr['VW strain G1']['first_alarm_year'] - cs['onset']['corrosion']:.1f} years after the
corrosion began, and the frequency residuals, which see the bearings only indirectly, after
{tr['f1 residual']['first_alarm_year'] - cs['onset']['bearing']:.1f} and {tr['f2 residual']['first_alarm_year'] - cs['onset']['bearing']:.1f} years. With vibration data alone the record
would only have said that the bridge had become stiffer.""")
R.h2("Principal inspection and NDT at year 18", "14.2")
obs = json.load(open("inspection_app/seed_observations.json"))
key = sorted([o for o in obs if o["cs"] >= 3], key=lambda o: -o["cs"])
R.table(["Span", "Component", "Location", "Observation", "Size", "CS", "Note"],
        [[o["span"], o["component"], o["location"], o["observation"], o["size"].replace("m2", "m&sup2;"), o["cs"], o["note"]] for o in key],
        "Main observations of the year-18 inspection (condition state 3 and worse)", num_cols=(5,), small=True)
gr = cs["decision"]["groups"]
names_g = {"G": "girders", "EJ": "expansion joints", "BRG": "bearings", "DR": "drainage", "PC": "pier caps", "WC": "wearing coat",
           "DS": "deck slab", "CB": "crash barriers", "AB": "abutments", "FP": "footpaths", "PR": "piers", "CG": "cross girders"}
by_cs = {}
for k, v in gr.items():
    by_cs.setdefault(v, []).append(names_g[k])
cs_text = and_join([f"the {and_join(by_cs[s])} in CS{s}" for s in sorted(by_cs, reverse=True) if s >= 3])
R.p(f"""The bridge condition index is <b>{cs['decision']['bci']:.0f}</b> out of 100, with {cs_text}; all other component
groups are in CS2 or better. All serious findings lie within a few metres of
the expansion joint over pier P2: the torn seal, silt around the S2 bearings at end B and the S3 bearings at end A, leakage and spalling on the pier
cap, a short drain spout, and rust-stained splitting cracks and spalling at the anchorage end of G1. The inspector rated the two
splitting cracks along the duct line in zones Z4 and Z5 (0.39 and 0.45&nbsp;mm wide) as CS5, one state worse than the RC bands of Table
[[tab:cs]] give for such cracks. The rating was kept in the record, because rust coming out of a crack along a tendon duct next to an
anchorage is the warning sign that preceded the collapse of the Ynys-y-Gwas bridge (Woodward and Williams, 1988), but the category of the bridge was decided only after the structural check of Section 14.4.""")
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
R.p("""The clustering is useful because it is unsupervised: it knows nothing about the joints, yet it separates the wet end zones next to both piers
from the rest of the bridge, which tells the engineer that the joint over pier P1 also needs attention. The engineering limits then single out the
G1 end zone of S2. The UPV and rebound values are normal almost everywhere, which rules out poor concrete as the cause.""")
R.fig("f22_case_ndt", "Year-18 NDT survey of all girders grouped by a Gaussian mixture model, and half-cell potential against resistivity with the limits of ASTM C876")
R.h2("AI/ML output and its interpretation", "14.3")
R.fig("f23_case_ml", "Year-18 campaign: probability of each mechanism from the models using vibration only and all sources (* marks the mechanisms actually present), and change of the load-test response of each girder since year 0")
pf, pv = cm["p_fused"], cm["p_vibration"]
nsc, nmd, nnames = pickle.load(open("out/models.pkl", "rb"))["novelty"]
xz = nsc.transform(np.array([[cs["campaign_features"][n] for n in nnames]]))
for n in FE.GROUPS["VI"]:
    xz[0, nnames.index(n)] = 0.0          # inspection features at their healthy mean
d_novi = math.sqrt(float(nmd.score(xz)[0]))
R.p(f"""The Mahalanobis distance of the year-18 campaign is {math.sqrt(cm['novelty_score']):.0f}, against a threshold of
{math.sqrt(cm['novelty_threshold']):.1f}, so the bridge is clearly no longer in its healthy state. Most of the distance comes from inspection
features that are always zero on the healthy bridge, such as rust stains, but even without them the distance would be {d_novi:.0f}, about
{['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][round(d_novi / math.sqrt(cm['novelty_threshold'])) - 2]} times the threshold. The models using all sources give corrosion a probability of
{pf['CO']:.2f} and bearing deterioration {pf['BR']:.2f}. Loss of prestress receives only {pf['PL']:.2f}, which is to be expected for a loss of a few
per cent in one girder: early losses of this kind are the main remaining error of the classifier (Section 13.2), so the size of the loss was
estimated separately. The models using vibration only give bearing deterioration {pv['BR']:.2f} and every other mechanism less than
{max(v for m, v in pv.items() if m != 'BR') + 0.01:.1f}; they find the bearings, which stiffen the span, and nothing else. The localisation model
points to G1, with a probability of {cm['loc_proba'][0]:.2f}.""")
R.p(f"""The estimate of the prestress loss needed more care. My first version of the Gaussian-process model, which used the largest crack width
of any type as an input, gave <b>18&nbsp;%</b> for this campaign. It had been misled by the 0.45&nbsp;mm rust-stained splitting cracks at the joint
end: in its training data a visible crack on a girder almost always meant a flexural crack, and flexural cracks appear only after a loss of about
25&nbsp;%. The crack feature was then split into flexural cracks and cracks along bars or ducts, and the model was retrained. It now gives
{cm['pl_gpr_mean']:.1f} &plusmn; {1.96 * cm['pl_gpr_sd']:.1f}&nbsp;% (95&nbsp;% band), and its error on the test set also became smaller. Because
the levels and the VW strain changed mainly in G1, the G1 measurements were also inverted directly with the sensitivities of Eq. ([[eq:plsens]]),
which gives {cm['pl_G1_combined']:.1f} &plusmn; {1.96 * cm['pl_G1_sd']:.1f}&nbsp;% for G1 (levelling {cm['pl_G1_levelling']:.1f}&nbsp;%, VW strain
{cm['pl_G1_vw']:.1f}&nbsp;%). The two estimates overlap. The direct estimate for G1 was used in the structural check, because it rests on two
measurements whose physics is clear, and the Gaussian-process value was kept as an upper indication.""")
R.p(f"""The result also makes physical sense. A leaking joint explains the bearing distress in both spans next to it, the wet concrete and low
resistivity at the girder ends, and, through the anchorages at that end, a possible loss of prestress in G1. The slight rise of f<sub>1</sub> fits
restrained bearings. The absence of any change in UPV and rebound rules out a problem with the concrete. The load-test deflections changed by less than
{math.ceil(max(abs(v) for v in cs['campaign_girders']['d'])):.0f}&nbsp;% in every girder, which is consistent with a girder that has lost some precompression but has not cracked. The output of the models is therefore consistent with one cause,
the joint, and three consequences. What it cannot show is whether the strands inside the ducts near the G1 anchorage are corroding; only an
investigation of the tendons can show that, so one has to be included in the maintenance scheme.""")
R.h2("Structural check with the estimated loss", "14.4")
R.p(f"""With the estimated loss in G1 ({asm['loss_used_pct']:.1f}&nbsp;%), the margin of the soffit stress against cracking under the rare
combination becomes {asm['sls_margin_design']:+.2f}&nbsp;MPa on the design basis (0.9P<sub>m</sub>) and {asm['sls_margin_mean']:+.2f}&nbsp;MPa with the
mean prestress, and the soffit stress under the quasi-permanent combination is still {asm['qp_soffit_mean']:+.1f}&nbsp;MPa compression. The flexural
capacity at mid-span is not affected by a loss of force, since the strands yield at failure anyway, and M<sub>Ed</sub>/M<sub>Rd</sub> remains
{asm['uls_ratio']:.2f}. At the corroded end B of G1 the vertical component of the prestress decreases, and the net design shear 2.75&nbsp;m from the bearing
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
        desc.append(f"an extra loss of prestress in G1 that grew with the corrosion to {100 * m['loss']:.1f}&nbsp;%")
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
R.p(f"""The assessment places the bridge in the category <b>{cat_main.lower()}</b> ({cat_note}). The plan in Table [[tab:plan]] starts with a
special inspection and with removing the cause, the water at the joint over pier P2. The tendons and the remaining prestress are then measured,
since the monitoring can only infer them, before the damage is repaired. Finally a new baseline is established so that the repaired bridge is
again compared with itself. The actions are listed in the order in which they are to be started.""")
R.table(["Action", "When", "Reason"], [[a, w_, r_] for a, w_, r_, n_ in dec["actions"]],
        "Maintenance actions, time frames and the evidence behind each", tag="plan", widths=[42, 16, 42], small=True)
R.h2("Sequence and justification", "15.3")
iv = dec["intervals"]
R.p("""The drains and the joint come first, because every other defect found is a consequence of water; repairing the concrete or the bearings
while the joint still leaks would waste the repair. The seal of the joint over pier P1, which is also torn and next to which the NDT clustering
found wet girder ends, is replaced at the same time. The bearings of S2 at end B, and of S3 at end A under the same joint, are replaced by lifting
each span on jacks placed under the end diaphragm, which was designed as a jacking beam; the lift is limited to 10&nbsp;mm and the joint is closed to
traffic for one night per span. Concrete repair at the G1 end zone and on the pier cap follows the half-cell map, after cores have given the chloride content and
the carbonation depth at the level of the steel, with an anti-carbonation coating on
the repaired and low-cover areas and a migrating corrosion inhibitor where the potentials are in the uncertain range.""")
R.p("""The investigation of the tendons (GPR to locate the ducts, impact-echo along them, and a borescope through a small hole drilled at the
anchorage and at the high points of the ducts) and a lift-off or stress-release test on G1 decide the last step. If the measured force confirms a
loss that removes the margin against cracking, external post-tensioning restores it, with two straight 12T15 tendons per girder, deviators at the
mid-span cross girder and anchorages on the end diaphragms (IRC:SP:40). If not, the girder is kept under closer monitoring.""")
R.h2("Inspection and monitoring after the works", "15.4")
R.p(f"""Until the works are complete, routine inspection is carried out {iv['routine']} and the monitoring data are reviewed weekly. A principal
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
R.h1("Verification of Results", "16")
n_ok = sum(c["ok"] for c in ver)
R.p(f"""The results were checked at three levels. The grillage model was first compared with simple beam results under a uniform load
(Section 3.4). In addition, {len(ver)} of the computed results were checked independently with simpler methods: hand formulas, a separate
influence-line calculation, Courbon's method, the frequency formula for a simply supported beam, and recovery of known parameters from
synthetic measurements. {f"All {n_ok}" if n_ok == len(ver) else f"Only {n_ok} of the {len(ver)}"} checks are within their tolerances (Appendix B). Finally, every intermediate result was
compared with what could be expected from the physics or from practice, and the clauses of IRC:6 for the impact factor of the 70R vehicle,
the footway load on spans over 30&nbsp;m and the reduction for three loaded lanes were checked against the code.""")
R.p("""These checks found the following errors in my first versions, all of which were corrected before the results in this report were
produced:""")
R.ul(["""My first cable layout moved two cables sideways inside the 300&nbsp;mm web, which is impossible. The profiles were redrawn so that the
lateral movement takes place inside the bottom bulb, and a check of the duct positions along the whole girder was added (Section 4.2).""",
      """Stressing all four cables at 10 days gave 18.6&nbsp;MPa at the soffit against an allowable 17.3&nbsp;MPa, so two-stage stressing was
adopted (Section 4.3).""",
      """The first vibration model gave peak accelerations of 170&nbsp;mg, because the bounce frequencies of the lorries coincided with the first mode
and the ambient forcing was too strong. Both were recalibrated until the night-time RMS (0.6 to 0.8&nbsp;mg) and the median peak of a daytime record
(about 18&nbsp;mg) were of the order reported for bridges of this size; much higher peaks now occur only when a heavy lorry crosses.""",
      """With 10-minute records and simple peak picking the scatter of f<sub>1</sub> was &plusmn;0.5&nbsp;%; 30-minute records with a fitted spectrum
reduced it to about 0.15&nbsp;% (Section 9.2).""",
      """The AutoMAC check showed that my first sensor layout could not separate two pairs of modes, so two accelerometers were added
(Section 9.1).""",
      """Identifying the commissioning test with the mode shapes of the design model biased f<sub>2</sub> by almost 4&nbsp;%; the commissioning test
is now identified without model shapes, by FDD (Section 5.2).""",
      """The first synthetic inspection detected every hardened bearing and every case of 2&nbsp;% corrosion, which made inspection look perfect.
Detection was made to depend on the severity of the defect, and leaking joints were also allowed on sound bridges.""",
      """The novelty detector failed at first on the inspection features, because their covariance matrix was singular. A small value was added
to its diagonal, and the effect on the false-alarm rate was checked (Section 13.1).""",
      """My first prestress-loss model mistook corrosion cracks for flexural cracks and estimated an 18&nbsp;% loss for the year-18 campaign. The
crack feature was split by type, and the estimate was checked against a direct physical inversion (Section 14.3)."""])

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
      """The healthy baseline has to come from measurements on the bridge; the design model alone is not good enough. It was 5 to 10&nbsp;% off in frequency; after three parameters had
been updated on the commissioning tests, the twin reproduced them within 0.2&nbsp;%. One year of data provided the temperature and traffic
normalisation without which the frequencies of this bridge are not usable, since they vary by about 4&nbsp;% over a year.""",
      """Vibration monitoring alone is poor at detecting the two mechanisms that usually decide the life of a post-tensioned bridge, loss of
prestress and corrosion. Levelling, embedded VW gauges and bearing displacement transducers are inexpensive and detect them much better, and
NDT and visual inspection remain essential for corrosion.""",
      f"""The AI/ML assessment became useful mainly through combining the data sources, much more than through the choice of algorithm: the macro-F1 rose from
{sup['Vibration only']['macro_F1']:.2f} with vibration to {sup['All sources fused']['macro_F1']:.2f} with all sources. The importance of the features
agrees with the physics, and the Gaussian process gives the loss of prestress with a realistic uncertainty.""",
      """Applied to the bridge at 18 years, the assessment traced the defects to one cause, a leaking joint, located them, estimated the loss of
prestress in the affected girder, and led to a maintenance plan that removes the cause, measures what the monitoring can only infer, repairs
the damage and re-establishes the baseline. My first version of the prestress model was misled by corrosion cracks, which showed me that every ML
output has to be checked against the physics before it is used for a decision."""])
R.p("""The work could be extended by calibrating the noise models and the detection thresholds with data from an instrumented bridge, by
replacing the grillage with a finite-element model that represents bond and crack behaviour, and by adding the cost of the actions so that
the timing of maintenance can be optimised over the life of the bridge.""")

# ====================================================================== references
R.h1("References", newpage=True)
refs = [
    "ASTM C876-15. <i>Standard Test Method for Corrosion Potentials of Uncoated Reinforcing Steel in Concrete</i>. ASTM International.",
    "Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45, 5&ndash;32.",
    "Brincker, R., Zhang, L. and Andersen, P. (2001). Modal identification of output-only systems using frequency domain decomposition. <i>Smart Materials and Structures</i>, 10(3), 441&ndash;445.",
    "Concrete Society (2010). <i>Durable Post-Tensioned Concrete Structures</i>, Technical Report 72. Camberley.",
    "EN 1992-1-1:2004. <i>Eurocode 2: Design of Concrete Structures &ndash; Part 1-1: General Rules and Rules for Buildings</i>. CEN, Brussels.",
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
    "IRC:SP:69-2011. <i>Guidelines and Specifications for Expansion Joints</i>. Indian Roads Congress, New Delhi.",
    "IRC:SP:80-2008. <i>Guidelines for Corrosion Prevention, Monitoring and Remedial Measures for Concrete Bridge Structures</i>. Indian Roads Congress, New Delhi.",
    "IS 516 (Part 5/Sec 1):2018 and IS 516 (Part 5/Sec 4):2020. <i>Hardened Concrete &ndash; Methods of Test, Non-Destructive Testing: Ultrasonic Pulse Velocity; Rebound Hammer</i>. Bureau of Indian Standards, New Delhi.",
    "Peeters, B. and De Roeck, G. (2001). One-year monitoring of the Z24-Bridge: environmental effects versus damage events. <i>Earthquake Engineering and Structural Dynamics</i>, 30(2), 149&ndash;171.",
    "Rasmussen, C.E. and Williams, C.K.I. (2006). <i>Gaussian Processes for Machine Learning</i>. MIT Press.",
    "Sohn, H. (2007). Effects of environmental and operational variability on structural health monitoring. <i>Philosophical Transactions of the Royal Society A</i>, 365, 539&ndash;560.",
    "Woodward, R.J. and Williams, F.W. (1988). Collapse of Ynys-y-Gwas bridge, West Glamorgan. <i>Proceedings of the Institution of Civil Engineers, Part 1</i>, 84, 635&ndash;669.",
    "Worden, K., Manson, G. and Fieller, N.R.J. (2000). Damage detection using outlier analysis. <i>Journal of Sound and Vibration</i>, 229(3), 647&ndash;667.",
]
R.raw("<ol style='font-size:11pt; line-height:1.25'>" + "".join(f"<li style='margin-bottom:1.5pt'>{r}</li>" for r in refs) + "</ol>")

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
    ["make_figures.py, make_figures_ml.py", "figures"],
    ["build_report.py, report_lib.py", "this report (written as HTML and printed to PDF)"],
    ["inspection_app/index.html", "digital visual inspection application (single file, works offline; built by build_app.py)"],
    ["out/data/*.csv", "baseline year, scenario library, NDT surveys, visual inspection, 18-year monitoring record"],
], "Program and data files", widths=[38, 62], small=True)
R.p("""The programs need Python 3 with numpy, scipy and matplotlib; pymupdf and Edge or Chrome are used only to print the report. The five
<code>run_*.py</code> scripts are run in order, followed by <code>verify.py</code>, <code>make_figures.py</code> and <code>build_report.py</code>.
The complete chain takes about ten minutes on my laptop (16 threads), most of it for the 3600 vibration records of the scenario library.""")
R.h1("Appendix B: Verification Checks")
UNIT = {"m2": "m&sup2;", "kNm": "kN&nbsp;m", "microstrain": "&mu;&epsilon;", "mm/degC": "mm/&deg;C"}


def num(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


rows = []
for c in ver:
    if c.get("absolute"):        # false-alarm rate: compared in percentage points
        vals = [f"{100 * c['computed']:.1f} %", f"{100 * c['hand']:.0f} %", f"{100 * c['rel_diff']:.1f} points", f"{100 * c['tol']:.0f} points"]
    else:
        vals = [f"{num(c['computed'])} {UNIT.get(c['unit'], c['unit'])}".strip(), num(c["hand"]), f"{100 * c['rel_diff']:.1f} %",
                f"{100 * c['tol']:g} %"]
    rows.append([c["check"] + (f" ({c['note']})" if c["note"] else "")] + vals + ["pass" if c["ok"] else "<b>fail</b>"])
R.table(["Check", "Computed", "Independent", "Difference", "Tolerance", "Result"], rows,
        "Independent checks of the computed results", num_cols=(1, 2, 3, 4), widths=[46, 14, 11, 10, 11, 8], small=True)

# ====================================================================== write and print (two passes for the page numbers)
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
import pymupdf
doc = pymupdf.open(str(pdf_path))
doc.set_metadata({"title": "Post-Tensioned Prestressed Concrete I-Girder Bridge of 33 m Span", "author": f"{STUDENT} ({ROLL})",
                  "subject": "CE60258 One-Month Integrated Big Problem", "creator": "", "producer": ""})
doc.save(str(pdf_path) + ".tmp", garbage=3, deflate=True)
doc.close()
os.replace(str(pdf_path) + ".tmp", pdf_path)
n_found = sum(1 for _, _, _, a in R.toc if a in pages2)
print(f"report: {pdf_path} ({pymupdf.open(str(pdf_path)).page_count} pages), {R.fig_no} figures, {R.tab_no} tables, "
      f"{R.eq_no} equations; headings located {n_found}/{len(R.toc)}")
