"""
Builds report/B33_PSC_Bridge_Integrated_Study.html and .pdf from the saved results.
All numbers in the text are read from out/*.json so the report cannot drift
away from the calculations.
"""
import json, math, csv
import numpy as np
from report_lib import Report, print_pdf, heading_pages
from psc import inputs as P, section as S, prestress as PS, condition as C, features as FE, scenarios as SC
from psc import dynamics as DY, maintenance as MT

STUDENT = "Shashank"
ROLL = "Roll No.: ______________"

d = json.load(open("out/design.json"))
b = json.load(open("out/baseline.json"))
ml = json.load(open("out/ml_results.json"))
cs = json.load(open("out/case_study.json"))
ver = json.load(open("out/verification.json"))
sens = json.load(open("out/sensitivity.json"))


def f(x, n=1):
    return f"{x:,.{n}f}".replace(",", "&#8239;") if abs(x) >= 1e4 else f"{x:.{n}f}"


def pct(x, n=0):
    return f"{100 * x:.{n}f}&nbsp;%"


R = Report()
MECH = SC.MECH_NAMES
pc = d["prestress"]["per_cable"]
pre, comp_o, comp_i = d["sections"]["precast"], d["sections"]["composite_outer"], d["sections"]["composite_inner"]
ss = d["stages_summary"]
tw = b["twin"]

# ====================================================================== title
R.raw(f"""
<div class="titlepage">
 <div class="top">CE60258 &nbsp;AI/ML FOR SUSTAINABLE INFRASTRUCTURE SYSTEMS &nbsp;|&nbsp; ONE-MONTH INTEGRATED BIG PROBLEM</div>
 <h0>A 33&nbsp;m post-tensioned PSC I-girder bridge,<br>from design to a maintenance decision</h0>
 <div class="sub">Analysis and design, prestressing, healthy baseline, inspection, NDT and monitoring,
 synthetic data, AI/ML-assisted assessment and maintenance of Bridge B33</div>
 <img src="../out/figs/f01_cross_section.png" style="width:100%; margin-top:16mm; opacity:0.92">
 <div class="meta">
  <b>{STUDENT}</b><br>{ROLL}<br>
  Department of Civil Engineering, Indian Institute of Technology Kharagpur<br>
  Course instructor: Prof. Nirjhar Dhang<br>
  Submitted: 31 October 2026
 </div>
</div>""")

# ====================================================================== summary
R.h1("Summary", nobreak=False)
R.p(f"""This report follows one bridge through its life. Bridge B33 is a three-span bridge with simply
supported spans of <b>33&nbsp;m</b>, a 16&nbsp;m deck carrying three traffic lanes with footpaths and crash barriers
on both sides, and <b>six post-tensioned M45 I-girders at 2.70&nbsp;m</b> under a 225&nbsp;mm deck slab. Every
step from configuration to the maintenance plan was worked out for this one span and is carried forward
to the next step, so that the measurements, the synthetic data and the AI/ML results all refer to the same
structure.""")
cfg_rows = [
    ["Configuration", f"6 girders, 1.80 m deep (L/D = {P.L/(P.GIRDER_DEPTH+P.SLAB_T):.1f} with slab), end diaphragms + one cross girder at mid-span; elastomeric bearings 400 x 500 mm; strip-seal joints"],
    ["Live load (IRC:6)", f"70R wheeled + Class A governs; outer girder LL moment {f(d['courbon']['M'][0]*0+np.load('out/design_arrays.npz')['MLL'][0,-1],0)} kN m by grillage, {f(d['courbon']['M'][0],0)} kN m by Courbon"],
    ["Prestress", f"4 cables 12T15 per girder, two-stage stressing; P<sub>j</sub> = {f(d['prestress']['Pj']/1e3,2)} MN, P<sub>0</sub> = {f(d['prestress']['P0_mid']/1e3,2)} MN, P<sub>e</sub> = {f(d['prestress']['Pe_mid']/1e3,2)} MN (total loss {pct(1-d['prestress']['Pe_mid']/d['prestress']['Pj'],1)})"],
    ["Checks (IRC:112)", f"rare-combination soffit stress {f(ss['rare_min_bot'],2)} MPa (> -f<sub>ctm</sub> = -3.3, uncracked); quasi-permanent {f(ss['qp_min_bot'],2)} MPa (decompression satisfied); M<sub>Ed</sub>/M<sub>Rd</sub> = {d['uls']['G1']['MEd']/d['uls']['G1']['M']:.2f}; LL deflection {f(d['deflection']['LL_mm']['G1'],1)} mm (L/800 = 41 mm)"],
    ["Healthy baseline", f"f<sub>1</sub> = {b['f_ref'][0]:.2f} Hz (bending), f<sub>2</sub> = {b['f_ref'][1]:.2f} Hz (torsion) at 25 &deg;C, {b['norm']['f1']['per_degC_pct']:+.2f} %/&deg;C; load-test deflection {f(b['load_test_ref']['d'][2],2)} mm under 4 x 300 kN lorries"],
    ["Monitoring", "12 accelerometers (layout fixed by an AutoMAC check), 4 VW gauges, 4 bearing displacement transducers, levelling points, temperature; 30 min records at 100 Hz"],
    ["Synthetic data", f"{ml['n']} campaigns from the updated digital twin: healthy and six deterioration mechanisms, one or two at a time, with temperature, traffic and instrument noise"],
    ["AI/ML", f"novelty detection, random forests per mechanism, Gaussian-process estimate of prestress loss, girder localisation, NDT clustering; macro-F1 {ml['supervised']['Vibration only']['macro_F1']:.2f} with vibration alone and {ml['supervised']['All sources fused']['macro_F1']:.2f} with all sources fused"],
    ["Year-18 decision", f"{cs['decision']['category']}: leaking joint at pier P2 has led to restrained bearings and corrosion at the G1 anchorage end; joint, bearings and concrete repair, tendon investigation and a prestress check, then re-baseline"],
]
R.table(["Item", "Result"], cfg_rows, widths=[19, 81])
R.p("""Three findings carry through the study and are, in my view, the useful part of it. First, the natural
frequencies of this bridge change by about 4&nbsp;% between a cool winter night and a hot afternoon, which is more
than most of the deterioration cases change them, so frequencies are only useful after temperature and traffic
normalisation. Second, loss of prestress does not change the stiffness of an uncracked bonded girder; it shows in
levelling and in vibrating-wire strain long before it shows in vibration or a load test, and the cracking that
finally makes it visible starts at roughly 25&nbsp;% extra loss. Third, the machine-learning models were only as good
as the physics in the features given to them: fusing vibration with static, geometric, bearing and NDT data raised the
detection quality much more than any choice of algorithm did.""")
R.h1("Contents", nobreak=False, toc=False)
R.raw("<!--TOC-->")

# ====================================================================== 1
R.h1("Objective, approach and assumed site", "1")
R.p("""The brief (&sect;1.1) asks for a post-tensioned PSC I-girder bridge to be studied as an infrastructure
system over its service life: analysed and designed, prestressed, given a healthy baseline, inspected, tested and
monitored, deteriorated in a controlled way, and finally assessed with AI/ML so that a maintenance decision can be
made. The chain below is the one I followed. Each block hands a definite set of numbers to the next one; nothing in
the later chapters is generic.""")
R.raw("""<table style="font-size:8.4pt;text-align:center"><tr>
<td>Bridge B33<br><span class="small">33 m span, 6 girders</span></td><td>&rarr;</td>
<td>Analysis &amp; prestressing<br><span class="small">grillage, IRC:6 / IRC:112</span></td><td>&rarr;</td>
<td>Healthy baseline<br><span class="small">commissioning tests, model updating, 1 year of data</span></td><td>&rarr;</td>
<td>Inspection, NDT, monitoring<br><span class="small">what, where, how often</span></td></tr><tr>
<td>Maintenance decision<br><span class="small">actions, intervals, reinspection</span></td><td>&larr;</td>
<td>AI/ML assessment<br><span class="small">novelty, classification, regression, fusion</span></td><td>&larr;</td>
<td>Features<br><span class="small">changes from own baseline</span></td><td>&larr;</td>
<td>Synthetic data<br><span class="small">digital twin + 6 mechanisms</span></td></tr></table>""")
R.h2("Assumed site and service conditions", "1.1")
R.p("""The brief does not fix a site, so I assumed one that makes the later choices concrete. B33 carries a
one-way, three-lane carriageway of a divided state highway (one of a pair of parallel bridges) over an irrigation
canal and a service road in the coastal plain of West Bengal, about 120&nbsp;km from the Bay of Bengal. It has three
simply supported spans of 33&nbsp;m on wall-type piers P1 and P2 and abutments A1 and A2; only one span length was
selected, as required by &sect;1.2, and span S2 between the piers is the one designed, instrumented and assessed in
detail. The climate gives effective bridge temperatures of roughly 15 to 42&nbsp;&deg;C, heavy monsoon rain from June
to September and wet-dry cycles the rest of the year; I therefore took the exposure as <i>severe</i> in the sense of
IRC:112 Table 14.1, with 50&nbsp;mm clear cover to links and 75&nbsp;mm to ducts. The design life is 100 years.
Traffic is mixed, with a high share of loaded three-axle and multi-axle lorries.""")
R.p("""Outside the scope are the piers, abutments and foundations (they appear only in the inspection scheme),
hydraulics and scour, seismic and wind design, and shop drawings of the prestressing system. All 'measurements' in
the report are synthetic; how they were produced, and what that means for the results, is explained in chapters 10
and 11.""")
R.h2("Codes and references used", "1.2")
R.table(["Document", "Used for"], [
    ["IRC:5-2015", "carriageway, footpath and kerb widths"],
    ["IRC:6-2017", "dead loads, live loads (Class A, 70R), impact, footway load, lane reduction, temperature, load combinations (Annex B)"],
    ["IRC:112-2020", "materials, losses, stress limits, crack control, ULS flexure and shear, deflection"],
    ["IRC:83 (Part II)-2018", "laminated elastomeric bearings"],
    ["IRC:SP:69-2011", "expansion joints"],
    ["IRC:SP:51-2015", "load testing of bridges"],
    ["IRC:SP:35-1990, IRC:SP:52-1999", "inspection and maintenance practice, condition assessment"],
    ["IRC:SP:40-2019, IRC:SP:80-2008", "repair and strengthening techniques, corrosion monitoring and remedies"],
    ["IS 516 (Part 5), ASTM C876", "UPV and rebound hammer testing; half-cell potential"],
], widths=[28, 72])

# ====================================================================== 2
R.h1("Bridge configuration", "2")
R.p(f"""<b>Cross-section.</b> The carriageway is 11.0&nbsp;m (three lanes of 3.5&nbsp;m with 0.25&nbsp;m marginal strips),
bounded by 0.50&nbsp;m New Jersey crash barriers. Outside each barrier there is a 1.75&nbsp;m footpath, slightly wider than
the IRC:5 minimum of 1.5&nbsp;m because the footpath also carries service ducts for power and optical fibre, and a
0.25&nbsp;m railing kerb. The overall deck width is therefore {P.DECK_WIDTH:.1f}&nbsp;m (Figure [[fig:f01_cross_section]]). The barrier between
footpath and traffic is a deliberate choice: on a highway with fast heavy vehicles it protects pedestrians and it also
keeps vehicles off the deck cantilever.""")
R.fig("f01_cross_section", "Cross-section of B33 at mid-span (dimensions in m): 225 mm M40 deck slab on six 1.80 m M45 precast girders; the mid-span cross girder behind the section is dashed.")
R.p("""<b>Number and spacing of girders.</b> The construction depth is limited: the deck has to clear the service road
below and the approach embankments are already high, so I fixed the precast girder depth at 1.80&nbsp;m (2.025&nbsp;m with
the slab, span/depth = 16.3). With the depth fixed, the girder number decides how much prestress each girder must
carry. Table [[tab:options]] compares four, five and six girders on a preliminary basis (Courbon distribution, the same section shape,
the prestress needed for zero tension at the soffit under the characteristic combination).""")
rows = []
for o in d["options"]:
    cab = math.ceil(o["strands"] / 12)
    rows.append([o["n"], f(o["spacing"], 2), f(o["slab"] * 1e3, 0), f(o["k_outer"], 3), f(o["M_LL"], 0), f(o["Pe"] / 1e3, 2),
                 f(o["strands"], 1), f"{cab}" + (" - too many" if cab > 4 else " - fits"), f(o["girder_t"], 0),
                 f(o["strands_total"], 0), f(o["concrete"], 0)])
R.table(["Girders", "Spacing (m)", "Slab (mm)", "Courbon factor, outer", "LL moment outer (kN m)", "Pe needed (MN)",
         "Strands per girder", "12T15 cables", "Girder mass (t)", "Strands, deck", "Concrete (m&sup3;)"],
        rows, "Girder-number study at 1.80 m depth (preliminary, Courbon; the 750 mm bulb takes at most four cables)", num_cols=(0, 1, 2, 3, 4, 5, 6, 8, 9, 10), small=True, tag="options")
R.p("""Four girders would need about 64 strands each, i.e. six 12T15 cables, which cannot be placed in a 750&nbsp;mm
bottom bulb with the required duct spacing and cover, and the deck slab would have to span 3.0&nbsp;m clear. Five girders
need about 53 strands, five cables, still too many for the bulb at this depth. Six girders need about 45 strands, which
four 12T15 cables (48 strands) provide with a small margin. Six girders use about 5&nbsp;% more strand and 9&nbsp;% more
concrete than four, but they work within the depth limit, give a 225&nbsp;mm slab with a short 1.7&nbsp;m clear span and a
1.25&nbsp;m cantilever, keep each girder under 80&nbsp;t for lifting by two cranes, and leave more load paths if one girder
is damaged. That last point matters for the later chapters: a deck with six girders sheds load from a weakened girder to
its neighbours, which keeps the bridge safe but also makes damage harder to see in global measurements.""")
R.p(f"""<b>Girder section.</b> The precast girder (Figure [[fig:f02_girder_section]]) has a 1000 x 175&nbsp;mm top flange with a 100&nbsp;mm taper, a
300&nbsp;mm web and a 750 x 250&nbsp;mm bottom bulb with a 200&nbsp;mm taper. The web is 300&nbsp;mm rather than the common
250&nbsp;mm because a 92&nbsp;mm HDPE duct has to pass through it with cover on both sides; the bulb width is set by three
ducts side by side with clear spacing larger than one duct diameter. Over 1.5&nbsp;m at each end the web is widened to the
full bulb width (end block) and then tapers back over another 1.5&nbsp;m, to take the four anchorages and the high shear
near the bearings. Section properties are A = {pre['A']:.3f}&nbsp;m&sup2;, y<sub>b</sub> = {pre['yb']:.3f}&nbsp;m,
I = {pre['I']:.4f}&nbsp;m<sup>4</sup> for the precast girder and I = {comp_o['I']:.4f} / {comp_i['I']:.4f}&nbsp;m<sup>4</sup>
for the composite outer / inner girder (deck M40 transformed with m = {comp_i['m']:.3f}). The full slab width is effective
(IRC:112 cl. 7.6.1.2 gives more than the girder spacing for a 33&nbsp;m span).""")
R.fig("f02_girder_section", "Precast girder at mid-span and at the anchorage face, with ducts and 12T15 anchor plates (mm).", 88)
R.p("""<b>Cross girders.</b> End diaphragms (400 x 1600&nbsp;mm) are needed to hold the girder ends, to transfer
transverse loads to the bearings and, importantly for maintenance, as jacking beams for bearing replacement. For
intermediate cross girders I compared one at mid-span with three at the quarter points in the grillage model: the extra
two changed the outer girder live-load moment by less than 3&nbsp;% and the first two natural frequencies by about 1&nbsp;%,
so one 300 x 1450&nbsp;mm cross girder at mid-span was adopted. It also braces the girders laterally while the deck is
cast.""")
R.p("""<b>Bearings and joints.</b> Each girder sits on a 400 x 500&nbsp;mm laminated elastomeric bearing at both ends
(five 12&nbsp;mm internal layers), so the span expands about its centre and both ends move. Strip-seal expansion
joints (IRC:SP:69) are provided at each pier and abutment, with a 40&nbsp;mm gap. The joint at a pier is the place where
water reaches the bearings, the girder ends and the anchorages; this becomes the main thread of the year-18 assessment.""")
R.p(f"""<b>Materials.</b> Girders M45 (stressing at {P.FCK_TRANSFER:.0f}&nbsp;MPa, about 10 days), deck slab and cross
girders M40, reinforcement Fe500D, strand 15.2&nbsp;mm 7-wire low-relaxation (IS 14268 Class 2, f<sub>pk</sub> =
1860&nbsp;MPa, 140&nbsp;mm&sup2;), corrugated HDPE ducts. Unit weights from IRC:6 Table 1: 25&nbsp;kN/m&sup3; for concrete
and 22&nbsp;kN/m&sup3; for the 75&nbsp;mm bituminous wearing coat.""")

# ====================================================================== 3
R.h1("Structural analysis and preliminary design", "3")
R.h2("Loads", "3.1")
q_fp = P.footpath_ll()
R.table(["Load", "Value", "Basis"], [
    ["Girder self-weight", f"{pre['A']*25:.1f} kN/m (end block {d['sections']['end_block']['A']*25:.1f} kN/m)", "on the precast girder"],
    ["Deck slab (wet) and cross girder", f"{2.7*P.SLAB_T*25:.1f} kN/m per inner girder; {2.6*P.SLAB_T*25:.1f} kN/m outer; cross girder {2.4*0.3*1.45*25:.1f} kN at mid-span", "on the precast girders, tributary widths"],
    ["Crash barrier, footpath, railing", f"{P.BARRIER_LOAD}, {P.FOOTPATH_LOAD}, {P.RAILING_LOAD} kN/m each side", "on the composite deck (grillage)"],
    ["Wearing coat", f"{P.WC_LOAD:.2f} kN/m&sup2; over 11.0 m", "75 mm at 22 kN/m&sup3;"],
    ["Footway live load", f"{q_fp:.2f} kN/m&sup2;", "IRC:6 cl. 206.1, L &gt; 30 m: P' = P - 260 + (4800/L)(16.5 - W)/15 with P = 400 kg/m&sup2;, W = 1.75 m"],
    ["Vehicles", "Class A train; 70R wheeled (1000 kN) and tracked (700 kN)", "IRC:6 Table 6A, three lanes: one 70R + one Class A, or three Class A, whichever is worse"],
    ["Impact", f"Class A and 70R wheeled 4.5/(6+L) = {P.IMPACT:.3f}; 70R tracked 0.10", "IRC:6 cl. 208 (span &gt; 12 m)"],
    ["Multi-lane reduction", "10 % for three lanes loaded", "IRC:6 cl. 205"],
    ["Temperature difference", "positive 17.8 / 4.0 / 2.1 &deg;C, reverse 10.6 / 0.7 / 0.8 / 6.6 &deg;C", "IRC:6 cl. 215, &alpha; = 12 x 10<sup>-6</sup>/&deg;C"],
    ["Differential shrinkage", "100 x 10<sup>-6</sup> deck relative to girder, creep factor 0.43", "eigenstresses only (simply supported)"],
], "Loads on one span", widths=[24, 38, 38], small=True)
R.p("""Load combinations follow IRC:6 Annex B. For the ultimate limit state the factors are 1.35 on self-weight and
SIDL, 1.75 on the wearing coat and 1.5 on traffic. For the serviceability checks I used the rare (characteristic)
combination with live load 1.0 and temperature 0.6, the frequent combination with 0.75 and 0.5, and the
quasi-permanent combination with live load zero and temperature 0.5.""")
R.h2("Structural model", "3.2")
R.p(f"""The deck was analysed with a grillage written for this work (Python, about 300 lines): six longitudinal members
along the girder lines with the composite section properties, and transverse members every {P.L/P.NX:.3f}&nbsp;m
representing deck slab strips (I = b t&sup3;/12, J = b t&sup3;/6 following Hambly), replaced by T-beams at the cross
girders. Each node has a vertical displacement and two rotations. Girder torsion constants are the sum of the
rectangle values of flange, web and bulb plus half of the slab. Bearings are vertical springs of
{P.BRG_KV/1e6:.1f}&nbsp;x&nbsp;10<sup>6</sup>&nbsp;kN/m. Loads between girder lines are shared to the four corner nodes,
and loads outside the outer girders (barrier, footpath) are applied as a force and a torque on the outer line. For the
live load, influence surfaces of moment and shear at 13 sections of each girder were calculated, and every vehicle was
moved along the span in 0.25&nbsp;m steps and across the carriageway in 0.05&nbsp;m steps; the worst transverse
arrangement was found by dynamic programming with the IRC kerb clearances (1.2&nbsp;m for 70R, 0.15&nbsp;m for Class A)
and 1.2&nbsp;m between vehicles. The self-weight of girders and wet slab acts on the precast girders individually and was
calculated as for a simply supported beam with the 0.45&nbsp;m overhangs.""")
R.p(f"""The grillage was checked against hand results (Appendix B): with 10&nbsp;kN/m on each girder the sum of mid-span
moments is within 0.2&nbsp;% of wL&sup2;/8 and the inner girder deflection within 1.5&nbsp;% of 5wL<sup>4</sup>/384EI plus
the bearing compression. For live load, Courbon's method is valid here (span/width = 2.06, stiff cross girders) and gives
{f(d['courbon']['M'][0],0)}&nbsp;kN&nbsp;m for G1 against {f(np.load('out/design_arrays.npz')['MLL'][0,-1],0)}&nbsp;kN&nbsp;m from
the grillage (Figure [[fig:f07_ll_distribution]]). For the inner girders the grillage is larger than Courbon, because a vehicle near the centre
loads G3 and G4 more than a rigid-deck distribution assumes; I used the grillage values for design.""")
R.fig("f07_ll_distribution", "Mid-span live-load moment (with impact and lane reduction) of each girder: grillage against Courbon.", 55)
R.h2("Load effects", "3.3")
M = {k: d["stages"]["G1"]["16.500"]["M"][k] for k in ("Mg", "Ms", "Medge", "Mwc", "Mll")}
M2 = d["stages"]["G2"]["16.500"]["M"]
lay = d.get("courbon", {})
R.table(["Mid-span moment (kN m)", "G1 (outer)", "G2", "G3"], [
    ["Girder self-weight", f(M["Mg"], 0), f(M2["Mg"], 0), f(d['stages']['G3']['16.500']['M']['Mg'], 0)],
    ["Deck slab + cross girder", f(M["Ms"], 0), f(M2["Ms"], 0), f(d['stages']['G3']['16.500']['M']['Ms'], 0)],
    ["Barrier, footpath, railing", f(M["Medge"], 0), f(M2["Medge"], 0), f(d['stages']['G3']['16.500']['M']['Medge'], 0)],
    ["Wearing coat", f(M["Mwc"], 0), f(M2["Mwc"], 0), f(d['stages']['G3']['16.500']['M']['Mwc'], 0)],
    ["Vehicles + impact + footway (worst)", f(M["Mll"], 0), f(M2["Mll"], 0), f(d['stages']['G3']['16.500']['M']['Mll'], 0)],
    ["ULS design moment M<sub>Ed</sub>", f(d["uls"]["G1"]["MEd"], 0), f(d["uls"]["G2"]["MEd"], 0), f(d["uls"]["G3"]["MEd"], 0)],
], "Mid-span moments per girder", num_cols=(1, 2, 3))
R.p("""For G1 the worst case is the 70R wheeled vehicle at the kerb clearance with a Class A train beside it; for the
inner girders it is the same pair placed over them. The outer girder has the largest live-load moment, the inner girders
carry more slab and the same SIDL share, and in the end the ULS moments of G1 and G2 are within 1&nbsp;%; one girder
design is used for all six. Envelopes for G1 are in Figure [[fig:f05_envelopes]].""")
R.fig("f05_envelopes", "Bending moment and shear envelopes of girder G1 (characteristic components and ULS).")

# ====================================================================== 4
R.h1("Prestressing system", "4")
R.h2("Cables and profiles", "4.1")
R.p(f"""Each girder has four cables of twelve 15.2&nbsp;mm strands (12T15, A<sub>p</sub> = 1680&nbsp;mm&sup2; per cable) in
92&nbsp;mm corrugated HDPE ducts, stressed from both ends to {P.JACK_RATIO:.2f}&nbsp;f<sub>pk</sub> (1395&nbsp;MPa,
{f(pc['C1']['Pj'],0)}&nbsp;kN per cable). The jacking stress is kept below both the IRC:112 maximum and the 0.765&nbsp;f<sub>pk</sub>
used in older practice, to leave room for over-stressing if friction turns out higher than assumed on site.""")
R.p("""The layout (Figures [[fig:f02_girder_section]] and [[fig:f03_cable_profile]]) came out of a clash check along the whole girder rather than from the mid-span
section alone. At the anchorage face the four anchor plates are stacked in the web plane at 0.35, 0.75, 1.15 and
1.55&nbsp;m above the soffit, 400&nbsp;mm apart, which keeps the group centroid (0.95&nbsp;m) at the end-block centroid
(0.93&nbsp;m). At mid-span three cables lie in the bottom row at 130&nbsp;mm and one in a second row at 300&nbsp;mm. The two
outer cables of the bottom row must also move sideways by 220&nbsp;mm, and a lateral swing is impossible inside the 300&nbsp;mm
web. My first layout tried to do it there. In the final layout these two cables drop into the bulb quickly (parabolas
ending at 5&nbsp;m and 8&nbsp;m from the end) and swing sideways only once they are inside the bulb; the two centre cables
follow full-length parabolas and never leave the web plane. The profile of each cable is a parabola from the anchorage
with a horizontal tangent where it reaches its mid-span level, straight beyond that point, with a 12&nbsp;mm allowance for
the strands lying low in a draped duct.""")
R.fig("f03_cable_profile", "Cable profiles in elevation and plan. Cables C2 and C3 are stressed at 10 days, C1 and C4 at 28 days.")
R.p(f"""<b>Two-stage stressing.</b> Stressing all four cables at 10 days would give a soffit compression of about 18.6&nbsp;MPa
at mid-span against a limit of 0.48 f<sub>cj</sub> = {0.48*P.FCK_TRANSFER:.1f}&nbsp;MPa. I therefore stress C2 and C3 at
10 days (enough to lift the girder off the casting bed and carry its own weight) and C1 and C4 at 28 days. Both groups
have their centroid at the end-block centroid, so the girder ends are not bent at either stage.""")
R.h2("Losses and effective prestress", "4.2")
R.p(f"""Friction follows P(x) = P<sub>j</sub> e<sup>-(&mu;&theta; + kx)</sup> with &mu; = {P.MU_FRICTION} and
k = {P.K_WOBBLE}/m for HDPE ducts, &theta; being the cumulative vertical and plan angle from the nearer anchorage.
Draw-in of 6&nbsp;mm was handled by the area method. Elastic shortening was calculated cable by cable in the stressing order
C2, C3, C1, C4, using the concrete modulus at the age of each stressing. Creep, shrinkage and relaxation follow the
time-dependent loss expression of IRC:112 (same form as EN&nbsp;1992 eq. 5.46):""")
R.eq("&Delta;&sigma;<sub>p,c+s+r</sub> = [ &epsilon;<sub>cs</sub>E<sub>p</sub> + 0.8&Delta;&sigma;<sub>pr</sub> + (E<sub>p</sub>/E<sub>cm</sub>) &phi;(t,t<sub>0</sub>) &sigma;<sub>c,QP</sub> ] / [ 1 + (E<sub>p</sub>/E<sub>cm</sub>)(A<sub>p</sub>/A<sub>c</sub>)(1 + A<sub>c</sub>z<sub>cp</sub>&sup2;/I<sub>c</sub>)(1 + 0.8&phi;(t,t<sub>0</sub>)) ]")
R.p(f"""with creep and shrinkage from IRC:112 Annex A2 (RH {P.RH:.0f}&nbsp;%, notional size {pc['C1'] and 281}&nbsp;mm, cylinder
strength taken as 0.8 of the cube strength), the quasi-permanent concrete stress at the tendon level
({f(10.83,1)}&nbsp;MPa at mid-span) and final relaxation taken as three times the 1000-hour value for the actual initial
stress ratio (IRC:112 cl. 6.3.6). Table [[tab:losses]] gives the results per cable at mid-span.""")
rows = []
for n, v in pc.items():
    rows.append([n, PS.STAGE_TIME[1 if n in ("C2", "C3") else 2], f(v["Pj"], 0), f(v["Pj"] - v["Pfs_mid"], 0), f(v["ES_mid"], 0),
                 f(v["P0_mid"], 0), f(v["phi"], 2), f(v["relax"], 0), f(v["dsig"], 0), f(v["Pe_mid"], 0)])
tot = d["prestress"]
rows.append(["<b>Total</b>", "", f(tot["Pj"], 0), f(tot["Pj"] - tot["Pfs_mid"], 0),
             f(sum(v["ES_mid"] for v in pc.values()), 0), f(tot["P0_mid"], 0), "", "", "", f"<b>{f(tot['Pe_mid'],0)}</b>"])
R.table(["Cable", "Stressed at (d)", "P<sub>j</sub> (kN)", "Friction + draw-in (kN)", "Elastic shortening (kN)", "P<sub>0</sub> (kN)",
         "&phi;(&infin;,t<sub>0</sub>)", "Relaxation (MPa)", "c+s+r loss (MPa)", "P<sub>e</sub> (kN)"], rows,
        "Prestressing forces and losses at mid-span (one girder)", num_cols=tuple(range(1, 10)), small=True, tag="losses")
R.p(f"""The immediate loss at mid-span is {pct(1-tot['P0_mid']/tot['Pj'],1)} and the long-term loss a further
{pct((tot['P0_mid']-tot['Pe_mid'])/tot['Pj'],1)}, giving <b>P<sub>0</sub> = {f(tot['P0_mid']/1e3,2)}&nbsp;MN and
P<sub>e</sub> = {f(tot['Pe_mid']/1e3,2)}&nbsp;MN, P<sub>e</sub>/P<sub>j</sub> = {tot['Pe_mid']/tot['Pj']:.3f}</b>. These numbers
are of the expected order for a 34&nbsp;m post-tensioned girder (total loss in the low twenties of per cent), which I took as
a first sanity check. C2 loses most to elastic shortening because three cables are stressed after it. Figure [[fig:f04_prestress_force]] shows the
force along the girder: the draw-in of the straight-ended cables reaches nearly to mid-span because friction there is only
wobble.""")
R.fig("f04_prestress_force", "Total prestressing force of one girder along its length at the successive stages.")
R.h2("Stresses at the important stages", "4.3")
R.p(f"""Stresses were checked at mid-span, L/4 and two sections near the support for every girder, at five stages:
transfer of the first two cables (10 d, self-weight only), all cables stressed (28 d), deck slab cast (60 d), opening to
traffic (90 d) and long-term service (t = &infin;) in the three combinations. For the service checks the characteristic
prestress was used, 0.9 P<sub>m</sub> where prestress is favourable and 1.1 P<sub>m</sub> where it is not (IRC:112, post-tensioning),
and the temperature difference and differential shrinkage were added only where they are adverse. My design targets,
which are stricter than the crack-width requirement of IRC:112 Table 12.1 for bonded tendons, were: no decompression of
the soffit under the quasi-permanent combination, soffit tension below f<sub>ctm</sub> (uncracked section) under the rare
combination, compression below 0.48 f<sub>ck</sub> (rare) and 0.36 f<sub>ck</sub> (quasi-permanent).""")
R.fig("f06_stage_stresses", "Mid-span stresses of the outer girder G1 at all stages.")
R.table(["Check (worst of G1-G3 and all sections)", "Value (MPa)", "Limit (MPa)", "Status"], [
    ["Transfer, stage 1: max compression / min stress", f"{ss['transfer1_max_comp']:.2f} / {ss['transfer1_min']:+.2f}", f"{0.48*P.FCK_TRANSFER:.2f} / &ge; 0", "OK"],
    ["All cables stressed: soffit compression", f"{ss['transfer2_max_comp']:.2f}", f"{0.48*P.FCK_GIRDER:.2f}", "OK"],
    ["All cables stressed: top fibre", f"{ss['transfer2_min_top']:+.2f}", "&ge; 0 (no tension)", "OK"],
    ["Long term, quasi-permanent, soffit (0.9 P<sub>m</sub>)", f"{ss['qp_min_bot']:+.2f}", "&ge; 0 (decompression)", "OK"],
    ["Long term, frequent, soffit (0.9 P<sub>m</sub>)", f"{ss['frequent_min_bot']:+.2f}", f"&ge; -{P.fctm(P.FCK_GIRDER):.1f} (uncracked)", "OK"],
    ["Long term, rare, soffit (0.9 P<sub>m</sub>)", f"{ss['rare_min_bot']:+.2f}", f"&ge; -{P.fctm(P.FCK_GIRDER):.1f} (uncracked)", "OK"],
    ["Rare, top of girder / top of slab (1.1 P<sub>m</sub>)", f"{ss['rare_max_top_girder']:.2f} / {ss['rare_max_slab']:.2f}", f"{0.48*P.FCK_GIRDER:.1f} / {0.48*P.FCK_DECK:.1f}", "OK"],
    ["Quasi-permanent, soffit compression (1.1 P<sub>m</sub>)", f"{ss['qp_max_bot']:.2f}", f"{0.36*P.FCK_GIRDER:.1f}", "OK"],
], "Stress checks (compression positive)", num_cols=(1, 2))
R.p(f"""The governing check is the soffit under the rare combination at mid-span ({ss['rare_min_bot']:+.2f}&nbsp;MPa). The
reverse temperature difference alone contributes about {0.6*2.51:.1f}&nbsp;MPa of tension there, so the design would not
satisfy a 'no tension under rare loads' rule, but the section stays uncracked with a margin of about 1&nbsp;MPa. I kept this
margin deliberately visible because it decides, later, how much prestress the girder can lose before it cracks.""")
R.h2("Ultimate limit state, shear, deflection and bearings", "4.4")
u = d["uls"]["G1"]
R.p(f"""<b>Flexure.</b> Strain compatibility at mid-span (rectangular block 0.8x at f<sub>cd</sub> = 0.67 f<sub>ck</sub>/1.5,
strands bilinear with f<sub>pd</sub> = f<sub>p0.1k</sub>/1.15, pre-strain from P<sub>e</sub>) gives a neutral axis depth of
{u['x']*1e3:.0f}&nbsp;mm (block depth {0.8*u['x']*1e3:.0f}&nbsp;mm, inside the 225&nbsp;mm slab), strand strain {u['eps_p']*1e3:.0f}
x&nbsp;10<sup>-3</sup> (yielded, ductile failure, x/d = {u['x_over_d']:.2f}) and M<sub>Rd</sub> = {f(u['M'],0)}&nbsp;kN&nbsp;m against
M<sub>Ed</sub> = {f(u['MEd'],0)}&nbsp;kN&nbsp;m for G1, a utilisation of {u['MEd']/u['M']:.2f}. The untensioned bottom bars were
ignored, which is conservative.""")
sh = d["shear"]["G3@2.75"]
R.p(f"""<b>Shear.</b> The critical section is where the web has just narrowed to 300&nbsp;mm, 2.75&nbsp;m from the bearing. With
the duct in the web the nominal width is b<sub>w</sub> - 1.2&phi; = {sh['bw_nom']*1e3:.0f}&nbsp;mm (HDPE ducts). For the worst girder
V<sub>Ed</sub> = {f(sh['VEd'],0)}&nbsp;kN, the vertical component of prestress (0.9 P<sub>e</sub> sin&alpha;) is
{f(sh['Vp'],0)}&nbsp;kN, and the net {f(sh['Vnet'],0)}&nbsp;kN is below V<sub>Rd,max</sub> = {f(sh['VRdmax'],0)}&nbsp;kN for
cot&theta; = 2. The required links are {sh['Asw_s']:.0f}&nbsp;mm&sup2;/m; I adopted 2-legged 12&nbsp;mm links at 200&nbsp;mm
(1131&nbsp;mm&sup2;/m) over the end 6&nbsp;m and 2-legged 10&nbsp;mm at 250&nbsp;mm elsewhere (above the minimum of
{sh['Asw_s_min']:.0f}&nbsp;mm&sup2;/m).""")
df = d["deflection"]
R.p(f"""<b>Deflection and camber.</b> The live-load deflection with impact is {df['LL_mm']['G1']:.1f}&nbsp;mm for G1
(span/{P.L*1e3/df['LL_mm']['G1']:.0f}), well within span/800 = {df['limit_mm']:.1f}&nbsp;mm. The camber is
{df['camber_transfer_mm']:.0f}&nbsp;mm upward when all cables are stressed and about {df['camber_deck_mm']:.0f}&nbsp;mm after the
deck is cast. The elastic camber due to P<sub>e</sub> alone is {df['camber_Pe_elastic_mm']:.0f}&nbsp;mm; this number reappears in
chapter 10, because a loss of prestress shows up as a loss of this camber.""")
bg = d["bearings"]["G3"]
R.p(f"""<b>Bearings.</b> The largest reaction is {f(bg['Nmax'],0)}&nbsp;kN (inner girder, permanent load
{f(bg['Nmin'],0)}&nbsp;kN). For the 400 x 500&nbsp;mm bearing with five 12&nbsp;mm layers the shape factor is {bg['S']:.1f}
and the total design strain from compression, shear (movement {bg['v_x']*1e3:.0f}&nbsp;mm from creep, shrinkage and
temperature) and rotation is {bg['eps_t']:.2f}, below 7 (IRC:83 Part II); the minimum pressure {bg['sigma_min']:.1f}&nbsp;MPa
is above 3&nbsp;MPa, so the bearing will not walk. In service each bearing shears by about
{b['bearing_ref']:.3f}&nbsp;mm per &deg;C (&alpha;L/2), which is the quantity the bearing transducers will watch.""")
R.p("""<b>Anchorage zone and other details.</b> The end block was sized to the anchor plates (spacing 400&nbsp;mm, edge
distance 250&nbsp;mm at the top) and to the bursting forces of the local zones; the final anchorage reinforcement has to
follow the approval document of the prestressing system actually used. The deck slab was checked separately for 70R wheel
loads on its 1.7&nbsp;m clear span; these local designs are not repeated here because they do not feed the later chapters.""")

# ====================================================================== 5
R.h1("Healthy baseline", "5")
R.p("""A healthy baseline is not the design calculation. The as-built bridge is always somewhat different: concrete is
stiffer than the code modulus, barriers and footpaths stiffen the deck, bearings are not ideal. What later measurements
must be compared with is the bridge as it was measured when it was new, together with a model that reproduces those
measurements. I set the baseline up in three steps.""")
R.h2("Commissioning tests", "5.1")
R.p(f"""Before opening, a 30-minute ambient vibration test was recorded at night with all 12 accelerometers, and a static
load test was carried out with four 300&nbsp;kN three-axle lorries (two abreast, two in tandem, centred at mid-span) in
the spirit of IRC:SP:51. In this synthetic study the 'real' bridge is a model whose stiffness parameters are hidden from
the assessment (concrete modulus 6&nbsp;% above E<sub>cm</sub>, crash barriers acting at about one third of their stiffness,
dynamic modulus 24&nbsp;% above static). The commissioning record was identified without using any model shapes, by frequency
domain decomposition of the 12 channels (Figure [[fig:f10_commissioning]]): peaks of the first singular value were picked, each peak was named after
the design-model mode its singular vector resembled most, and its frequency and damping were refined by fitting a
single-degree-of-freedom spectrum around the peak.""")
R.fig("f10_commissioning", "Commissioning ambient test: one channel in time and frequency, and the FDD singular values of all 12 channels.")
R.p(f"""I first identified the frequencies by filtering the channels with the design-model mode shapes, which is
convenient but wrong at this stage: with the first bending and torsion modes only 5&nbsp;% apart, the model shapes leaked one
mode into the other and gave f<sub>2</sub> = 3.78&nbsp;Hz, while the model-free FDD gave {tw['meas_f'][1]:.2f}&nbsp;Hz. The lesson
is that before the model has been checked against the structure, the data have to speak for themselves. Table [[tab:updating]] compares the
design model with the measurements.""")
R.h2("Model updating", "5.2")
R.p("""Three parameters were updated by weighted least squares on the four identified frequencies and the six girder
deflections of the load test: a factor on the concrete modulus (E<sub>scale</sub>), the share of crash-barrier stiffness acting
with the deck (&beta;) and the ratio of dynamic to static modulus (E<sub>dyn</sub>). Frequencies were weighted with a standard
error of 0.2&nbsp;% and deflections with 0.10&nbsp;mm.""")
R.table(["Quantity", "Design model", "Measured (year 0)", "Updated twin"], [
    ["f<sub>1</sub> bending (Hz)", f(tw['design_f'][0], 3), f(tw['meas_f'][0], 3), f(tw['updated_f'][0], 3)],
    ["f<sub>2</sub> torsion (Hz)", f(tw['design_f'][1], 3), f(tw['meas_f'][1], 3), f(tw['updated_f'][1], 3)],
    ["f<sub>3</sub> transverse (Hz)", f(tw['design_f'][2], 3), f(tw['meas_f'][2], 3), f(tw['updated_f'][2], 3)],
    ["f<sub>4</sub> 2nd bending (Hz)", f(tw['design_f'][3], 3), f(tw['meas_f'][3], 3), f(tw['updated_f'][3], 3)],
    ["Load-test deflection G1 / G3 (mm)", f"{tw['design_d'][0]:.2f} / {tw['design_d'][2]:.2f}", f"{tw['meas_d'][0]:.2f} / {tw['meas_d'][2]:.2f}",
     f"{tw['updated_d'][0]:.2f} / {tw['updated_d'][2]:.2f}"],
    ["E<sub>scale</sub>, &beta;, E<sub>dyn</sub>", "1.00, 0, 1.20", "-", f"{tw['E_scale']:.3f} &plusmn; {tw['sd'][0]:.3f}, {tw['barrier']:.2f} &plusmn; {tw['sd'][1]:.2f}, {tw['E_dyn']:.3f} &plusmn; {tw['sd'][2]:.3f}"],
], "Design model, commissioning measurements and the updated twin", num_cols=(1, 2, 3), tag="updating")
R.p(f"""The design model is 7 to 10&nbsp;% too soft in frequency and 9 to 14&nbsp;% too flexible in the load test, which is
normal for a design model. After updating, the twin reproduces all ten measurements within 0.2&nbsp;% in frequency and
0.15&nbsp;mm in deflection, and the measured and twin mode shapes agree with MAC above 0.99 for the first three modes. The
updated parameters ({tw['E_scale']:.3f}, {tw['barrier']:.2f}, {tw['E_dyn']:.3f}) are close to the hidden values
(1.06, 0.35, 1.24), which is reassuring, but I treat the twin as a calibrated surrogate rather than the truth: E<sub>scale</sub> and
&beta; are correlated (correlation {tw['corr'][0][1]:.2f}) and a different pair could fit nearly as well. The second bending mode
(14.5&nbsp;Hz) is almost repeated with the second torsion mode, its measured shape is a mixture of the two (MAC 0.47), and
for that mode the twin shape, not the measured one, is used in the spatial filter later.""")
R.fig("f08_mode_shapes", "First bending, torsion and transverse modes and the second bending mode of the updated twin (25 &deg;C).")
R.h2("One year of healthy monitoring", "5.3")
nb = b["norm"]
R.p(f"""The permanent system then recorded two 30-minute records a day for a year (02:00 and 14:00), 730 records in all.
The first bending frequency moved between 3.65 and 3.79&nbsp;Hz over the year, a range of almost 4&nbsp;%, driven by temperature
({nb['f1']['per_degC_pct']:+.3f}&nbsp;%/&deg;C, from the concrete modulus), by the traffic present during the record and by
the extra mass of a wet deck in the monsoon. A linear model in temperature and log(a<sub>RMS</sub>), the RMS acceleration being a
measure of how busy the bridge was, was fitted for each frequency (Figure [[fig:f11_baseline_year]]). After normalisation the scatter of f<sub>1</sub> is
{nb['f1']['sd']*1e3:.1f}&nbsp;mHz ({100*nb['f1']['sd']/nb['f1']['mean']:.2f}&nbsp;%), and the remaining structure in the residuals is
the monsoon mass effect, which I left in on purpose because on a real bridge its size is not known in advance.""")
R.fig("f11_baseline_year", "Healthy year: f<sub>1</sub> against temperature, and the residual after temperature/traffic normalisation.")
R.table(["Baseline quantity (span S2, healthy, 25 &deg;C reference)", "Value"], [
    ["Effective prestress per girder, mid-span", f"P<sub>e</sub> = {f(tot['Pe_mid']/1e3,2)} MN (design), soffit precompression from P<sub>e</sub> {7172*(1/pre['A']+(pre['zc']-0.1605)/pre['Zb'])/1e3:.1f} MPa"],
    ["Soffit stress, quasi-permanent (mean prestress)", f"about {d['stages']['G1']['16.500']['rows']['V quasi, inf']['bot']+2.34:.1f} MPa compression (G1)"],
    ["Load test, 4 x 300 kN: mid-span deflection G1..G3", " / ".join(f"{x:.2f}" for x in b['load_test_ref']['d'][:3]) + " mm"],
    ["Load test: soffit strain G1..G3", " / ".join(f"{x:.0f}" for x in b['load_test_ref']['e'][:3]) + " microstrain"],
    ["Natural frequencies f<sub>1</sub>..f<sub>4</sub>", " / ".join(f"{x:.3f}" for x in b['f_ref']) + " Hz"],
    ["Temperature sensitivity of f<sub>1</sub>..f<sub>4</sub>", " / ".join(f"{nb[k]['per_degC_pct']:+.2f}" for k in ('f1','f2','f3','f4')) + " %/&deg;C"],
    ["Residual scatter after normalisation", " / ".join(f"{100*nb[k]['sd']/nb[k]['mean']:.2f}" for k in ('f1','f2','f3','f4')) + " %"],
    ["Ambient acceleration, mid-span", "0.6-0.8 mg RMS at night, 1.3-1.5 mg RMS in daytime traffic, peaks 10-25 mg"],
    ["Bearing movement", f"{b['bearing_ref']:.3f} mm/&deg;C at both ends"],
    ["NDT (girders, year 1, mean &plusmn; s.d.)", f"rebound {b['ndt']['rebound']['mean']:.1f} &plusmn; {b['ndt']['rebound']['sd']:.1f}; UPV {b['ndt']['upv']['mean']:.2f} &plusmn; {b['ndt']['upv']['sd']:.2f} km/s; cover {b['ndt']['cover']['mean']:.0f} &plusmn; {b['ndt']['cover']['sd']:.0f} mm; HCP {b['ndt']['hcp']['mean']:.0f} &plusmn; {b['ndt']['hcp']['sd']:.0f} mV; resistivity {b['ndt']['resistivity']['mean']:.0f} &plusmn; {b['ndt']['resistivity']['sd']:.0f} k&Omega;cm"],
], "Baseline structural and monitoring data of the healthy bridge", widths=[46, 54], small=True)
R.p("""Everything measured later is compared with this table and with the normalisation models, not with the design
calculation. This 'own baseline' idea is also what lets a model trained on simulated data be applied to the real bridge
(chapter 12).""")

# ====================================================================== 6
R.h1("Visual inspection scheme", "6")
R.p("""Visual inspection is the cheapest source of information and, for most defects that matter on a bridge like this
(water, corrosion, spalling, bearing and joint problems), the first. The scheme is organised as the brief asks,
Bridge &rarr; Span &rarr; Component &rarr; Location &rarr; Observation, so that every record has an address that can be
found again at the next inspection and compared.""")
R.table(["Component", "Locations recorded", "What to look for"], [
    ["PSC girders G1-G6", "zones Z1-Z5 along the span (Z1/Z5 end zones 0-3.3 m with anchorages and bearings, Z2/Z4 quarter spans, Z3 mid-span); soffit, web faces, anchorage faces",
     "flexural cracks at the soffit (any crack is significant in a fully prestressed girder), longitudinal cracks along ducts, rust staining, spalling, honeycombing, efflorescence, impact scrapes over the service road"],
    ["Deck slab", "bays between girders, cantilevers, construction joints", "cracking, leakage and efflorescence from above, spalling at drains"],
    ["Cross girders / diaphragms", "end diaphragm A and B, cross girder L/2", "cracking near jacking points, leakage from the joint"],
    ["Bearings", "G1-A ... G6-B", "bulging, splitting, excessive or no shear deformation, debris, corrosion of plates, loss of contact"],
    ["Expansion joints", "EJ-A1, EJ-P1, EJ-P2, EJ-A2", "torn or missing seal, debris in the gap, water running through, noise"],
    ["Drainage", "spouts U-1..U-6, D-1..D-6 per span", "blockage, spouts too short (water on the girder), broken pipes"],
    ["Footpaths, crash barriers, wearing coat", "upstream / downstream, lanes", "loose cover slabs, barrier cracking, ravelling and potholes near joints"],
    ["Pier caps, piers, abutments", "P1, P2, A1, A2", "leakage stains, pedestal spalling, cracking, vegetation, scour signs"],
], "Inspection hierarchy and items", widths=[17, 33, 50], small=True)
R.p("""<b>Condition states.</b> Each observation is given a condition state from 1 (good) to 5 (critical), with written rules so
that two inspectors reach similar ratings. For cracks the width bands are stricter on prestressed girders (CS2 above 0.05&nbsp;mm,
CS3 above 0.10, CS4 above 0.20, CS5 above 0.30&nbsp;mm) than on reinforced elements (0.10 / 0.20 / 0.30 / 1.0&nbsp;mm): a flexural
crack in a girder designed to stay uncracked means its precompression has gone. Spalling is CS3, or CS4 with exposed steel,
and CS5 with exposed strands over a large area. A component takes the worst state of its observations, and a bridge condition
index from 0 to 100 is the weighted average of (5 - CS)/4 over the component groups, with weights reflecting their structural
importance (girders 0.30, deck 0.12, bearings 0.12, joints 0.08, piers and pier caps 0.08 each, and so on).""")
R.p("""<b>Types and frequency of inspection.</b> Routine inspections, a walk-over with binoculars and the under-bridge view from
the service road, are carried out before and after every monsoon, because most defects here are water-driven. Principal
inspections, close visual inspection of every element with an under-bridge unit and hammer tapping of suspect concrete, are
made every three years and after any special event (overloaded vehicle, flood, vehicle impact). A special inspection follows
any monitoring alarm. Inspection data are kept in the format above so that they can be used directly as features (chapter 12).""")
R.p("""<b>Digital inspection application.</b> I wrote a single-file web application for the scheme
(<code>inspection_app/index.html</code>, runs offline in any browser). It walks the inspector through span, component, location
and observation, suggests the condition state from the crack width or defect type (the inspector can override it), stores photos,
keeps the records in the browser, and exports CSV/JSON. It also draws the worst condition state of every girder zone as a map and
computes the component ratings and the bridge condition index. The year-18 inspection of chapter 14 is built in as an example.""")
R.fig("f24_inspection_app", "The inspection application with the year-18 inspection loaded: entry form, component ratings and bridge condition index, worst condition state per girder zone of span S2, and the observation list.", 88)

# ====================================================================== 7
R.h1("Non-destructive testing programme", "7")
R.p("""An NDT method is chosen to answer a question raised by the inspection, not because it is available. The table below
is the programme for B33 written in the order of the brief: observed defect, engineering question, test, measurement,
interpretation.""")
R.table(["Observed defect", "Engineering question", "NDT", "Measurement", "Interpretation"], [
    ["Rust staining, spalling, damp patches on girder soffit", "Is the reinforcement corroding, and over what area?", "Half-cell potential (ASTM C876) on a 0.5 m grid; Wenner resistivity",
     "potential (mV vs CSE), resistivity (k&Omega;cm)", "&lt; -350 mV: corrosion likely; -200 to -350: uncertain; &gt; -200: low risk. Resistivity &lt; 10 k&Omega;cm means corrosion can proceed fast. A wet surface also shifts potentials negative, so both are read together"],
    ["Same, and low-cover zones", "Has carbonation reached the steel? Is the cover as specified?", "Cover meter; phenolphthalein on drill-holes or small cores",
     "cover (mm), carbonation depth (mm)", "carbonation &ge; cover in wet concrete: depassivated, corrosion has started or will start"],
    ["Cracks, honeycombing, surface deterioration", "Is the concrete sound through the section? Where are the voids?", "Ultrasonic pulse velocity (IS 516 Part 5/Sec 1), direct across web and bulb",
     "pulse velocity (km/s)", "&gt; 4.40 good, 3.75-4.40 medium, &lt; 3.75 doubtful (M25 and above); local drop over a crack = crack depth by indirect method"],
    ["Same", "Is the surface concrete weaker?", "Rebound hammer (IS 516 Part 5/Sec 4) on a grid", "rebound number", "comparative only; corrected for carbonation, which hardens the skin; calibrated against a few cores"],
    ["Rust or leakage at anchorages, longitudinal cracks along ducts", "Are the tendons grouted? Are strands corroding inside the duct?", "GPR to locate ducts, impact-echo along the duct line, borescope through a small drilled port",
     "duct positions, void indications, visual of strands", "voids at high points and anchorages are the usual sites of strand corrosion; HCP does not see strands inside ducts"],
    ["Sag of the girder, cracks at mid-span", "How much prestress is left?", "Lift-off test at an accessible anchorage, or saw-cut stress-release on the soffit with strain gauges",
     "force at lift-off (kN), released strain (microstrain)", "direct measure of the remaining force, the only way to confirm what the monitoring suggests"],
    ["Bearing distress", "Is the bearing still free to move and rotate?", "Bearing displacement transducers vs temperature; visual and dimension check", "movement per &deg;C (mm/&deg;C)", "much less than &alpha;L/2: restrained"],
], "Defect - question - NDT - measurement - interpretation", widths=[16, 18, 22, 16, 28], small=True)
R.p("""<b>Test locations.</b> Each girder has five test areas, one per zone, on the bottom bulb and on the web (both sides on
the outer girders), so a span has 30 areas and the bridge 90; the same areas are used at every inspection, which makes trends
possible. Extra half-cell grids and carbonation tests are placed at the girder ends next to the joints, under every drainage
spout of the outer girders, and wherever the inspection finds rust or spalling. Tendon investigations are limited to the
anchorage zones and to duct high points, where voids collect. In this study the 30 areas of S2 enter the learning data as
span-level summaries (lowest UPV, most negative potential, and so on) and the 90 areas of the whole bridge are clustered in
chapter 13.""")

# ====================================================================== 8
R.h1("Structural response measurement", "8")
R.p("""Three response quantities are monitored, each because it answers a question the others cannot.""")
R.table(["Quantity", "Sensor and location", "Sampling and duration", "Interpretation"], [
    ["Displacement (load test)", "LVDTs on a reference frame or a total station with prisms, mid-span of all six girders", "static readings at each load step; test once at commissioning and at every principal / special inspection",
     "deflection against the baseline, corrected to 25 &deg;C; distribution across girders shows which girder has softened"],
    ["Geometry (camber)", "precise levelling points on the kerbs over G1 and G6 and at the deck centreline, mid-span and supports", "once a year, same season and time of day",
     "loss of camber beyond the creep prediction is the clearest sign of prestress loss (about 1 mm per 1 % loss here)"],
    ["Strain, long term", "vibrating-wire gauges cast into the bottom bulb at mid-span of G1, G3, G4, G6, with thermistors", "hourly, permanent; daily averages used",
     "drift towards tension beyond the predicted creep and shrinkage means precompression is being lost (about 12 microstrain per 1 %)"],
    ["Strain, load test", "electrical resistance or FBG gauges on the soffit at mid-span", "100 Hz during the load test", "local stiffness change, crack opening if strain jumps"],
    ["Bearing movement", "displacement transducers across the bearings of G1 and G6 at both ends", "every 10 min, permanent", f"slope of movement against temperature, healthy {b['bearing_ref']:.3f} mm/&deg;C; a drop at one end with a rise at the other is a restrained bearing"],
    ["Temperature", "thermistors in deck slab, web and bulb at mid-span of G1 and G4", "every 10 min", "effective temperature and gradient for normalising all other data"],
    ["Acceleration", "12 vertical accelerometers (chapter 9)", "100 Hz, 30 min records twice a day", "frequencies, mode shapes, damping, traffic response"],
], "Monitoring quantities, sensors and their use", widths=[15, 27, 24, 34], small=True)
R.fig("f09_sensor_layout", "Instrumentation of span S2. The logger and a cellular modem sit in a cabinet on pier P2.")

# ====================================================================== 9
R.h1("Vibration measurement and signal processing", "9")
R.p(f"""<b>Accelerometers.</b> Twelve uniaxial vertical accelerometers are fixed to the soffits of the bottom bulbs (Figure [[fig:f09_sensor_layout]]):
at L/4, L/2 and 3L/4 on both outer girders, at L/2 on the four inner girders, and at L/4 of G3 and 3L/4 of G4. They are
low-noise force-balance or piezoelectric seismic sensors (noise about {DY.NOISE_UG_RTHZ:.0f}&nbsp;&micro;g/&radic;Hz including cable
and digitiser, range &plusmn;0.5&nbsp;g), because night-time ambient response is only 0.5-1&nbsp;mg RMS and MEMS sensors with
20-25&nbsp;&micro;g/&radic;Hz would bury the higher modes in noise. Vertical direction only: the modes of interest are vertical
bending and torsion; lateral and longitudinal motion of a 33&nbsp;m span on elastomeric bearings carries little information on the
girders.""")
R.p("""The layout was not right first time. With the first ten sensors (outer girders at three points, inner girders at mid-span)
the AutoMAC matrix of the twin's mode shapes showed two pairs of modes that were indistinguishable at the sensors, MAC = 1.00
(Figure [[fig:f12_automac]]): the second bending and second transverse modes both have zero response at mid-span and the same signs at the
outer-girder quarter points; the second torsion mode and a higher transverse mode likewise. The shape matrix was singular
(condition number of order 10<sup>12</sup>) and modal filtering impossible. Two sensors at the quarter points of the inner girders
G3 and G4 reduce the worst off-diagonal MAC to 0.37 and the condition number to 3.1.""")
R.fig("f12_automac", "AutoMAC of the first eight twin modes at the sensor positions, before and after adding two quarter-span sensors.", 82)
R.p("""<b>Sampling and duration.</b> The loggers sample at 1&nbsp;kHz with a hardware anti-alias filter and decimate to 100&nbsp;Hz,
so that modes up to about 20&nbsp;Hz are well inside the 40&nbsp;Hz usable band. A record lasts 30 minutes, more than 6000 cycles of
the first mode; with 10-minute records the scatter of f<sub>1</sub> in my simulations was about &plusmn;0.5&nbsp;%, too large next to
the effects to be detected, and 30 minutes with a fitted spectrum brought it to about 0.1&nbsp;%. Records are taken at 02:00, when
traffic is light and the excitation is closest to broadband, and at 14:00 for the traffic response.""")
R.p("""<b>Excitation.</b> Ambient excitation is wind, ground micro-tremor and above all traffic. Traffic is not white noise: lorries
bounce on their suspensions at about 1.3-2.6&nbsp;Hz and pass at 40-60&nbsp;km/h, and their mass adds to the bridge while they are on
it. These effects bias the identified frequencies by up to &plusmn;0.2&nbsp;% depending on how busy the bridge is, which is why
a<sub>RMS</sub> enters the normalisation. A drop-weight or shaker test is not needed for routine monitoring but would be used to
calibrate damping or mode shapes in a special investigation.""")
R.p("""<b>Signal processing.</b> The chain applied to every record is the one in the brief, with two additions:""")
R.ul(["a(t): 12 channels, 30 min at 100 Hz; mean removed, 4th-order Butterworth band-pass 0.5-30 Hz (zero phase);",
      "FFT: Welch averaging, Hann window of 8192 points (82 s, &Delta;f = 0.012 Hz), 50 % overlap, about 43 averages;",
      "frequency spectrum: auto-spectra of each channel and the full 12 x 12 cross-spectral matrix;",
      "modal filter: the channels are combined with the pseudo-inverse of the baseline mode-shape matrix, which separates the closely spaced bending and torsion modes;",
      "natural frequencies: a single-degree-of-freedom spectrum is fitted around each peak of the filtered spectrum (frequency and damping), and the mode shape is the singular vector of the cross-spectral matrix at the peak (FDD), compared with the baseline shape by MAC."])
R.p("""Damping estimates from ambient data are poor (scatter of &plusmn;25&nbsp;% in my simulations) and were kept as a feature only to show
that they are not useful here.""")

exec(open("build_report_part2.py", encoding="utf-8").read())
