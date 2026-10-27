"""
Maintenance decision support: integrates visual inspection, NDT, structural
response, vibration monitoring and the AI/ML output into actions.

The ML output is treated as one line of evidence. An action is triggered by
the evidence itself (crack, potential map, bearing movement, ...) and the ML
probability mainly sets the priority and what to check next.
"""

ACTIONS = {
    "routine": ("Routine inspection and monitoring", "continue", "pre- and post-monsoon walk-through, monthly data review"),
    "drain": ("Clean and extend drainage spouts", "within 1 month", "spouts discharge clear of girder soffit, add drip groove"),
    "joint": ("Replace strip-seal expansion joint", "within 3 months", "stops water and silt reaching bearings, anchorages and pier cap"),
    "bearing": ("Jack deck and replace / reset bearings", "within 6 months", "jacking points on end diaphragm (designed for this), lift <= 10 mm"),
    "patch": ("Concrete repair: remove carbonated / delaminated cover, treat bars, polymer-modified mortar", "within 6 months",
              "after marking out by hammer tapping and half-cell map"),
    "coating": ("Anti-carbonation coating on repaired and low-cover zones", "after repair", "extends time to re-initiation"),
    "inhibitor": ("Migrating corrosion inhibitor on wet, low-resistivity zones", "with repair", "where HCP is in the uncertain band"),
    "pt_investigation": ("Tendon investigation: GPR to locate ducts, impact-echo for voids, borescope through drilled ports at anchorages",
                         "within 3 months", "only direct way to confirm strand corrosion inside the duct"),
    "liftoff": ("Prestress check: lift-off or saw-cut (stress-release) test on the suspect girder", "within 6 months",
                "measures the remaining force rather than inferring it"),
    "assessment": ("Detailed structural assessment with measured losses (SLS decompression, ULS, shear at the end zone)",
                   "within 3 months", "decides whether strengthening is needed"),
    "ept": ("Provision for external post-tensioning (deviators at cross girders, anchors at end diaphragms)", "if assessment requires",
            "restores lost precompression; design for the measured loss + margin"),
    "crack_inject": ("Epoxy injection of cracks wider than 0.2 mm, sealing of finer cracks", "within 6 months", "after the cause is removed"),
    "instrument": ("Additional instrumentation: crack gauges, extra VW gauges at the end zone, bearing LVDTs on all girders",
                   "within 1 month", "to watch the suspect area until the repair"),
    "load_limit": ("Interim load restriction / lane management for heavy vehicles", "immediately", "until the assessment is complete"),
    "special": ("Special inspection by a bridge engineer (under-bridge unit, close visual of all girders)", "within 2 weeks", ""),
    "reinspect": ("Re-inspection and repeat monitoring campaign after the works", "6 months after works", "new baseline for the repaired bridge"),
}


def decide(ev):
    """ev: dict with keys
      vi_groups        worst CS per component group (dict)
      crack_w          max girder crack width (mm)
      hcp_min, rho_min, carb_margin  NDT extremes on girders
      bearing_ratio    min bearing movement / healthy
      novelty          True / False (fused Mahalanobis alarm)
      p                ML probabilities per mechanism
      pl_est, pl_sd    prestress-loss estimate (%) and its spread
      sls_margin       soffit stress margin (MPa) under the rare combination with the estimated loss
      uls_ratio        M_Ed / M_Rd with the estimated losses
      trend_alarms     list of monitored quantities whose EWMA chart has alarmed
    Returns (category, actions [(key, reason)], intervals)"""
    acts = []
    why = lambda k, r: acts.append((k, r))
    g = ev["vi_groups"]
    worst = max(g.values()) if g else 1

    if g.get("DR", 1) >= 3:
        why("drain", "blocked / short drainage spouts wetting the outer girder (VI)")
    if g.get("EJ", 1) >= 3:
        why("joint", "expansion joint seal leaking (VI)")
    if ev["bearing_ratio"] < 0.5 or g.get("BRG", 1) >= 4:
        why("bearing", "practically no bearing movement at one end + bearing distress (VI)" if ev["bearing_ratio"] < 0.1
            else f"bearing movement {ev['bearing_ratio']:.2f} of healthy at one end + bearing distress (VI)")
    corrosion_ndt = ev["hcp_min"] < -350 and ev["rho_min"] < 15
    if corrosion_ndt or g.get("G", 1) >= 4:
        why("patch", f"HCP {ev['hcp_min']:.0f} mV with resistivity {ev['rho_min']:.0f} kOhm cm, spalling/rust (NDT + VI)")
        why("coating", "carbonation front at or beyond cover in repaired zones" if ev["carb_margin"] > 0 else "low cover zones")
        why("inhibitor", "wet zones with potentials in the uncertain band")
    if ev["p"].get("CO", 0) > 0.5 or (corrosion_ndt and ev.get("near_anchorage", False)):
        why("pt_investigation", f"corrosion near tendon anchorages / ducts, ML p(CO) = {ev['p'].get('CO', 0):.2f}")
    girder_trend = [a for a in ev["trend_alarms"] if a.startswith(("VW", "level"))]
    if ev["p"].get("PL", 0) > 0.5 or ev["pl_est"] + 1.96 * ev["pl_sd"] > 5 or girder_trend:
        why("liftoff", f"estimated extra prestress loss {ev['pl_est']:.1f} +- {1.96*ev['pl_sd']:.1f} %, p(PL) = {ev['p'].get('PL', 0):.2f}"
            + (f"; trend alarms: {', '.join(girder_trend)}" if girder_trend else ""))
        why("assessment", f"rare-combination soffit margin {ev['sls_margin']:+.2f} MPa with the estimated loss")
    if ev["sls_margin"] < 0:
        why("ept", "decompression / cracking limit exceeded with the estimated loss: restore precompression")
    if ev["crack_w"] > 0.2:
        why("crack_inject", f"crack width {ev['crack_w']:.2f} mm")
    if ev["novelty"] or ev["trend_alarms"]:
        why("instrument", "monitoring alarm: " + ", ".join(ev["trend_alarms"]) if ev["trend_alarms"] else "novelty alarm on fused features")
    if ev["uls_ratio"] > 1.0:
        why("load_limit", f"M_Ed / M_Rd = {ev['uls_ratio']:.2f} > 1")
    if worst >= 4 or ev["novelty"]:
        why("special", "serious defect or monitoring alarm")
    if len(acts) == 0:
        why("routine", "no defect beyond CS2, no alarm")
    else:
        why("reinspect", "close the loop: confirm the effect of the works")

    worst_girder = g.get("G", 1)
    # an inspector's CS5 on a girder is reviewed against the structural check before the
    # bridge is called critical (load restriction etc.)
    if ev["uls_ratio"] > 1.0 or (worst_girder >= 5 and (ev["sls_margin"] < 0 or ev["uls_ratio"] > 0.9)):
        cat = "Critical"
    elif worst >= 4 or ev["sls_margin"] < 0.5 or ev["novelty"]:
        cat = "Serious - detailed investigation"
    elif worst >= 3:
        cat = "Poor - preventive maintenance"
    elif worst == 2:
        cat = "Fair - routine monitoring"
    else:
        cat = "Good - routine monitoring"
    intervals = {
        "Critical": dict(routine="weekly", principal="immediate special inspection", monitoring="continuous with alarms"),
        "Serious - detailed investigation": dict(routine="monthly until works", principal="after works, then 1 year",
                                                 monitoring="continuous, weekly review"),
        "Poor - preventive maintenance": dict(routine="pre/post monsoon", principal="every 2 years", monitoring="monthly review"),
        "Fair - routine monitoring": dict(routine="pre/post monsoon", principal="every 3 years", monitoring="quarterly review"),
        "Good - routine monitoring": dict(routine="pre/post monsoon", principal="every 3 years", monitoring="quarterly review"),
    }[cat]
    return cat, acts, intervals


DECISION_MATRIX = [
    # evidence pattern, interpretation, action
    ("All sources within baseline, CS <= 2", "healthy, normal ageing", "routine inspection and monitoring"),
    ("Frequency residual alarm only, nothing else", "environment / mass / traffic effect or early stiffness loss",
     "check temperature model and records, repeat in 1 month; do not act on frequency alone"),
    ("Frequency rise + bearing movement drop", "bearing restraint (seizure / debris)", "inspect bearings and joint, clean, replace"),
    ("Level sag + VW tension, no cracks", "prestress loss beyond the creep prediction", "lift-off / stress release test, assessment"),
    ("Level sag + load-test softening + flexural cracks", "decompression lost, girder cracked", "assessment, load management, external PT"),
    ("HCP < -350 mV + low resistivity + carbonation at cover", "active corrosion of links / bars", "patch repair, coating, inhibitor"),
    ("Corrosion signs near anchorages / joint ends", "possible strand corrosion in ducts", "GPR + impact-echo + borescope, assessment"),
    ("Low UPV + low rebound over a zone", "concrete deterioration / honeycombing", "cores, petrography, repair or encasement"),
    ("Local deflection / strain jump at one girder + spalling", "local stiffness loss (impact, delamination)", "special inspection, repair"),
]
