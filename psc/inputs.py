"""
Adopted data for the bridge. Everything that is a decision (not a result)
lives here so that the design can be re-run with other choices.

Units: m, kN, kPa (=kN/m2), unless the name says otherwise (_mm, _MPa ...).
"""
import numpy as np

BRIDGE_ID = "B33"
N_SPANS = 3                    # 3 x 33 m simply supported spans, studied span = S2

# ---------------------------------------------------------------- geometry
L = 33.0                       # effective span (c/c bearings)
END_OVERHANG = 0.45            # girder end beyond bearing centreline
L_GIRDER = L + 2 * END_OVERHANG
JOINT_GAP = 0.040              # expansion gap at each pier / abutment

# transverse layout, measured from deck centreline (y = 0)
CARRIAGEWAY = 11.0             # 3 lanes x 3.5 m + 2 x 0.25 m marginal strips
CRASH_BARRIER_W = 0.50
FOOTPATH_W = 1.75              # clear footpath (utility ducts below)
RAILING_W = 0.25
DECK_WIDTH = CARRIAGEWAY + 2 * (CRASH_BARRIER_W + FOOTPATH_W + RAILING_W)   # 16.0

N_GIRDERS = 6
GIRDER_SPACING = 2.70
GIRDER_Y = (np.arange(N_GIRDERS) - (N_GIRDERS - 1) / 2) * GIRDER_SPACING
KERB_Y = CARRIAGEWAY / 2       # face of crash barrier

# precast I-girder (mm) - see section.py for the polygon
GIRDER_DEPTH = 1.80
TOP_FLANGE_W = 1.00
TOP_FLANGE_T = 0.175
TOP_TAPER_T = 0.100
WEB_T = 0.300
BOT_TAPER_T = 0.200
BOT_FLANGE_W = 0.750
BOT_FLANGE_T = 0.250
END_BLOCK_LEN = 1.50           # full-width web from girder end
END_BLOCK_TRANS = 1.50         # linear transition to the normal web

SLAB_T = 0.225                 # cast in-situ deck slab
WEARING_T = 0.075              # bituminous wearing coat

# cross girders (below the slab)
CROSS_X = [0.0, L / 2, L]      # end diaphragms + one at mid-span (justified in report)
END_DIAPH_B, END_DIAPH_D = 0.40, 1.60
CROSS_B, CROSS_D = 0.30, 1.45

# ---------------------------------------------------------------- materials
GAMMA_RC = 25.0                # kN/m3, IRC:6 Table 1
GAMMA_WC = 22.0                # bituminous
FCK_GIRDER = 45.0              # MPa (cube), M45
FCK_DECK = 40.0                # MPa, M40
FCK_TRANSFER = 36.0            # 0.8 fck at stressing (about 10 days)
T_STRESS = 10                  # days, stressing of all cables
T_DECK = 60                    # days, deck cast
T_SIDL = 90                    # days, SIDL in place / open to traffic
RH = 70.0                      # % ambient relative humidity
ALPHA_T = 12e-6                # /degC, IRC:6 cl. 215

# IRC:112 Table 6.5 (fctm, Ecm in MPa / GPa)
CONCRETE_TABLE = {35: (2.8, 32.0), 40: (3.0, 33.0), 45: (3.3, 34.0), 50: (3.5, 35.0)}

def fctm(fck):
    return CONCRETE_TABLE[int(fck)][0]

def Ecm(fck):
    return CONCRETE_TABLE[int(fck)][1] * 1e3   # MPa

def Ecm_t(fck, t):
    # IRC:112 A2 type time development of modulus (beta_cc^0.3), s = 0.25
    bcc = np.exp(0.25 * (1 - np.sqrt(28.0 / t)))
    return Ecm(fck) * bcc ** 0.3

E_DYN_FACTOR = 1.20            # dynamic / static modulus for modal analysis

# prestressing steel: 15.2 mm 7-ply low relaxation strand (IS 14268, Class 2)
STRAND_AREA = 140.0            # mm2
FPK = 1860.0                   # MPa
FP01K = 1600.0                 # MPa (0.1 % proof)
EP = 195e3                     # MPa
RELAX_1000H = 2.5              # % at 0.7 fpk (low relaxation)
JACK_RATIO = 0.75              # jacking stress / fpk
MU_FRICTION = 0.17             # corrugated HDPE sheathing
K_WOBBLE = 0.0020              # per m
ANCHOR_SLIP = 0.006            # m

STRANDS_PER_CABLE = 12         # 12T15 cables
DUCT_OD = 0.092
# cable layout: (z at anchorage from soffit, z at mid-span from soffit, lateral y at mid-span)
CABLES = [
    (0.350, 0.130, -0.180),
    (0.750, 0.130, +0.180),
    (1.150, 0.300, -0.180),
    (1.550, 0.300, +0.180),
]
DUCT_ECC_MID = 0.012           # strand bundle sits low in a draped duct

# untensioned steel
FY = 500.0                     # Fe500D
COVER = 0.050                  # clear cover to links (severe exposure)
COVER_DUCT = 0.075

# ---------------------------------------------------------------- loads
BARRIER_LOAD = 7.3             # kN/m each side (New Jersey type, ~0.29 m2)
FOOTPATH_LOAD = 6.0            # kN/m each side (kerb + cover slabs over ducts)
RAILING_LOAD = 1.5             # kN/m each side
WC_LOAD = WEARING_T * GAMMA_WC # kN/m2 over carriageway

# IRC:6 cl. 206.1 pedestrian load for L > 30 m
def footpath_ll(L=L, W=FOOTPATH_W, P=400.0):
    kgm2 = P - 260.0 + (4800.0 / L) * (16.5 - W) / 15.0
    return kgm2 * 9.81 / 1000.0        # kN/m2

IMPACT = 4.5 / (6.0 + L)       # Class A and 70R wheeled, span > 12 m
IMPACT_TRACKED = 0.10
LANE_REDUCTION = 0.10          # three lanes, IRC:6 cl. 205

# partial / combination factors (IRC:6 Annex B)
ULS = dict(g=1.35, sidl=1.35, wc=1.75, q=1.50)
SLS = {
    "rare":     dict(q=1.00, temp=0.60),
    "frequent": dict(q=0.75, temp=0.50),
    "quasi":    dict(q=0.00, temp=0.50),
}
GAMMA_C, GAMMA_S = 1.5, 1.15

# temperature difference, IRC:6 cl. 215.3 (deg C, depth limits in m)
TEMP_POS = dict(T1=17.8, T2=4.0, T3=2.1)
TEMP_REV = dict(T1=10.6, T2=0.7, T3=0.8, T4=6.6)
DIFF_SHRINKAGE = 100e-6        # residual deck-girder differential shrinkage
DIFF_SHR_RELIEF = 0.43         # creep relief factor

# ---------------------------------------------------------------- bearings & substructure
BRG_A, BRG_B = 0.40, 0.50      # plan size (m), a along span
BRG_LAYERS, BRG_TI = 5, 0.012  # internal elastomer layers
BRG_TE = BRG_LAYERS * BRG_TI + 2 * 0.006      # total elastomer incl. covers
BRG_G = 1.0e3                  # kPa shear modulus at 23 degC
BRG_KV = 1.6e6                 # kN/m vertical stiffness per bearing
K_SUB_PIER = 9.0e4             # kN/m longitudinal, per girder line (wall pier 1.2 m)
K_SUB_ABUT = 5.0e5

# ---------------------------------------------------------------- grillage mesh
NX = 24                        # longitudinal segments (1.375 m)
