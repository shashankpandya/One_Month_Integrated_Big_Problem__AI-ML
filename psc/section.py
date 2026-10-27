"""Section properties of the precast girder, end block and composite section."""
import numpy as np
from . import inputs as P


def polygon_props(pts):
    """Area, centroid height and second moments of a closed polygon (y, z)."""
    p = np.asarray(pts, float)
    y, z = p[:, 0], p[:, 1]
    y1, z1 = np.roll(y, -1), np.roll(z, -1)
    cr = y * z1 - y1 * z
    A = cr.sum() / 2
    zc = ((z + z1) * cr).sum() / (6 * A)
    yc = ((y + y1) * cr).sum() / (6 * A)
    Iyy = ((z * z + z * z1 + z1 * z1) * cr).sum() / 12 - A * zc ** 2    # about horizontal axis
    Izz = ((y * y + y * y1 + y1 * y1) * cr).sum() / 12 - A * yc ** 2    # about vertical axis
    per = np.hypot(y1 - y, z1 - z).sum()
    return dict(A=A, zc=zc, I=Iyy, Iz=Izz, perimeter=per)


def girder_outline(end_block=False):
    D = P.GIRDER_DEPTH
    btf, ttf, ttt = P.TOP_FLANGE_W, P.TOP_FLANGE_T, P.TOP_TAPER_T
    bbf, tbf, tbt = P.BOT_FLANGE_W, P.BOT_FLANGE_T, P.BOT_TAPER_T
    bw = bbf if end_block else P.WEB_T
    if end_block:
        # web as wide as the bottom bulb, only the top taper remains
        half = [(bbf / 2, 0), (bbf / 2, D - ttf - ttt), (btf / 2, D - ttf), (btf / 2, D)]
    else:
        half = [(bbf / 2, 0), (bbf / 2, tbf), (bw / 2, tbf + tbt),
                (bw / 2, D - ttf - ttt), (btf / 2, D - ttf), (btf / 2, D)]
    right = half
    left = [(-y, z) for (y, z) in reversed(half)]
    return [(-bbf / 2, 0)] + right + left[:-1]


def _rect_J(b, t):
    b, t = max(b, t), min(b, t)
    return (1 / 3 - 0.21 * t / b * (1 - t ** 4 / (12 * b ** 4))) * b * t ** 3


def precast(end_block=False):
    s = polygon_props(girder_outline(end_block))
    D = P.GIRDER_DEPTH
    s["yb"] = s["zc"]
    s["yt"] = D - s["zc"]
    s["Zb"] = s["I"] / s["yb"]
    s["Zt"] = s["I"] / s["yt"]
    # St Venant torsion, flanges + web as rectangles (tapers shared out)
    web_h = D - P.TOP_FLANGE_T - P.BOT_FLANGE_T
    bw = P.BOT_FLANGE_W if end_block else P.WEB_T
    s["J"] = (_rect_J(P.TOP_FLANGE_W, P.TOP_FLANGE_T + 0.5 * P.TOP_TAPER_T)
              + _rect_J(web_h, bw)
              + _rect_J(P.BOT_FLANGE_W, P.BOT_FLANGE_T + 0.5 * P.BOT_TAPER_T))
    return s


def composite(b_eff, end_block=False, m=None, E_slab=None, E_girder=None):
    """Girder + deck slab of width b_eff transformed to girder concrete."""
    if m is None:
        m = P.Ecm(P.FCK_DECK) / P.Ecm(P.FCK_GIRDER)
    g = precast(end_block)
    D, t = P.GIRDER_DEPTH, P.SLAB_T
    As = m * b_eff * t
    zs = D + t / 2
    A = g["A"] + As
    zc = (g["A"] * g["zc"] + As * zs) / A
    I = g["I"] + g["A"] * (g["zc"] - zc) ** 2 + m * b_eff * t ** 3 / 12 + As * (zs - zc) ** 2
    out = dict(A=A, zc=zc, I=I, m=m, b_eff=b_eff)
    out["yb"] = zc                          # to girder soffit
    out["yt_girder"] = D - zc               # to top of precast girder
    out["yt_slab"] = D + t - zc             # to top of slab
    out["Zb"] = I / zc
    out["Zt_girder"] = I / (D - zc)
    out["Zt_slab"] = I / (D + t - zc) / m   # stress in slab concrete = M / Zt_slab
    # torsion: girder rectangles + half of slab (Hambly)
    out["J"] = g["J"] + b_eff * t ** 3 / 6
    out["Iz"] = g["Iz"] + m * t * b_eff ** 3 / 12
    return out


def effective_width(which="inner"):
    """IRC:112 cl. 7.6.1.2 - full width is effective here (b_i << 0.1 l0)."""
    l0 = P.L
    if which == "inner":
        bi = (P.GIRDER_SPACING - P.WEB_T) / 2
        beff = 2 * min(0.2 * bi + 0.1 * l0, 0.2 * l0, bi) + P.WEB_T
    else:
        bi_in = (P.GIRDER_SPACING - P.WEB_T) / 2
        bi_out = P.DECK_WIDTH / 2 - P.GIRDER_Y[-1] - P.WEB_T / 2
        beff = (min(0.2 * bi_in + 0.1 * l0, 0.2 * l0, bi_in)
                + min(0.2 * bi_out + 0.1 * l0, 0.2 * l0, bi_out) + P.WEB_T)
    return beff


def web_width_at(x):
    """Web thickness at distance x from the girder end (end block + transition)."""
    a, b = P.END_BLOCK_LEN, P.END_BLOCK_LEN + P.END_BLOCK_TRANS
    xx = np.minimum(x, P.L_GIRDER - x)
    w = np.where(xx <= a, P.BOT_FLANGE_W,
                 np.where(xx >= b, P.WEB_T,
                          P.BOT_FLANGE_W + (P.WEB_T - P.BOT_FLANGE_W) * (xx - a) / (b - a)))
    return w


if __name__ == "__main__":
    for eb in (False, True):
        s = precast(eb)
        print("end block" if eb else "precast  ", {k: round(v, 4) for k, v in s.items()})
    for w in ("inner", "outer"):
        b = effective_width(w)
        c = composite(b)
        print(w, round(b, 3), {k: round(v, 4) for k, v in c.items()})
