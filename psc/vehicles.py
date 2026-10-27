"""
IRC:6-2017 live loads and the worst-position search on grillage influence surfaces.
"""
import itertools
import numpy as np
from . import inputs as P


class Vehicle:
    def __init__(self, name, axle_loads, spacings, wheel_dy, half_width, kerb_clear, impact):
        self.name = name
        self.loads = np.asarray(axle_loads, float)
        self.ax = np.concatenate([[0.0], np.cumsum(spacings)])     # behind the front axle
        self.wheel_dy = np.asarray(wheel_dy, float)                # lateral offsets of wheel lines
        self.half_width = half_width
        self.kerb_clear = kerb_clear
        self.impact = impact
        self.length = self.ax[-1]
        self.total = self.loads.sum()

    def points(self, x_front, yc, reverse=False):
        ax = self.length - self.ax if reverse else self.ax
        pts = []
        share = 1.0 / len(self.wheel_dy)
        for a, w in zip(ax, self.loads):
            for dy in self.wheel_dy:
                pts.append((x_front - a, yc + dy, w * share))
        return pts


def _tracked():
    n = 9
    seg = 4.57 / (n - 1)
    return Vehicle("70R tracked", [700.0 / n] * n, [seg] * (n - 1),
                   [-1.03, 1.03], 1.45, 1.2, P.IMPACT_TRACKED)


CLASS_A = Vehicle("Class A", [27, 27, 114, 114, 68, 68, 68, 68],
                  [1.1, 3.2, 1.2, 4.3, 3.0, 3.0, 3.0], [-0.9, 0.9], 1.15, 0.15, P.IMPACT)
R70_W = Vehicle("70R wheeled", [80, 120, 120, 170, 170, 170, 170],
                [3.96, 1.52, 2.13, 1.37, 3.05, 1.37], [-0.965, 0.965], 1.395, 1.2, P.IMPACT)
R70_T = _tracked()
# test lorry used for load tests / monitoring (3-axle tipper, 6x4, loaded to 300 kN)
TEST_TRUCK = Vehicle("Test truck 300 kN", [60, 120, 120], [3.8, 1.35], [-0.9, 0.9], 1.25, 0.6, 0.0)

GAP = 1.2        # clear distance between adjacent vehicles (carriageway > 7.5 m)

# IRC:6 Table 6A, carriageway 9.6 m - 13.1 m (three lanes): the worse of
# 70R in one lane + Class A in the remaining lane, or Class A in three lanes.
# Fewer loaded lanes are also tried, as they can govern for a given girder.
COMBOS = {
    "70RW + A": ([R70_W, CLASS_A], 1 - P.LANE_REDUCTION),
    "70RT + A": ([R70_T, CLASS_A], 1 - P.LANE_REDUCTION),
    "3 x A": ([CLASS_A] * 3, 1 - P.LANE_REDUCTION),
    "2 x A": ([CLASS_A] * 2, 1.0),
    "70RW": ([R70_W], 1.0),
    "70RT": ([R70_T], 1.0),
    "1 x A": ([CLASS_A], 1.0),
}


def bilinear(IS, xg, yg, x, y):
    """Interpolate influence surface IS[iy, ix] (defined at girder lines yg and
    nodes xg) at points x, y. Points outside the span give zero."""
    x = np.asarray(x); y = np.asarray(y)
    inside = (x >= 0) & (x <= xg[-1])
    fx = np.clip(x / (xg[1] - xg[0]), 0, len(xg) - 1 - 1e-9)
    ix = fx.astype(int); ax = fx - ix
    iy = np.clip(np.searchsorted(yg, y) - 1, 0, len(yg) - 2)
    ay = np.clip((y - yg[iy]) / (yg[iy + 1] - yg[iy]), 0, 1)
    v = (IS[iy, ix] * (1 - ax) * (1 - ay) + IS[iy, ix + 1] * ax * (1 - ay)
         + IS[iy + 1, ix] * (1 - ax) * ay + IS[iy + 1, ix + 1] * ax * ay)
    return np.where(inside, v, 0.0)


def vehicle_effect_table(IS, xg, yg, veh, y_centres, dx=0.25):
    """max over longitudinal position and direction of the effect of one vehicle,
    for each transverse centre position. Returns (best, x_front_at_best, reversed)."""
    xs = np.arange(0, xg[-1] + veh.length + dx, dx)
    best = np.full(len(y_centres), -np.inf)
    arg = np.zeros(len(y_centres)); rev = np.zeros(len(y_centres), bool)
    for r in (False, True):
        ax = veh.length - veh.ax if r else veh.ax
        X = xs[:, None, None] - ax[None, :, None]                 # (nx, naxle, 1)
        for j, yc in enumerate(y_centres):
            tot = np.zeros(len(xs))
            for dy in veh.wheel_dy:
                vals = bilinear(IS, xg, yg, X[..., 0], np.full(X[..., 0].shape, yc + dy))
                tot += (vals * veh.loads[None, :]).sum(axis=1) / len(veh.wheel_dy)
            k = int(np.argmax(tot))
            if tot[k] > best[j]:
                best[j], arg[j], rev[j] = tot[k], xs[k], r
    return best, arg, rev


def worst_combination(IS, xg, yg, sign=+1, dy=0.05, combos=COMBOS):
    """Worst (largest sign*effect) arrangement of vehicles across the carriageway,
    impact included. Returns dict name -> (value, layout)."""
    lo, hi = -P.KERB_Y, P.KERB_Y
    tables = {}
    results = {}
    for cname, (vehs, red) in combos.items():
        best_val, best_layout = -np.inf, None
        orders = {tuple(vehs[i].name for i in p): p for p in itertools.permutations(range(len(vehs)))}
        for order in orders.values():
            seq = [vehs[i] for i in order]
            # feasible centre ranges, dynamic programming over a common grid
            grid = np.arange(lo, hi + 1e-9, dy)
            dp = None; back = []
            for k, v in enumerate(seq):
                key = (v.name, sign)
                if key not in tables:
                    t, xa, rv = vehicle_effect_table(sign * IS, xg, yg, v, grid)
                    tables[key] = (t * (1 + v.impact), xa, rv)
                E = tables[key][0].copy()
                cmin = lo + v.kerb_clear + v.half_width
                cmax = hi - v.kerb_clear - v.half_width
                E[(grid < cmin - 1e-9) | (grid > cmax + 1e-9)] = -np.inf
                if dp is None:
                    dp = E
                    back.append(None)
                else:
                    prev = seq[k - 1]
                    need = prev.half_width + GAP + v.half_width
                    shift = int(np.ceil(need / dy - 1e-9))
                    cm = np.maximum.accumulate(dp)
                    am = np.zeros(len(dp), int)
                    cur = 0
                    for i in range(len(dp)):
                        if dp[i] >= dp[cur]:
                            cur = i
                        am[i] = cur
                    new = np.full(len(dp), -np.inf); bk = np.zeros(len(dp), int)
                    new[shift:] = cm[:-shift] + E[shift:]
                    bk[shift:] = am[:-shift]
                    dp = new
                    back.append(bk)
            i = int(np.argmax(dp))
            val = dp[i]
            if val > best_val:
                pos = [i]
                for bk in reversed(back[1:]):
                    pos.append(bk[pos[-1]])
                pos = pos[::-1]
                best_val = val
                best_layout = [(seq[k].name, round(grid[p], 2)) for k, p in enumerate(pos)]
        results[cname] = (sign * best_val * red, best_layout)
    return results
