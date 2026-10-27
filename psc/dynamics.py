"""
Acceleration records by modal superposition (traffic + ambient + sensor noise)
and the identification chain  a(t) -> filtering -> FFT/Welch -> modal filter / FDD -> f_n.
"""
import numpy as np
from scipy import signal
from scipy.optimize import least_squares
from . import inputs as P

FS = 100.0              # Hz, sampling rate after the logger's anti-alias filter
DURATION = 1800.0       # s, one record (30 min), > 6000 cycles of f1
NFFT = 8192             # Welch segment, df = 0.012 Hz

# accelerometers: (name, girder index, grillage node index along span)
SENSORS = [
    ("A1", 0, 6), ("A2", 0, 12), ("A3", 0, 18),
    ("A4", 1, 12), ("A5", 2, 12), ("A6", 3, 12), ("A7", 4, 12),
    ("A8", 5, 6), ("A9", 5, 12), ("A10", 5, 18),
    # added after the AutoMAC check: without them the 2nd bending and 2nd
    # transverse modes look identical at the outer-girder quarter points
    ("A11", 2, 6), ("A12", 3, 18),
]
LANES = (-3.667, 0.0, 3.667)
NOISE_UG_RTHZ = 2.0     # sensor + cable + ADC noise density, micro-g / sqrt(Hz)


def sensor_shapes(w_grid):
    """w_grid (ny, nx+1, nmodes) -> (nsensors, nmodes)."""
    return np.array([w_grid[iy, ix] for _, iy, ix in SENSORS])


def _modal_filters(f, zeta, fs):
    filt = []
    for fi, zi in zip(f, zeta):
        w = 2 * np.pi * fi
        A = np.array([[0, 1], [-w * w, -2 * zi * w]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[-w * w, -2 * zi * w]])
        D = np.array([[1.0]])
        Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), 1 / fs, method="foh")
        b, a = signal.ss2tf(Ad, Bd, Cd, Dd)
        filt.append((b[0], a))
    return filt


def _shape_at(w_grid, x, y, xg, yg):
    """Bilinear interpolation of mode shapes at load points; returns (npts, nmodes)."""
    inside = (x >= 0) & (x <= xg[-1])
    fx = np.clip(x / (xg[1] - xg[0]), 0, len(xg) - 1 - 1e-9)
    ix = fx.astype(int); ax = (fx - ix)[:, None]
    iy = np.clip(np.searchsorted(yg, y) - 1, 0, len(yg) - 2)
    ay = np.clip((y - yg[iy]) / (yg[iy + 1] - yg[iy]), 0, 1)[:, None]
    v = (w_grid[iy, ix] * (1 - ax) * (1 - ay) + w_grid[iy, ix + 1] * ax * (1 - ay)
         + w_grid[iy + 1, ix] * (1 - ax) * ay + w_grid[iy + 1, ix + 1] * ax * ay)
    return v * inside[:, None]


def random_traffic(rng, duration, rate_per_10min, heavy_share):
    n = rng.poisson(rate_per_10min * duration / 600.0)
    veh = []
    for _ in range(n):
        t0 = rng.uniform(-10, duration)
        lane = LANES[rng.choice(3, p=[0.45, 0.35, 0.20])]     # slow lane busiest
        heavy = rng.random() < heavy_share
        if heavy:
            kind = rng.choice(3, p=[0.35, 0.40, 0.25])
            if kind == 0:     # 2-axle truck
                ax = [0, 4.2]; W = rng.uniform(80, 160); split = [0.35, 0.65]
            elif kind == 1:   # 3-axle
                ax = [0, 3.8, 5.15]; W = rng.uniform(140, 280); split = [0.22, 0.39, 0.39]
            else:             # multi-axle articulated
                ax = [0, 3.5, 4.8, 10.8, 12.1, 13.4]; W = rng.uniform(220, 450)
                split = [0.12, 0.17, 0.17, 0.18, 0.18, 0.18]
            v = rng.uniform(11, 17)                # m/s (40 - 60 km/h)
            amp = rng.uniform(0.02, 0.06); fv = rng.uniform(1.3, 2.6)     # body bounce of loaded lorries
        else:
            ax = [0, 2.6]; W = rng.uniform(10, 25); split = [0.55, 0.45]
            v = rng.uniform(15, 22)
            amp = rng.uniform(0.01, 0.03); fv = rng.uniform(1.2, 2.0)
        veh.append(dict(t0=t0, y=lane + rng.normal(0, 0.25), ax=np.array(ax),
                        P=W * np.array(split), v=v, amp=amp, fv=fv, ph=rng.uniform(0, 2 * np.pi)))
    return veh


def simulate(f, zeta, w_grid, xg, yg, rng, duration=DURATION, fs=FS,
             rate=250, heavy_share=0.25, ambient=0.03, noise=NOISE_UG_RTHZ, return_forces=False):
    """Vertical accelerations (m/s2) at the sensors, shape (nsensors, N)."""
    N = int(duration * fs)
    t = np.arange(N) / fs
    nm = len(f)
    Q = np.zeros((nm, N))
    for v in random_traffic(rng, duration, rate, heavy_share):
        L_tot = xg[-1] + v["ax"][-1]
        i0 = max(int(np.floor(v["t0"] * fs)), 0)
        i1 = min(int(np.ceil((v["t0"] + L_tot / v["v"]) * fs)), N)
        if i1 <= i0:
            continue
        tt = t[i0:i1]
        xf = v["v"] * (tt - v["t0"])
        dyn = 1 + v["amp"] * np.sin(2 * np.pi * v["fv"] * tt + v["ph"])
        for a, Pa in zip(v["ax"], v["P"]):
            xa = xf - a
            for dy in (-0.9, 0.9):
                sh = _shape_at(w_grid, xa, np.full_like(xa, v["y"] + dy), xg, yg)
                Q[:, i0:i1] -= (0.5 * Pa * dyn)[None, :] * sh.T       # downward load
    # ambient (wind, micro-tremor): broadband modal forces, kN
    Q += ambient * rng.standard_normal((nm, N))
    filt = _modal_filters(f, zeta, fs)
    qdd = np.array([signal.lfilter(b, a, Q[i]) for i, (b, a) in enumerate(filt)])
    S = sensor_shapes(w_grid)
    acc = S @ qdd
    sig_n = noise * 1e-6 * 9.81 * np.sqrt(fs / 2)
    acc += sig_n * rng.standard_normal(acc.shape)
    if return_forces:
        return acc, Q
    return acc


# ---------------------------------------------------------------- identification
def preprocess(acc, fs=FS, band=(0.5, 30.0)):
    sos = signal.butter(4, band, btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, acc - acc.mean(axis=1, keepdims=True), axis=1)


def segment_fft(acc, fs=FS, nfft=NFFT):
    step = nfft // 2
    nseg = (acc.shape[1] - nfft) // step + 1
    win = signal.windows.hann(nfft, sym=False)
    idx = np.arange(nfft)[None, :] + step * np.arange(nseg)[:, None]
    seg = acc[:, idx] * win                     # (nch, nseg, nfft)
    X = np.fft.rfft(seg, axis=2)                # (nch, nseg, nf)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    scale = 1.0 / (fs * (win ** 2).sum())
    return X, freqs, scale


def fdd(X, scale):
    """Singular values of the cross-spectral matrix at every frequency line."""
    G = np.einsum("isf,jsf->fij", X, np.conj(X)) * scale / X.shape[1]
    U, s, _ = np.linalg.svd(G)
    return s, U                                 # s (nf, nch), U (nf, nch, nch)


def mac(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def sdof_fit(fr, S, f0):
    """Fit A/((fn^2-f^2)^2 + (2 z fn f)^2) + B to a PSD window, in log space."""
    def model(p):
        fn, z, lA, lB = p
        return np.log(np.exp(lA) / ((fn ** 2 - fr ** 2) ** 2 + (2 * z * fn * fr) ** 2) + np.exp(lB))
    y = np.log(S + 1e-30)
    A0 = S.max() * (2 * 0.015 * f0 * f0) ** 2
    p0 = [f0, 0.015, np.log(A0), np.log(S.min() + 1e-30)]
    try:
        r = least_squares(lambda p: model(p) - y, p0,
                          bounds=([fr[0], 0.002, -np.inf, -np.inf], [fr[-1], 0.08, np.inf, np.inf]))
        return r.x[0], r.x[1]
    except Exception:
        return f0, np.nan


def identify(acc, Psi_all, f_ref, track, fs=FS, band=0.06):
    """Modal-filter peak picking for the tracked modes, FDD shape at the peak.
    Psi_all: (nch, nm) reference shapes used as spatial filter, f_ref their frequencies,
    track: indices of the modes to identify."""
    y = preprocess(acc, fs)
    X, freqs, scale = segment_fft(y, fs)
    Pinv = np.linalg.pinv(Psi_all)
    out = {}
    for k in track:
        Xm = np.einsum("c,csf->sf", Pinv[k], X)
        S = (np.abs(Xm) ** 2).mean(axis=0) * scale
        lo, hi = f_ref[k] * (1 - band), f_ref[k] * (1 + band)
        sel = np.where((freqs >= lo) & (freqs <= hi))[0]
        i = sel[np.argmax(S[sel])]
        fk, zk = sdof_fit(freqs[sel], S[sel], freqs[i])
        if not (lo < fk < hi):
            fk, zk = freqs[i], np.nan
        G = np.einsum("is,js->ij", X[:, :, i], np.conj(X[:, :, i]))
        u, s, _ = np.linalg.svd(G)
        # pick the singular vector closest to the reference shape (close modes)
        cand = [u[:, j] for j in range(2)]
        m = [mac(c, Psi_all[:, k]) for c in cand]
        j = int(np.argmax(m))
        shape = np.real(cand[j] * np.exp(-1j * np.angle(cand[j][np.argmax(np.abs(cand[j]))])))
        out[k] = dict(f=fk, zeta=zk, mac=m[j], shape=shape, psd_peak=S[i])
    rms = np.sqrt((y ** 2).mean(axis=1))
    mids = [i for i, (_, iy, ix) in enumerate(SENSORS) if ix == 12]
    return out, dict(a_rms=float(rms[mids].mean()), a_max=float(np.abs(y).max()))


def oma_fdd(acc, f_model, Psi_model, fs=FS, search=0.15, min_mac=0.5):
    """Output-only identification for the commissioning test: peaks of the first
    singular value, each assigned to the model mode its singular vector resembles
    most. The model only names the modes, it does not filter the data."""
    y = preprocess(acc, fs)
    X, freqs, scale = segment_fft(y, fs)
    s, U = fdd(X, scale)
    s1 = s[:, 0]
    pk, _ = signal.find_peaks(np.log(s1), prominence=0.5)
    out = {}
    for k in range(len(f_model)):
        cand = [i for i in pk if abs(freqs[i] / f_model[k] - 1) < search]
        if not cand:
            continue
        macs = [mac(U[i, :, 0], Psi_model[:, k]) for i in cand]
        j = int(np.argmax(macs))
        if macs[j] < min_mac:
            continue
        i = cand[j]
        w = np.where(np.abs(freqs / freqs[i] - 1) < 0.025)[0]
        fk, zk = sdof_fit(freqs[w], s1[w], freqs[i])
        u = U[i, :, 0]
        u = np.real(u * np.exp(-1j * np.angle(u[np.argmax(np.abs(u))])))
        out[k] = dict(f=fk, zeta=zk, shape=u / np.abs(u).max(), mac_model=macs[j])
    return out, (freqs, s)
