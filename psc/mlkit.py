"""
Small machine-learning toolkit written with numpy / scipy only
(scikit-learn could not be installed on the machine used for this work).
Each method is the textbook version, kept short enough to read and check.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------- preprocessing
class Scaler:
    def fit(self, X):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PCA:
    def fit(self, X, n=None):
        self.mu = X.mean(axis=0)
        U, s, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.var = s ** 2 / (len(X) - 1)
        self.ratio = self.var / self.var.sum()
        self.components = Vt[:n] if n else Vt
        return self

    def transform(self, X):
        return (X - self.mu) @ self.components.T


# ---------------------------------------------------------------- novelty detection
class Mahalanobis:
    """Squared Mahalanobis distance to the healthy cloud, covariance shrunk
    towards its diagonal (helps when features are many and correlated)."""
    def fit(self, X, shrink=0.05, q=0.99):
        self.mu = X.mean(axis=0)
        S = np.cov(X, rowvar=False)
        S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
        # features that never vary on the healthy bridge (no cracks, no spalling) get a
        # small floor so that any non-zero reading counts as strongly abnormal
        S = S + 1e-3 * np.eye(len(S))
        self.Si = np.linalg.inv(S)
        d = self.score(X)
        self.threshold = float(np.quantile(d, q))
        return self

    def score(self, X):
        D = X - self.mu
        return np.einsum("ij,jk,ik->i", D, self.Si, D)


def ewma(x, lam=0.2):
    z = np.zeros_like(x, float)
    z[0] = x[0]
    for i in range(1, len(x)):
        z[i] = lam * x[i] + (1 - lam) * z[i - 1]
    return z


# ---------------------------------------------------------------- linear models
def ridge(X, y, alpha=1e-6):
    A = np.column_stack([np.ones(len(X)), X])
    I = np.eye(A.shape[1]); I[0, 0] = 0
    return np.linalg.solve(A.T @ A + alpha * I, A.T @ y)


def ridge_predict(b, X):
    return b[0] + X @ b[1:]


class LogisticRegression:
    """Multinomial logistic regression with L2 penalty, L-BFGS."""
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.classes = np.unique(y)
        K, n, p = len(self.classes), *X.shape
        Y = (y[:, None] == self.classes[None, :]).astype(float)
        Xa = np.column_stack([X, np.ones(n)])
        def f(w):
            W = w.reshape(p + 1, K)
            Z = Xa @ W
            Z -= Z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(Z).sum(axis=1))
            ll = (Y * Z).sum() - lse.sum()
            pen = 0.5 / self.C * (W[:-1] ** 2).sum()
            Pm = np.exp(Z - lse[:, None])
            g = -(Xa.T @ (Y - Pm))
            g[:-1] += W[:-1] / self.C
            return -ll + pen, g.ravel()
        r = minimize(f, np.zeros((p + 1) * K), jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        self.W = r.x.reshape(p + 1, K)
        return self

    def predict_proba(self, X):
        Z = np.column_stack([X, np.ones(len(X))]) @ self.W
        Z -= Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------- trees and forests
class Tree:
    def __init__(self, task="clf", max_depth=14, min_leaf=2, max_features=None, rng=None, n_classes=None):
        self.task, self.max_depth, self.min_leaf = task, max_depth, min_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.K = n_classes

    def _impurity_split(self, xs, ys):
        n = len(xs)
        ml = self.min_leaf
        valid = np.zeros(n - 1, bool)
        valid[ml - 1:n - ml] = True
        valid &= xs[1:] > xs[:-1]
        if not valid.any():
            return None
        if self.task == "clf":
            oh = np.zeros((n, self.K)); oh[np.arange(n), ys] = 1
            cl = np.cumsum(oh, axis=0)[:-1]
            tot = cl[-1] + oh[-1]
            cr = tot - cl
            nl = np.arange(1, n)[:, None]; nr = n - nl
            gl = 1 - ((cl / nl) ** 2).sum(axis=1)
            gr = 1 - ((cr / nr) ** 2).sum(axis=1)
            cost = nl[:, 0] * gl + nr[:, 0] * gr
        else:
            s1 = np.cumsum(ys)[:-1]; s2 = np.cumsum(ys ** 2)[:-1]
            t1, t2 = ys.sum(), (ys ** 2).sum()
            nl = np.arange(1, n); nr = n - nl
            cost = (s2 - s1 ** 2 / nl) + ((t2 - s2) - (t1 - s1) ** 2 / nr)
        cost = np.where(valid, cost, np.inf)
        i = int(np.argmin(cost))
        if not np.isfinite(cost[i]):
            return None
        return cost[i], 0.5 * (xs[i] + xs[i + 1])

    def _leaf(self, y):
        if self.task == "clf":
            return np.bincount(y, minlength=self.K) / len(y)
        return np.array([y.mean()])

    def fit(self, X, y):
        n, p = X.shape
        mf = self.max_features or p
        self.feat, self.thr, self.left, self.right, self.val = [], [], [], [], []
        self.gain = np.zeros(p)
        def node_cost(yy):
            if self.task == "clf":
                pr = np.bincount(yy, minlength=self.K) / len(yy)
                return len(yy) * (1 - (pr ** 2).sum())
            return ((yy - yy.mean()) ** 2).sum()
        stack = [(np.arange(n), 0, -1, None)]
        while stack:
            idx, depth, parent, side = stack.pop()
            me = len(self.feat)
            self.feat.append(-1); self.thr.append(0.0); self.left.append(-1); self.right.append(-1)
            self.val.append(self._leaf(y[idx]))
            if parent >= 0:
                (self.left if side == "L" else self.right)[parent] = me
            yy = y[idx]
            pure = (self.task == "clf" and np.all(yy == yy[0])) or len(idx) < 2 * self.min_leaf
            if depth >= self.max_depth or pure:
                continue
            best = None
            c0 = node_cost(yy)
            for f in self.rng.choice(p, mf, replace=False):
                x = X[idx, f]
                o = np.argsort(x, kind="stable")
                r = self._impurity_split(x[o], yy[o])
                if r and (best is None or r[0] < best[0]):
                    best = (r[0], f, r[1])
            if best is None or best[0] >= c0 - 1e-12:
                continue
            cost, f, t = best
            self.feat[me], self.thr[me] = int(f), float(t)
            self.gain[f] += c0 - cost
            m = X[idx, f] <= t
            stack.append((idx[~m], depth + 1, me, "R"))
            stack.append((idx[m], depth + 1, me, "L"))
        self.feat = np.array(self.feat); self.thr = np.array(self.thr)
        self.left = np.array(self.left); self.right = np.array(self.right)
        self.val = np.array(self.val)
        return self

    def apply(self, X):
        node = np.zeros(len(X), int)
        active = self.feat[node] >= 0
        while active.any():
            i = np.where(active)[0]
            n = node[i]
            go_left = X[i, self.feat[n]] <= self.thr[n]
            node[i] = np.where(go_left, self.left[n], self.right[n])
            active = self.feat[node] >= 0
        return node

    def predict_value(self, X):
        return self.val[self.apply(X)]


class HistTree:
    """CART on pre-binned features: split search by histogram (bincount), O(n) per node."""
    def __init__(self, task, max_depth, min_leaf, mf, rng, K, nb):
        self.task, self.max_depth, self.min_leaf, self.mf, self.rng, self.K, self.nb =             task, max_depth, min_leaf, mf, rng, K, nb

    def fit(self, B, y, edges):
        n, p = B.shape
        K, nb = self.K, self.nb
        feat, thr, left, right, val = [], [], [], [], []
        self.gain = np.zeros(p)
        stack = [(np.arange(n), 0, -1, 0)]
        while stack:
            idx, depth, parent, side = stack.pop()
            me = len(feat)
            yy = y[idx]
            if self.task == "clf":
                cnt = np.bincount(yy, minlength=K).astype(float)
                v = cnt / cnt.sum(); c0 = len(yy) * (1 - (v ** 2).sum())
            else:
                v = np.array([yy.mean()]); c0 = ((yy - yy.mean()) ** 2).sum()
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(v)
            if parent >= 0:
                (left if side == 0 else right)[parent] = me
            if depth >= self.max_depth or len(idx) < 2 * self.min_leaf or c0 <= 1e-12:
                continue
            best = None
            for f in self.rng.choice(p, self.mf, replace=False):
                b = B[idx, f]
                if self.task == "clf":
                    H = np.bincount(b * K + yy, minlength=nb * K).reshape(nb, K).astype(float)
                    cl = np.cumsum(H, axis=0)[:-1]; tot = H.sum(axis=0)
                    cr = tot - cl
                    nl = cl.sum(1); nr = cr.sum(1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cost = nl * (1 - ((cl / nl[:, None]) ** 2).sum(1)) + nr * (1 - ((cr / nr[:, None]) ** 2).sum(1))
                else:
                    c = np.bincount(b, minlength=nb).astype(float)
                    s1 = np.bincount(b, weights=yy, minlength=nb); s2 = np.bincount(b, weights=yy * yy, minlength=nb)
                    nl = np.cumsum(c)[:-1]; nr = len(yy) - nl
                    l1 = np.cumsum(s1)[:-1]; l2 = np.cumsum(s2)[:-1]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cost = (l2 - l1 ** 2 / nl) + ((s2.sum() - l2) - (s1.sum() - l1) ** 2 / nr)
                ok = (nl >= self.min_leaf) & (nr >= self.min_leaf)
                if not ok.any():
                    continue
                cost = np.where(ok, cost, np.inf)
                j = int(np.argmin(cost))
                if best is None or cost[j] < best[0]:
                    best = (cost[j], f, j)
            if best is None or not np.isfinite(best[0]) or best[0] >= c0 - 1e-12:
                continue
            cost, f, j = best
            feat[me], thr[me] = int(f), float(edges[f][j]) if j < len(edges[f]) else np.inf
            self.gain[f] += c0 - cost
            m = B[idx, f] <= j
            stack.append((idx[~m], depth + 1, me, 1))
            stack.append((idx[m], depth + 1, me, 0))
        self.feat, self.thr = np.array(feat), np.array(thr)
        self.left, self.right, self.val = np.array(left), np.array(right), np.array(val)
        return self

    def predict_value(self, X):
        node = np.zeros(len(X), int)
        active = self.feat[node] >= 0
        while active.any():
            i = np.where(active)[0]
            nd = node[i]
            go_left = X[i, self.feat[nd]] < self.thr[nd]
            node[i] = np.where(go_left, self.left[nd], self.right[nd])
            active = self.feat[node] >= 0
        return self.val[node]


class RandomForest:
    def __init__(self, task="clf", n_trees=150, max_depth=14, min_leaf=2, max_features="sqrt",
                 seed=0, n_bins=48):
        self.task, self.n_trees, self.max_depth, self.min_leaf = task, n_trees, max_depth, min_leaf
        self.max_features, self.seed, self.n_bins = max_features, seed, n_bins

    def _bin(self, X):
        B = np.zeros(X.shape, np.int64)
        for f in range(X.shape[1]):
            B[:, f] = np.searchsorted(self.edges[f], X[:, f], side="right")
        return B

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        mf = max(1, int(np.sqrt(p))) if self.max_features == "sqrt" else (
            max(1, int(p / 3)) if self.max_features == "third" else p)
        self.edges = [np.unique(np.quantile(X[:, f], np.linspace(0, 1, self.n_bins + 1)[1:-1])) for f in range(p)]
        B = self._bin(X)
        nb = self.n_bins
        if self.task == "clf":
            self.classes = np.unique(y)
            yi = np.searchsorted(self.classes, y)
            K = len(self.classes)
        else:
            yi, K = y.astype(float), None
        self.trees = []
        self.oob = np.zeros((n, K if K else 1)); self.oob_n = np.zeros(n)
        for t in range(self.n_trees):
            b = rng.integers(0, n, n)
            tr = HistTree(self.task, self.max_depth, self.min_leaf, mf,
                          np.random.default_rng(rng.integers(1 << 31)), K, nb)
            tr.fit(B[b], yi[b], self.edges)
            self.trees.append(tr)
            out = np.setdiff1d(np.arange(n), b)
            self.oob[out] += tr.predict_value(X[out]); self.oob_n[out] += 1
        imp = np.sum([t.gain for t in self.trees], axis=0)
        self.importance = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_value(X) for t in self.trees], axis=0)

    def predict(self, X):
        P_ = self.predict_proba(X)
        if self.task == "clf":
            return self.classes[np.argmax(P_, axis=1)]
        return P_[:, 0]

    def predict_spread(self, X):
        """Regression: mean and spread over the trees (a rough uncertainty)."""
        V = np.array([t.predict_value(X)[:, 0] for t in self.trees])
        return V.mean(axis=0), V.std(axis=0)


# ---------------------------------------------------------------- clustering
def kmeans(X, k, n_init=10, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(X[rng.choice(len(X), p=d / d.sum())])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((X - c[lab]) ** 2).sum()
        if best is None or inertia < best[2]:
            best = (lab, c, inertia)
    return best


class GaussianMixture:
    def __init__(self, k, seed=0, reg=1e-4, iters=300):
        self.k, self.seed, self.reg, self.iters = k, seed, reg, iters

    def _logpdf(self, X):
        n, p = X.shape
        L = np.zeros((n, self.k))
        for j in range(self.k):
            S = self.cov[j]
            c, low = cho_factor(S)
            D = X - self.mu[j]
            sol = cho_solve((c, low), D.T).T
            logdet = 2 * np.log(np.diag(c)).sum()
            L[:, j] = -0.5 * ((D * sol).sum(1) + logdet + p * np.log(2 * np.pi)) + np.log(self.w[j])
        return L

    def fit(self, X):
        n, p = X.shape
        lab, c, _ = kmeans(X, self.k, seed=self.seed)
        self.mu = c
        self.cov = np.array([np.cov(X[lab == j], rowvar=False) + self.reg * np.eye(p) if (lab == j).sum() > p
                             else np.cov(X, rowvar=False) for j in range(self.k)])
        self.w = np.array([(lab == j).mean() for j in range(self.k)])
        prev = -np.inf
        for _ in range(self.iters):
            L = self._logpdf(X)
            m = L.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(L - m).sum(axis=1))
            R = np.exp(L - lse[:, None])
            Nk = R.sum(axis=0) + 1e-12
            self.w = Nk / n
            self.mu = (R.T @ X) / Nk[:, None]
            for j in range(self.k):
                D = X - self.mu[j]
                self.cov[j] = (R[:, j, None] * D).T @ D / Nk[j] + self.reg * np.eye(p)
            ll = lse.sum()
            if ll - prev < 1e-6 * abs(ll):
                break
            prev = ll
        self.loglik = ll
        self.bic = -2 * ll + (self.k - 1 + self.k * p + self.k * p * (p + 1) / 2) * np.log(n)
        return self

    def predict_proba(self, X):
        L = self._logpdf(X)
        L -= L.max(axis=1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------- Gaussian process regression
class GPR:
    """RBF kernel with one length scale per input + white noise; hyper-parameters
    by maximising the log marginal likelihood."""
    def fit(self, X, y, restarts=3, seed=0):
        self.X = X; self.ym = y.mean(); self.ys = y.std() or 1.0
        yn = (y - self.ym) / self.ys
        n, p = X.shape
        def nll(th):
            sf, ell, sn = np.exp(th[0]), np.exp(th[1:1 + p]), np.exp(th[-1])
            K = self._k(X, X, sf, ell) + (sn ** 2 + 1e-8) * np.eye(n)
            try:
                c = cho_factor(K)
            except np.linalg.LinAlgError:
                return 1e10
            a = cho_solve(c, yn)
            return 0.5 * yn @ a + np.log(np.diag(c[0])).sum() + 0.5 * n * np.log(2 * np.pi)
        rng = np.random.default_rng(seed)
        best = None
        for r in range(restarts):
            th0 = np.concatenate([[0.0], np.log(rng.uniform(0.5, 3.0, p)), [np.log(0.2)]])
            res = minimize(nll, th0, method="L-BFGS-B",
                           bounds=[(-3, 3)] + [(-3, 4)] * p + [(-6, 1)])
            if best is None or res.fun < best.fun:
                best = res
        th = best.x
        self.sf, self.ell, self.sn = np.exp(th[0]), np.exp(th[1:1 + p]), np.exp(th[-1])
        K = self._k(X, X, self.sf, self.ell) + (self.sn ** 2 + 1e-8) * np.eye(n)
        self.c = cho_factor(K)
        self.alpha = cho_solve(self.c, yn)
        return self

    @staticmethod
    def _k(A, B, sf, ell):
        D = ((A[:, None, :] - B[None, :, :]) / ell) ** 2
        return sf ** 2 * np.exp(-0.5 * D.sum(-1))

    def predict(self, Xs, return_std=True):
        Ks = self._k(Xs, self.X, self.sf, self.ell)
        mu = Ks @ self.alpha
        if not return_std:
            return mu * self.ys + self.ym
        v = cho_solve(self.c, Ks.T)
        var = self.sf ** 2 + self.sn ** 2 - (Ks * v.T).sum(1)
        return mu * self.ys + self.ym, np.sqrt(np.maximum(var, 0)) * self.ys


# ---------------------------------------------------------------- metrics and validation
def confusion(y, yp, labels):
    M = np.zeros((len(labels), len(labels)), int)
    for a, b in zip(y, yp):
        M[labels.index(a), labels.index(b)] += 1
    return M


def prf(y, yp):
    tp = np.sum((y == 1) & (yp == 1)); fp = np.sum((y == 0) & (yp == 1)); fn = np.sum((y == 1) & (yp == 0))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def roc(y, s):
    o = np.argsort(-s)
    y = np.asarray(y)[o]
    tpr = np.concatenate([[0], np.cumsum(y) / max(y.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / max((1 - y).sum(), 1)])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def regression_scores(y, yp):
    e = yp - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                R2=float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"))


def stratified_folds(y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), int)
    for c in np.unique(y):
        i = np.where(y == c)[0]
        rng.shuffle(i)
        folds[i] = np.arange(len(i)) % k
    return folds


def permutation_importance(score_fn, X, y, n_rep=5, seed=0):
    rng = np.random.default_rng(seed)
    base = score_fn(X, y)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        s = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(X)), j]
            s.append(base - score_fn(Xp, y))
        out[j] = np.mean(s)
    return out, base
