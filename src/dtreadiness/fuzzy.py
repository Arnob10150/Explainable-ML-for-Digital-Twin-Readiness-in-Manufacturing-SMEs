"""Fuzzy MCDM core: Triangular Fuzzy Numbers, Fuzzy-AHP (Chang extent + Buckley),
Fuzzy-TOPSIS (Chen 2000) and Fuzzy-VIKOR (cross-check).

All equations are numbered to match the paper (Eq. 1-13).
"""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
import numpy as np


# ---------------------------------------------------------------------------
# Triangular Fuzzy Number
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TFN:
    l: float
    m: float
    u: float

    def __post_init__(self):
        # allow tiny numerical slack
        assert self.l <= self.m + 1e-9 and self.m <= self.u + 1e-9, \
            f"TFN order violated: {self.l},{self.m},{self.u}"

    # Eq. (1) fuzzy addition
    def __add__(self, o: "TFN") -> "TFN":
        return TFN(self.l + o.l, self.m + o.m, self.u + o.u)

    def __sub__(self, o: "TFN") -> "TFN":
        return TFN(self.l - o.u, self.m - o.m, self.u - o.l)

    # Eq. (2) fuzzy multiplication
    def __mul__(self, o) -> "TFN":
        if isinstance(o, TFN):
            return TFN(self.l * o.l, self.m * o.m, self.u * o.u)
        return TFN(self.l * o, self.m * o, self.u * o)  # scalar

    def inv(self) -> "TFN":
        return TFN(1.0 / self.u, 1.0 / self.m, 1.0 / self.l)

    def defuzz(self) -> float:
        """Centroid defuzzification."""
        return (self.l + self.m + self.u) / 3.0

    # Eq. (9) vertex distance
    def distance(self, o: "TFN") -> float:
        return sqrt(((self.l - o.l) ** 2 + (self.m - o.m) ** 2 + (self.u - o.u) ** 2) / 3.0)

    def as_tuple(self):
        return (self.l, self.m, self.u)


def tfn(t) -> TFN:
    return TFN(*t) if not isinstance(t, TFN) else t


# ---------------------------------------------------------------------------
# Fuzzy AHP -- Chang (1996) extent analysis
# ---------------------------------------------------------------------------
def _degree_possibility(s1: TFN, s2: TFN) -> float:
    """V(S1 >= S2), Eq. (4)."""
    if s1.m >= s2.m:
        return 1.0
    if s2.l >= s1.u:
        return 0.0
    denom = (s1.m - s1.u) - (s2.m - s2.l)
    return (s2.l - s1.u) / denom


def chang_extent_weights(fpcm):
    """Chang extent analysis. fpcm: n x n list/array of TFN. Returns crisp weights.

    Implements Eq. (3) synthetic extent, Eq. (4) possibility degree, Eq. (5) weight.
    """
    n = len(fpcm)
    # row-wise fuzzy sum
    row_sums = []
    for i in range(n):
        s = TFN(0, 0, 0)
        for j in range(n):
            s = s + tfn(fpcm[i][j])
        row_sums.append(s)
    # grand sum
    total = TFN(0, 0, 0)
    for s in row_sums:
        total = total + s
    total_inv = total.inv()
    # Eq. (3) synthetic extent S_i
    S = [row_sums[i] * total_inv for i in range(n)]
    # Eq. (5): d(A_i) = min_j V(S_i >= S_j)
    d = np.zeros(n)
    for i in range(n):
        mins = [_degree_possibility(S[i], S[j]) for j in range(n) if j != i]
        d[i] = min(mins) if mins else 1.0
    d = np.clip(d, 0, None)
    if d.sum() == 0:
        d = np.ones(n)
    return d / d.sum()


def buckley_weights(fpcm):
    """Buckley (1985) geometric-mean method. Returns list[TFN] fuzzy weights."""
    n = len(fpcm)
    r = []  # geometric mean per row (TFN)
    for i in range(n):
        prod_l = prod_m = prod_u = 1.0
        for j in range(n):
            t = tfn(fpcm[i][j])
            prod_l *= t.l; prod_m *= t.m; prod_u *= t.u
        r.append(TFN(prod_l ** (1.0 / n), prod_m ** (1.0 / n), prod_u ** (1.0 / n)))
    sum_l = sum(x.l for x in r); sum_m = sum(x.m for x in r); sum_u = sum(x.u for x in r)
    inv = TFN(1.0 / sum_u, 1.0 / sum_m, 1.0 / sum_l)
    return [x * inv for x in r]


RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
      9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56}


def consistency_ratio(crisp_pcm) -> float:
    """Saaty consistency ratio on the defuzzified pairwise matrix."""
    A = np.asarray(crisp_pcm, dtype=float)
    n = A.shape[0]
    w = A.mean(axis=1)  # simple priority estimate
    w = w / w.sum()
    lam = float(np.mean((A @ w) / w))
    ci = (lam - n) / (n - 1)
    return ci / RI.get(n, 1.49)


# ---------------------------------------------------------------------------
# Fuzzy TOPSIS -- Chen (2000)
# ---------------------------------------------------------------------------
def fuzzy_topsis(D, w, benefit=None):
    """D: (m alternatives x n criteria) of TFN. w: list[TFN] criterion weights.

    Returns dict with CC (Eq. 11), d_plus, d_minus.
    """
    m = len(D); n = len(D[0])
    if benefit is None:
        benefit = [True] * n
    w = [tfn(x) for x in w]

    # Eq. (6) linear normalization
    R = [[None] * n for _ in range(m)]
    for j in range(n):
        col = [tfn(D[i][j]) for i in range(m)]
        if benefit[j]:
            u_star = max(c.u for c in col) or 1.0
            for i in range(m):
                c = col[i]
                R[i][j] = TFN(c.l / u_star, c.m / u_star, c.u / u_star)
        else:  # cost criterion
            l_minus = min(c.l for c in col)
            for i in range(m):
                c = col[i]
                R[i][j] = TFN(l_minus / c.u, l_minus / c.m, l_minus / c.l)

    # Eq. (7) weighted normalized matrix
    V = [[R[i][j] * w[j] for j in range(n)] for i in range(m)]

    # Eq. (8) FPIS / FNIS
    FPIS = [TFN(1, 1, 1) for _ in range(n)]
    FNIS = [TFN(0, 0, 0) for _ in range(n)]

    # Eq. (10) distances
    d_plus = np.zeros(m); d_minus = np.zeros(m)
    for i in range(m):
        d_plus[i] = sum(V[i][j].distance(FPIS[j]) for j in range(n))
        d_minus[i] = sum(V[i][j].distance(FNIS[j]) for j in range(n))

    # Eq. (11) closeness coefficient
    CC = d_minus / (d_plus + d_minus)
    return {"CC": CC, "d_plus": d_plus, "d_minus": d_minus, "V": V}


# ---------------------------------------------------------------------------
# Fuzzy VIKOR (cross-method sanity check)
# ---------------------------------------------------------------------------
def fuzzy_vikor(D, w, benefit=None, v=0.5):
    """Defuzzified VIKOR on TFN decision matrix. Returns Q (lower = better)."""
    m = len(D); n = len(D[0])
    if benefit is None:
        benefit = [True] * n
    X = np.array([[tfn(D[i][j]).defuzz() for j in range(n)] for i in range(m)])
    wv = np.array([tfn(x).defuzz() for x in w]); wv = wv / wv.sum()

    fstar = np.where(benefit, X.max(axis=0), X.min(axis=0))
    fminus = np.where(benefit, X.min(axis=0), X.max(axis=0))
    denom = np.where(np.abs(fstar - fminus) < 1e-12, 1e-12, fstar - fminus)

    ratio = (fstar - X) / denom  # benefit orientation
    S = (wv * ratio).sum(axis=1)
    Rr = (wv * ratio).max(axis=1)
    def nrm(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-12)
    Q = v * nrm(S) + (1 - v) * nrm(Rr)
    return Q


# ---------------------------------------------------------------------------
# Best-Worst Method (Rezaei 2015) -- second elicitation method for comparison
# ---------------------------------------------------------------------------
def bwm_weights(importance, eps: float = 1e-9):
    """Derive BWM weights from an importance vector.

    The original paper listed a Fuzzy-AHP vs. BWM comparison as future work; it is
    implemented here as an independent cross-check on the elicited weight vector.
    BWM needs only 2n-3 comparisons (each criterion against the single best and the
    single worst) instead of AHP's n(n-1)/2, so it is far less prone to the rounding
    inconsistency that AHP pairwise construction introduces.

    We derive best-to-others (a_Bj) and others-to-worst (a_jW) preference ratios from
    the same importance vector used for AHP, mapped onto the 1..9 scale, then solve
    the linear min-max BWM programme of Rezaei (2016) in closed form via its
    analytical solution for consistent comparisons, falling back to a least-squares
    solve when the system is inconsistent.

    Returns (weights ndarray, consistency_ratio float).
    """
    v = np.asarray(importance, dtype=float)
    n = len(v)
    b, w_ = int(np.argmax(v)), int(np.argmin(v))
    # map importances onto 1..9 preference ratios relative to best / worst
    rng = v.max() - v.min()
    if rng < eps:
        return np.ones(n) / n, 0.0
    scaled = 1.0 + 8.0 * (v - v.min()) / rng          # 1..9, higher = more important
    a_Bj = scaled[b] / scaled                          # best vs. others
    a_jW = scaled / scaled[w_]                         # others vs. worst
    a_Bj = np.clip(np.round(a_Bj), 1, 9)
    a_jW = np.clip(np.round(a_jW), 1, 9)
    a_BW = float(np.clip(np.round(scaled[b] / scaled[w_]), 1, 9))

    # Least-squares BWM: minimise sum (w_B - a_Bj w_j)^2 + (w_j - a_jW w_W)^2
    # subject to sum w = 1, w >= 0.  Solved as a small constrained LS problem.
    from scipy.optimize import lsq_linear
    rows, rhs = [], []
    for j in range(n):
        if j != b:
            r = np.zeros(n); r[b] = 1.0; r[j] = -a_Bj[j]
            rows.append(r); rhs.append(0.0)
        if j != w_:
            r = np.zeros(n); r[j] = 1.0; r[w_] = -a_jW[j]
            rows.append(r); rhs.append(0.0)
    r = np.ones(n); rows.append(r * 10.0); rhs.append(10.0)   # sum-to-one, upweighted
    A = np.vstack(rows); bvec = np.asarray(rhs, float)
    sol = lsq_linear(A, bvec, bounds=(0, np.inf))
    w = np.clip(sol.x, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n

    # Rezaei (2016) consistency index for the given a_BW
    CI_TABLE = {1: 0.0, 2: 0.44, 3: 1.0, 4: 1.63, 5: 2.30, 6: 3.00, 7: 3.73,
                8: 4.47, 9: 5.23}
    xi = 0.0
    for j in range(n):
        if j != b:
            xi = max(xi, abs(w[b] - a_Bj[j] * w[j]))
        if j != w_:
            xi = max(xi, abs(w[j] - a_jW[j] * w[w_]))
    ci = CI_TABLE.get(int(a_BW), 5.23)
    cr = float(xi / ci) if ci > 0 else 0.0
    return w, cr
