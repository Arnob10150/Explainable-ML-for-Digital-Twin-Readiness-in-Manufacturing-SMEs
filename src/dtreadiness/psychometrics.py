"""Reliability & validity: Cronbach alpha, Composite Reliability, AVE, KMO, Bartlett.

Self-contained numpy/scipy implementation (no factor_analyzer dependency, to stay
compatible across sklearn versions).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import inv, det, eigh
from scipy.stats import chi2 as chi2_dist


def cronbach_alpha(X: pd.DataFrame) -> float:
    X = X.dropna()
    k = X.shape[1]
    if k < 2:
        return np.nan
    var_items = X.var(axis=0, ddof=1).sum()
    var_total = X.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return np.nan
    return (k / (k - 1)) * (1 - var_items / var_total)


def _first_factor_loadings(X: pd.DataFrame):
    """Standardized loadings on the first principal component of the correlation matrix."""
    R = np.corrcoef(X.dropna().values, rowvar=False)
    vals, vecs = eigh(R)
    idx = np.argsort(vals)[::-1]
    lam1 = max(vals[idx][0], 0.0)
    v1 = vecs[:, idx[0]]
    load = v1 * np.sqrt(lam1)
    # orient so the majority of loadings are positive
    if np.sum(load) < 0:
        load = -load
    return np.abs(load)


def cr_ave(X: pd.DataFrame):
    X = X.dropna()
    if X.shape[1] < 2:
        return np.nan, np.nan
    load = _first_factor_loadings(X)
    sum_l = load.sum()
    err = (1 - load ** 2).sum()
    cr = sum_l ** 2 / (sum_l ** 2 + err)
    ave = (load ** 2).mean()
    return float(cr), float(ave)


def kmo(X: pd.DataFrame) -> float:
    Xv = X.dropna().values
    R = np.corrcoef(Xv, rowvar=False)
    try:
        Rinv = inv(R)
    except np.linalg.LinAlgError:
        Rinv = np.linalg.pinv(R)
    # partial correlation matrix
    d = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    P = -Rinv / d
    np.fill_diagonal(P, 0)
    np.fill_diagonal(R, 0)
    r2 = np.sum(R ** 2)
    p2 = np.sum(P ** 2)
    return float(r2 / (r2 + p2))


def bartlett_sphericity(X: pd.DataFrame):
    Xv = X.dropna().values
    n, p = Xv.shape
    R = np.corrcoef(Xv, rowvar=False)
    dR = det(R)
    dR = max(dR, 1e-12)
    stat = -(n - 1 - (2 * p + 5) / 6.0) * np.log(dR)
    df = p * (p - 1) / 2
    pval = chi2_dist.sf(stat, df)
    return float(stat), float(pval)


def kmo_bartlett(X: pd.DataFrame):
    try:
        k = kmo(X)
        chi2, p = bartlett_sphericity(X)
    except Exception:
        return np.nan, np.nan, np.nan
    return k, chi2, p


def reliability_table(construct_items: dict, item_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, cols in construct_items.items():
        X = item_frame[cols]
        a = cronbach_alpha(X)
        cr, ave = cr_ave(X)
        rows.append({"Criterion": cid, "k_items": len(cols),
                     "Cronbach_alpha": a, "CR": cr, "AVE": ave})
    return pd.DataFrame(rows)
