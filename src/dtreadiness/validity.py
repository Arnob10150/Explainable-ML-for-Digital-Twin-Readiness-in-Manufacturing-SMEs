"""Construct-validity and common-method-bias diagnostics added in the revision.

Three families of check, each answering a specific reviewer concern:

1. ``htmt`` -- heterotrait-monotrait ratio (Henseler et al. 2015): discriminant
   validity, i.e. are the 16 criteria empirically distinguishable from each other?
2. ``harman_single_factor`` / ``full_collinearity_vif`` / ``marker_variable_test``
   -- common-method bias. Because every criterion and the performance outcome come
   from one self-report instrument answered by one respondent per firm, CMB has to
   be ruled out explicitly rather than assumed away.
3. ``weight_scheme_benchmark`` -- does ANY weighting beat equal weighting against an
   independently measured outcome, and where do the elicited weights fall relative
   to a random-weight null? This separates "weighting is useless here" from
   "these particular elicited weights fail to capture the available signal".
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Discriminant validity: HTMT
# ---------------------------------------------------------------------------
def htmt(item_frame: pd.DataFrame, construct_items: dict,
         threshold: float = 0.85):
    """Heterotrait-monotrait ratio of correlations.

    HTMT_ab = mean(|r| between items of a and b) / sqrt(mean within-a * mean within-b).
    Values below 0.85 support discriminant validity.
    """
    ids = list(construct_items)
    cols = [c for v in construct_items.values() for c in v]
    R = item_frame[cols].astype(float).corr().abs()

    def mono(a):
        cs = construct_items[a]
        v = [R.loc[i, j] for n, i in enumerate(cs) for j in cs[n + 1:]]
        return float(np.mean(v)) if v else np.nan

    def hetero(a, b):
        return float(np.mean([R.loc[i, j] for i in construct_items[a]
                              for j in construct_items[b]]))

    H = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for a in ids:
        for b in ids:
            H.loc[a, b] = np.nan if a == b else hetero(a, b) / np.sqrt(mono(a) * mono(b))
    viol = [(a, b, float(H.loc[a, b])) for a in ids for b in ids
            if a < b and H.loc[a, b] >= threshold]
    summary = {"max": float(np.nanmax(H.values)), "mean": float(np.nanmean(H.values)),
               "threshold": threshold, "n_violations": len(viol), "violations": viol,
               "passes": len(viol) == 0}
    return H, summary


# ---------------------------------------------------------------------------
# 2. Common-method bias
# ---------------------------------------------------------------------------
def harman_single_factor(item_frame: pd.DataFrame) -> dict:
    """Harman's one-factor test: does a single factor explain >50% of variance?"""
    X = item_frame.astype(float).dropna()
    Z = (X - X.mean()) / X.std()
    R = np.corrcoef(Z.values, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    tot = ev.sum()
    return {"n": int(len(X)), "n_items": int(X.shape[1]),
            "first_factor_pct": float(100 * ev[0] / tot),
            "n_eigen_gt1": int((ev > 1).sum()),
            "cum_first5_pct": float(100 * ev[:5].sum() / tot),
            "passes": bool(ev[0] / tot < 0.50)}


def full_collinearity_vif(construct_scores: pd.DataFrame, ids: list,
                          threshold: float = 3.3) -> dict:
    """Kock (2015) full collinearity test: all construct VIF < 3.3 => no CMB."""
    S = construct_scores[ids].astype(float).dropna()
    Sz = (S - S.mean()) / S.std()
    Cm = np.corrcoef(Sz.values, rowvar=False)
    vif = pd.Series(np.diag(inv(Cm)), index=ids).sort_values(ascending=False)
    return {"max_vif": float(vif.iloc[0]), "argmax": str(vif.index[0]),
            "mean_vif": float(vif.mean()), "threshold": threshold,
            "passes": bool(vif.max() < threshold), "vif": vif}


def marker_variable_test(construct_scores: pd.DataFrame, outcome: pd.Series,
                         ids: list) -> dict:
    """Marker-variable control over every candidate marker.

    For each criterion k, treat k as the marker: partial it out of both the
    readiness composite (built WITHOUT k, so construct signal is not removed) and
    the outcome, then re-estimate the association. A link that survives every
    choice of marker is not a method artifact. Reporting the full range avoids the
    arbitrariness of picking one marker.
    """
    S = construct_scores[ids].astype(float)
    Sz = (S - S.mean()) / S.std()
    idx = Sz.dropna().index.intersection(outcome.dropna().index)
    Sz, y = Sz.loc[idx], outcome.loc[idx]
    raw = float(spearmanr(Sz.mean(axis=1), y)[0])

    def _resid(a, ctrl):
        b = np.polyfit(ctrl, a, 1)
        return a - np.polyval(b, ctrl)

    rows = {}
    for k in ids:
        rest = Sz[[c for c in ids if c != k]].mean(axis=1).values
        ctrl = Sz[k].values
        rows[k] = float(spearmanr(_resid(rest, ctrl), _resid(y.values, ctrl))[0])
    v = pd.Series(rows)
    return {"raw_rho": raw, "min_partialled": float(v.min()),
            "max_partialled": float(v.max()), "median_partialled": float(v.median()),
            "all_above_030": bool((v > 0.30).all()), "by_marker": v,
            "passes": bool((v > 0.30).all())}


# ---------------------------------------------------------------------------
# 3. Does weighting carry signal at all?
# ---------------------------------------------------------------------------
def weight_scheme_benchmark(construct_scores: pd.DataFrame, outcome: pd.Series,
                            ids: list, w_elicited: pd.Series,
                            n_null: int = 2000, seed: int = 42) -> dict:
    """Compare elicited / equal / random / outcome-fitted weights on one outcome.

    The outcome-fitted scheme is scored OUT OF SAMPLE (repeated 5-fold) so it is an
    honest ceiling rather than an in-sample fit. The random-weight null locates each
    scheme in the distribution of arbitrary weightings.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import RepeatedKFold

    S = construct_scores[ids].astype(float)
    idx = S.dropna().index.intersection(outcome.dropna().index)
    Sz = ((S - S.mean()) / S.std()).loc[idx]
    y = outcome.loc[idx]
    X, yv = Sz.values, y.values
    n = len(ids)
    rng = np.random.default_rng(seed)

    def rho(w):
        return float(spearmanr(X @ np.asarray(w, float), yv)[0])

    eq = np.ones(n) / n
    w_el = w_elicited.reindex(ids).values.astype(float)
    null = np.array([rho(rng.dirichlet(np.ones(n))) for _ in range(n_null)])

    alphas = np.logspace(-3, 3, 25)
    rk = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)
    oof, cnt = np.zeros(len(yv)), np.zeros(len(yv))
    for tr, te in rk.split(X):
        m = RidgeCV(alphas=alphas).fit(X[tr], yv[tr])
        w = np.clip(m.coef_, 0, None)
        w = w / (w.sum() + 1e-12)
        oof[te] += X[te] @ w
        cnt[te] += 1
    r_fit_oos = float(spearmanr(oof / np.maximum(cnt, 1), yv)[0])

    r_eq, r_el = rho(eq), rho(w_el)
    return {"n": int(len(idx)),
            "rho_equal": r_eq, "rho_elicited": r_el, "rho_fitted_oos": r_fit_oos,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_p05": float(np.percentile(null, 5)),
            "null_p95": float(np.percentile(null, 95)),
            "pctile_equal": float(100 * (null < r_eq).mean()),
            "pctile_elicited": float(100 * (null < r_el).mean()),
            "headroom_fitted_vs_equal": r_fit_oos - r_eq,
            "elicited_vs_equal": r_el - r_eq}


# ---------------------------------------------------------------------------
# 4. Is there genuine cluster structure, or only a high/low split?
# ---------------------------------------------------------------------------
def cluster_structure_test(scores: pd.DataFrame, ids: list, c_range=range(2, 7),
                           n_null: int = 200, seed: int = 42) -> dict:
    """Compare observed clustering quality against a no-structure null.

    A reviewer rightly noted that a fuzzy partition coefficient near 1/c indicates
    essentially no cluster structure, so two "archetypes" may be nothing more than a
    high/low split of a single underlying readiness gradient. We therefore test this
    explicitly rather than reporting FPC alone:

    * the FPC/Xie-Beni of the real data is compared against data simulated from a
      multivariate normal with the SAME mean vector and covariance (a one-gradient,
      no-cluster null);
    * silhouette on the hard partition is compared against the same null;
    * the share of total variance captured by the first principal component
      quantifies how close the data is to a single dimension.

    A partition that is no better than the null is reported as a *stratification* of
    one gradient, not as distinct archetypes.
    """
    import skfuzzy as fuzz
    from sklearn.metrics import silhouette_score

    X = scores[ids].astype(float).dropna().values
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(seed)

    # how one-dimensional is the data?
    Cv = np.cov(Xz, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(Cv))[::-1]
    pc1 = float(ev[0] / ev.sum())

    def quality(A, c):
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            A.T, c, 2.0, error=1e-6, maxiter=1000, seed=seed)
        lab = np.argmax(u, axis=0)
        sil = float(silhouette_score(A, lab)) if len(set(lab)) > 1 else np.nan
        return float(fpc), sil

    out = {}
    for c in c_range:
        fpc_obs, sil_obs = quality(Xz, c)
        null_fpc, null_sil = [], []
        for _ in range(n_null):
            Y = rng.multivariate_normal(np.zeros(Xz.shape[1]), Cv, size=len(Xz))
            f, s = quality(Y, c)
            null_fpc.append(f); null_sil.append(s)
        null_fpc = np.array(null_fpc); null_sil = np.array(null_sil, dtype=float)
        out[int(c)] = {
            "fpc": fpc_obs, "fpc_null_mean": float(null_fpc.mean()),
            "fpc_null_p95": float(np.percentile(null_fpc, 95)),
            "fpc_exceeds_null": bool(fpc_obs > np.percentile(null_fpc, 95)),
            "silhouette": sil_obs,
            "sil_null_mean": float(np.nanmean(null_sil)),
            "sil_null_p95": float(np.nanpercentile(null_sil, 95)),
            "sil_exceeds_null": bool(sil_obs > np.nanpercentile(null_sil, 95)),
            "fpc_equals_1_over_c": bool(abs(fpc_obs - 1.0 / c) < 0.02),
        }
    any_struct = any(v["sil_exceeds_null"] for v in out.values())
    return {"pc1_variance_share": pc1, "by_c": out,
            "structure_beyond_null": any_struct,
            "verdict": ("distinct clusters supported" if any_struct else
                        "no structure beyond a single gradient: "
                        "partition is a stratification, not archetypes")}
