"""Outcome-supervised and hybrid criterion weighting.

This module gives the machine-learning component a substantive role in the framework.

In the original design ML predicted readiness *tiers*, but those tiers are a
deterministic function of the same construct scores used as predictors, so accuracy
there is circular and establishes nothing about readiness. Here the learner is pointed
at the one quantity it can genuinely be evaluated on: the independently measured
firm-performance outcome, which appears in no criterion.

Three weighting schemes are compared against that outcome:

* ``elicited``  -- Buckley fuzzy-AHP weights from the D1 practitioner survey;
* ``learned``   -- non-negative ridge coefficients fitted to the outcome;
* ``hybrid``    -- ``(1-lambda) * elicited + lambda * learned``.

The hybrid is the operationally interesting case: elicitation encodes domain knowledge
that a small sample cannot recover, while outcome data corrects importances that experts
systematically mis-state. The mixing weight ``lambda`` is a hyper-parameter, so it is
selected by *inner* cross-validation and scored on an untouched *outer* fold
(:func:`nested_cv_benchmark`). Reporting an in-sample optimum for ``lambda`` would
overstate the gain by roughly 0.025 Spearman in our data, which is the same class of
error this module exists to avoid.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import config as C

ALPHAS = np.logspace(-3, 3, 25)
LAMBDAS = np.linspace(0.0, 1.0, 21)


def _standardise(scores: pd.DataFrame, ids: list) -> pd.DataFrame:
    return pd.DataFrame(StandardScaler().fit_transform(scores[ids]),
                        columns=ids, index=scores.index)


def learned_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative, sum-to-one ridge weights fitted to the outcome.

    Coefficients are clipped at zero because a readiness criterion that *reduces*
    readiness is not interpretable; the surviving coefficients are renormalised.
    """
    coef = RidgeCV(alphas=ALPHAS).fit(X, y).coef_
    w = np.clip(coef, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(w)) / len(w)


def blend(w_elicited: np.ndarray, w_learned: np.ndarray, lam: float) -> np.ndarray:
    """Convex combination of elicited and learned weights."""
    w = (1.0 - lam) * w_elicited + lam * w_learned
    return w / w.sum()


def nested_cv_benchmark(scores: pd.DataFrame, outcome: pd.Series, ids: list,
                        w_elicited: pd.Series, n_outer: int = 5, n_inner: int = 5,
                        n_boot: int = 2000, seed: int = 42) -> dict:
    """Nested-CV comparison of equal / elicited / learned / hybrid weighting.

    For each outer fold, ``lambda`` is chosen by inner cross-validation on the training
    part only, and the resulting weight vector scores the held-out part. Out-of-fold
    predictions are pooled and correlated with the outcome, so no scheme is evaluated on
    data that informed it. A bootstrap confidence interval is reported for the
    hybrid-minus-elicited gain.
    """
    S = scores[ids].astype(float)
    idx = S.dropna().index.intersection(outcome.dropna().index)
    Z = _standardise(S.loc[idx], ids)
    X, y = Z.values, outcome.loc[idx].values
    n = len(ids)

    w_el = w_elicited.reindex(ids).values.astype(float)
    w_eq = np.ones(n) / n

    pred = {k: np.zeros(len(y)) for k in ("equal", "elicited", "learned", "hybrid")}
    chosen_lams = []

    outer = KFold(n_outer, shuffle=True, random_state=seed)
    for tr, te in outer.split(X):
        Xtr, ytr = X[tr], y[tr]
        # --- select lambda on inner folds only ---
        best_lam, best_r = 0.0, -np.inf
        inner = KFold(n_inner, shuffle=True, random_state=seed + 1)
        for lam in LAMBDAS:
            oof = np.zeros(len(tr))
            for i_tr, i_te in inner.split(Xtr):
                wl = learned_weights(Xtr[i_tr], ytr[i_tr])
                oof[i_te] = Xtr[i_te] @ blend(w_el, wl, lam)
            r = spearmanr(oof, ytr)[0]
            if r > best_r:
                best_r, best_lam = r, lam
        chosen_lams.append(float(best_lam))

        wl = learned_weights(Xtr, ytr)
        pred["equal"][te] = X[te] @ w_eq
        pred["elicited"][te] = X[te] @ w_el
        pred["learned"][te] = X[te] @ wl
        pred["hybrid"][te] = X[te] @ blend(w_el, wl, best_lam)

    rho = {k: float(spearmanr(v, y)[0]) for k, v in pred.items()}

    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        gains[b] = (spearmanr(pred["hybrid"][s], y[s])[0]
                    - spearmanr(pred["elicited"][s], y[s])[0])

    # final weight vector, refit on all data at the modal lambda, for reporting
    lam_star = float(np.median(chosen_lams))
    w_final = blend(w_el, learned_weights(X, y), lam_star)

    return {
        "n": int(len(idx)),
        "rho": rho,
        "lambda_per_fold": chosen_lams,
        "lambda_median": lam_star,
        "hybrid_minus_elicited": float(gains.mean()),
        "gain_ci_lo": float(np.percentile(gains, 2.5)),
        "gain_ci_hi": float(np.percentile(gains, 97.5)),
        "gain_p_positive": float((gains > 0).mean()),
        "w_hybrid": pd.Series(w_final, index=ids),
        "w_learned": pd.Series(learned_weights(X, y), index=ids),
    }
