#!/usr/bin/env python3
"""External validation on an INDEPENDENT international cohort (issue 6).

Mendeley Data 76z794cxdg: "Survey dataset of Dynamic Capabilities for leading
and managing Digital Transformation initiatives" (446 respondents, 51 countries).
It carries (a) Dynamic-Capability items across three phases -- Sensing, Seizing,
Transforming -- that serve as CAPABILITY predictors, and (b) a separately-measured
"DT success" item that serves as an INDEPENDENT outcome. We therefore replicate,
out-of-sample and cross-border, the Malaysian criterion-validity result: does a
capability-readiness score predict realized DT success? Because this cohort never
touched the framework's weight fitting, it is a genuine external test. We also
lock the Malaysia-fitted phase emphasis and check ranking transfer + tier stability.
Saves external_validation.json.
"""
import os, re, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XLSX = os.path.abspath(os.environ.get(
    "DT_READINESS_EXTERNAL_DATA",
    os.path.join(ROOT, "data", "raw", "external", "Survey data for DT study.xlsx")))
OUT = os.path.join(ROOT, "data", "processed", "external_validation.json")
RNG = 42; np.random.seed(RNG)


def num(x):
    if isinstance(x, str):
        m = re.match(r"\s*(\d+)\s*[.\u2022]", x)
        if m:
            return int(m.group(1))
    return np.nan


d = pd.ExcelFile(XLSX).parse("Mass Survey Data")
data = d.iloc[1:].reset_index(drop=True)

succ = data.iloc[:, 4].map(num)                          # DT success (independent outcome)
resist = data.iloc[:, 5].map(num)                        # tech-change resistance (context)
sensing = list(range(24, 29)); seizing = list(range(29, 34)); transforming = list(range(35, 40))
phase = {"Sensing": sensing, "Seizing": seizing, "Transforming": transforming}

M = pd.DataFrame({f"i{c}": data.iloc[:, c].map(num) for ph in phase.values() for c in ph})
ph_scores = pd.DataFrame({p: data.iloc[:, cs].applymap(num).mean(axis=1) for p, cs in phase.items()})

df = pd.concat([ph_scores, succ.rename("success"), resist.rename("resist")], axis=1).dropna()
n = len(df)
R = {"dataset": "Mendeley 76z794cxdg (Dynamic Capabilities DT survey)", "n": int(n),
     "n_capability_items": M.shape[1], "phases": {k: len(v) for k, v in phase.items()}}

# --- equal-weight capability readiness (phase-level composite) --------------
capz = pd.DataFrame(StandardScaler().fit_transform(df[list(phase)]), columns=list(phase))
readiness = capz.mean(axis=1)
y = df["success"].values

# (1) correlation readiness -> DT success
rho, p_rho = spearmanr(readiness, y); r, p_r = pearsonr(readiness, y)
R["spearman_readiness_success"] = {"rho": float(rho), "p": float(p_rho)}
R["pearson_readiness_success"] = {"r": float(r), "p": float(p_r)}

# (2) predictive CV R2 (non-circular: outcome never used to build readiness)
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RNG)
X = capz.values
R["cv_r2_predict_success"] = {
    "ridge": float(cross_val_score(Ridge(1.0), X, y, cv=rkf, scoring="r2").mean()),
    "rf": float(cross_val_score(RandomForestRegressor(400, random_state=RNG), X, y,
                                cv=rkf, scoring="r2").mean())}

# (3) discrimination: top vs bottom success tercile
q1, q2 = np.quantile(y, [1/3, 2/3])
mask = (y >= q2) | (y <= q1)
yb = (y[mask] >= q2).astype(int); Xb = X[mask]
scv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
R["cv_auc_hi_vs_lo"] = float(cross_val_score(LogisticRegression(max_iter=1000), Xb, yb,
                                             cv=scv, scoring="roc_auc").mean())
R["auc_readiness_only"] = float(roc_auc_score(yb, readiness.values[mask]))

# (4) ranking transfer: lock Malaysia-fitted phase emphasis (build->scale->sustain)
locked_w = np.array([0.25, 0.35, 0.40])                  # transforming-heavy, Malaysia-shaped
readiness_locked = (capz.values * locked_w).sum(axis=1)
rho_locked, _ = spearmanr(readiness_locked, y)
rho_equal, _ = spearmanr(capz.values.mean(axis=1), y)
# Calibrate against a random-weight null: with only 3 phase composites that are
# positively intercorrelated, almost ANY non-degenerate weighting gives a similar
# rho, so "locked weights transfer as well as equal weights" is NOT evidence that
# the weighting generalises -- it is evidence that this outcome cannot discriminate
# between weightings. The null below makes that explicit.
rng_ = np.random.default_rng(RNG)
null3 = np.array([spearmanr(capz.values @ rng_.dirichlet(np.ones(3)), y)[0]
                  for _ in range(2000)])
# honest ceiling: outcome-fitted weights, scored out of sample
from sklearn.linear_model import RidgeCV
oof = np.zeros(len(y)); cnt = np.zeros(len(y))
for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=RNG).split(X):
    mdl = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], y[tr])
    wv = np.clip(mdl.coef_, 0, None); wv = wv / (wv.sum() + 1e-12)
    oof[te] += X[te] @ wv; cnt[te] += 1
rho_fit = float(spearmanr(oof / np.maximum(cnt, 1), y)[0])
R["ranking_transfer_locked_weights"] = {
    "weights": {"Sensing": 0.25, "Seizing": 0.35, "Transforming": 0.40},
    "spearman_locked_vs_success": float(rho_locked),
    "spearman_equal_vs_success": float(rho_equal),
    "random_weight_null": {"mean": float(null3.mean()), "sd": float(null3.std()),
                           "p05": float(np.percentile(null3, 5)),
                           "p95": float(np.percentile(null3, 95))},
    "spearman_outcome_fitted_oos": rho_fit,
    "null_spread": float(np.percentile(null3, 95) - np.percentile(null3, 5)),
    "interpretation": ("the locked and equal weightings differ by far less than the "
                       "random-weight null spread, so this outcome has insufficient "
                       "resolution to validate a weight vector; it validates only the "
                       "capability-outcome link"),
}

# (5) tier stability: terciles on readiness, mean success per tier
tier = pd.qcut(readiness, 3, labels=["Emerging", "Developing", "Advancing"])
R["tier_mean_success"] = {str(k): float(v) for k, v in df["success"].groupby(tier).mean().items()}
R["tier_sizes"] = {str(k): int(v) for k, v in tier.value_counts().items()}

json.dump(R, open(OUT, "w"), indent=2)
print(f"n={n} | cap items={M.shape[1]}")
print(f"Spearman(readiness, DT success) rho={rho:.3f} (p={p_rho:.1e})")
print(f"CV R2 predict success: Ridge={R['cv_r2_predict_success']['ridge']:.3f}, "
      f"RF={R['cv_r2_predict_success']['rf']:.3f}")
print(f"CV AUC hi-vs-lo tercile={R['cv_auc_hi_vs_lo']:.3f} "
      f"(readiness-only {R['auc_readiness_only']:.3f})")
print(f"locked-weight ranking transfer rho={rho_locked:.3f}")
print("tier mean success:", {k: round(v, 2) for k, v in R['tier_mean_success'].items()})
print("saved", OUT)
