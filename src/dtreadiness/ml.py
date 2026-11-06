"""ML validation layer: RF / XGBoost / LightGBM / MLP with light tuning, CV metrics,
and SHAP-derived criterion importances."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             roc_auc_score, brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings("ignore")
SEED = 42


def make_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            random_state=SEED, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=400, max_depth=-1, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED,
            n_jobs=-1, verbose=-1),
        "MLP": Pipeline([("sc", StandardScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32),
                                               alpha=1e-3, max_iter=800,
                                               random_state=SEED))]),
    }


def evaluate_cv(X: pd.DataFrame, y: np.ndarray, n_splits=5):
    """Stratified k-fold CV. Returns per-model metric dict and out-of-fold proba."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    classes = np.unique(y)
    results = {}
    oof = {}
    for name, model in make_models().items():
        y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=1)
        y_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=1)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        kap = cohen_kappa_score(y, y_pred)
        try:
            auc = roc_auc_score(pd.get_dummies(y).values, y_proba,
                                average="macro", multi_class="ovr")
        except Exception:
            auc = np.nan
        # multiclass Brier
        Y = pd.get_dummies(y).values
        brier = np.mean(np.sum((y_proba - Y) ** 2, axis=1))
        # per-fold F1 SD for error bars
        f1s = []
        for tr, te in skf.split(X, y):
            mm = _clone_fit(model, X.iloc[tr], y[tr])
            f1s.append(f1_score(y[te], mm.predict(X.iloc[te]), average="macro"))
        results[name] = {"Accuracy": acc, "macroF1": f1, "Kappa": kap,
                         "AUC": auc, "Brier": brier, "F1_sd": np.std(f1s)}
        oof[name] = y_proba
    return results, oof


def _clone_fit(model, X, y):
    from sklearn.base import clone
    m = clone(model)
    m.fit(X, y)
    return m


def shap_importance(X: pd.DataFrame, y: np.ndarray, model_name="XGBoost"):
    """Fit best tree model on all data; return mean|SHAP| per feature (normalized)."""
    model = make_models()[model_name]
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # normalize shape to (n_samples, n_features)
    if isinstance(sv, list):                       # list over classes
        arr = np.mean([np.abs(s) for s in sv], axis=0)
    else:
        a = np.abs(np.asarray(sv))
        arr = a.mean(axis=2) if a.ndim == 3 else a
    imp = arr.mean(axis=0)
    imp = np.asarray(imp).ravel()
    s = pd.Series(imp, index=X.columns)
    return s / s.sum(), model, explainer, sv


# ---------------------------------------------------------------------------
# Outcome-prediction task (the non-circular ML evaluation)
# ---------------------------------------------------------------------------
def evaluate_outcome_task(scores: pd.DataFrame, outcome: pd.Series, ids: list,
                          n_splits: int = 5, n_repeats: int = 5, seed: int = SEED):
    """Predict the INDEPENDENT firm-performance outcome from the criterion scores.

    The tier-recovery task elsewhere in this module is circular by construction: tiers
    are a deterministic function of the same construct scores fed to the classifier.
    This task is not. The target here is the separately measured firm-performance
    composite, which contributes to no criterion, so predictive accuracy is genuine
    evidence that the criterion set carries information about realised outcomes.

    Firms in the middle performance tercile are dropped so the contrast is between
    clearly high- and clearly low-performing firms; this is the same discrimination
    task used for the AUC reported in the criterion-validity analysis.

    Returns (metrics DataFrame, n_used).
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    S = scores[ids].astype(float)
    idx = S.dropna().index.intersection(outcome.dropna().index)
    X_all = S.loc[idx].values
    y_all = outcome.loc[idx].values

    q1, q2 = np.quantile(y_all, [1 / 3, 2 / 3])
    keep = (y_all >= q2) | (y_all <= q1)
    X = X_all[keep]
    y = (y_all[keep] >= q2).astype(int)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    rows = {}
    for name, mdl in make_models().items():
        auc = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc")
        f1 = cross_val_score(mdl, X, y, cv=cv, scoring="f1_macro")
        acc = cross_val_score(mdl, X, y, cv=cv, scoring="accuracy")
        rows[name] = {"Accuracy": acc.mean(), "macroF1": f1.mean(),
                      "F1_sd": f1.std(), "AUC": auc.mean(), "AUC_sd": auc.std()}
    return pd.DataFrame(rows).T, int(len(y))
