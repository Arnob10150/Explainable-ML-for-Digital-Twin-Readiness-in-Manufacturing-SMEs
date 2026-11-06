"""Data ingestion, Likert mapping, and construct-score aggregation for D1 and D2."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from . import config as C


# ---------------------------------------------------------------------------
# D1 -- Digital-Twin adoption survey (Malaysian semiconductor, n=150)
# ---------------------------------------------------------------------------
def load_d1() -> pd.DataFrame:
    df = pd.read_excel(C.D1_PATH)
    return df


def d1_construct_importance() -> pd.Series:
    """Mean Likert agreement per DT construct = expert-proxy importance for AHP.

    Returns a Series indexed by criterion id (C1..C12).
    """
    df = load_d1()
    # map likert strings -> ints
    def m(x):
        if pd.isna(x):
            return np.nan
        return C.LIKERT_MAP.get(str(x).strip().lower(), np.nan)

    imp = {}
    for cid, prefix in C.CROSSWALK_D1.items():
        cols = [c for c in df.columns if c.startswith(prefix + " [") or c == prefix]
        if not cols:
            # fall back: any column containing the prefix text
            cols = [c for c in df.columns if prefix.lower() in c.lower()]
        vals = df[cols].applymap(m)
        imp[cid] = np.nanmean(vals.values)
    return pd.Series(imp).reindex(C.CRIT_IDS)


# ---------------------------------------------------------------------------
# D2 -- Malaysia SMEs survey (n=321, 1..5 Likert integers)
# ---------------------------------------------------------------------------
DEMOG = ["Age", "Race", "Education", "Position", "Tenure",
         "Firmage", "Location", "Employee", "Ownership", "Industry"]


def load_d2() -> pd.DataFrame:
    return pd.read_excel(C.D2_PATH)


def _group_cols(df: pd.DataFrame, prefix: str):
    pat = re.compile(rf"^{re.escape(prefix)}\d+$")
    return [c for c in df.columns if pat.match(str(c))]


def d2_construct_scores() -> pd.DataFrame:
    """Aggregate D2 item groups into the 12 criteria construct scores (mean, 1..5).

    Returns DataFrame (321 x 12) indexed by SME id, columns = C1..C12, plus demographics.
    """
    df = load_d2().copy()
    scores = pd.DataFrame(index=df.index)
    for cid, prefixes in C.CROSSWALK_D2.items():
        cols = []
        for p in prefixes:
            cols += _group_cols(df, p)
        block = df[cols].apply(pd.to_numeric, errors="coerce")
        scores[cid] = block.mean(axis=1)
    # median-impute any residual missing construct scores
    scores = scores.fillna(scores.median())
    # attach demographics
    for d in DEMOG:
        if d in df.columns:
            scores[d] = df[d].values
    scores.index.name = "SME"
    return scores


def d2_item_frame() -> pd.DataFrame:
    """Return the raw item-level frame with only the modelled Likert items."""
    df = load_d2().copy()
    keep = []
    for prefixes in C.CROSSWALK_D2.values():
        for p in prefixes:
            keep += _group_cols(df, p)
    return df[keep].apply(pd.to_numeric, errors="coerce")


def construct_item_map() -> dict:
    """criterion id -> list of D2 item columns (for reliability analysis)."""
    df = load_d2()
    out = {}
    for cid, prefixes in C.CROSSWALK_D2.items():
        cols = []
        for p in prefixes:
            cols += _group_cols(df, p)
        out[cid] = cols
    return out


def d1_respondent_importance() -> pd.DataFrame:
    """Per-respondent importance ratings per criterion (for bootstrap CIs).

    Same crosswalk as :func:`d1_construct_importance`, but keeps one row per D1
    respondent instead of collapsing to the mean, so the elicitation uncertainty
    can be resampled.
    """
    df = load_d1()

    def m(x):
        if pd.isna(x):
            return np.nan
        return C.LIKERT_MAP.get(str(x).strip().lower(), np.nan)

    out = {}
    for cid, prefix in C.CROSSWALK_D1.items():
        cols = [c for c in df.columns if c.startswith(prefix + " [") or c == prefix]
        if not cols:
            cols = [c for c in df.columns if prefix.lower() in c.lower()]
        vals = df[cols].apply(lambda s: s.map(m))
        out[cid] = vals.mean(axis=1)
    return pd.DataFrame(out).reindex(columns=C.CRIT_IDS)


def d2_outcome_score() -> pd.Series:
    """Independent firm-performance outcome (z-scored mean of OUTCOME_D2 items).

    These items are deliberately excluded from every readiness criterion so the
    criterion-validity test is non-circular.
    """
    df = load_d2().copy()
    cols = []
    for p in C.OUTCOME_D2:
        cols += _group_cols(df, p)
    O = df[cols].apply(pd.to_numeric, errors="coerce")
    Oz = (O - O.mean()) / O.std()
    s = Oz.mean(axis=1)
    s.index.name = "SME"
    s.name = "performance"
    return s


def d2_capability_scores() -> pd.DataFrame:
    """Leakage-free construct scores for outcome-validation analyses.

    In the 12-criterion taxonomy C9 ("Production & Performance") is built from
    OPSF/OPSNF/EPRO, and OPSF/OPSNF are exactly the firm-performance items used as
    the independent outcome. Feeding C9 into a model that predicts that outcome is
    circular. This accessor returns the same 12 construct scores but rebuilds C9
    from its capability-only item group (EPRO), so the outcome-prediction,
    hybrid-weighting and common-method-bias analyses are non-circular. All other
    criteria are unchanged, so the main framework (which legitimately includes C9
    as a readiness construct) is not affected.
    """
    df = load_d2().copy()
    scores = pd.DataFrame(index=df.index)
    for cid, prefixes in C.CROSSWALK_D2.items():
        pref = prefixes
        if cid == "C9":
            pref = [p for p in prefixes if p not in C.OUTCOME_D2]  # drop OPSF/OPSNF
            if not pref:
                pref = prefixes
        cols = []
        for p in pref:
            cols += _group_cols(df, p)
        block = df[cols].apply(pd.to_numeric, errors="coerce")
        scores[cid] = block.mean(axis=1)
    scores = scores.fillna(scores.median())
    scores.index.name = "SME"
    return scores
