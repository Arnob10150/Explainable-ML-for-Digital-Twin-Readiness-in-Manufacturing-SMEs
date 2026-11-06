"""AHP matrix construction, clustering (FCM), sensitivity, weight-agreement,
and per-SME prescriptive gap decomposition."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import skfuzzy as fuzz
from . import config as C
from .fuzzy import TFN, tfn, chang_extent_weights, buckley_weights, consistency_ratio, fuzzy_topsis


# ---------------------------------------------------------------------------
# Build fuzzy pairwise comparison matrix from an importance vector
# ---------------------------------------------------------------------------
def importance_to_fpcm(importance: pd.Series, intensity_max: float = 5.0):
    """Map construct-mean importances to a priority-intensity scale, then to a
    Saaty-rounded, fuzzified pairwise comparison matrix.

    The survey importances are compressed (respondents mostly "Agree"), so we
    min-max scale them onto a priority intensity p in [1, intensity_max]. Pairwise
    intensities a_ij = round(p_i / p_j) on the Saaty 1..9 scale; rounding induces
    a small, realistic inconsistency (CR > 0). Returns (fpcm[list[list[TFN]]], crisp).
    """
    v = importance.reindex(C.CRIT_IDS).values.astype(float)
    n = len(v)
    frac = (v - v.min()) / (v.max() - v.min() + 1e-12)
    p = 1.0 + (intensity_max - 1.0) * frac        # priority intensity in [1, imax]
    fpcm = [[None] * n for _ in range(n)]
    crisp = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ratio = p[i] / p[j]
            if ratio >= 1:
                s = int(np.clip(round(ratio), 1, 9))
                crisp[i][j] = s; crisp[j][i] = 1.0 / s
                fpcm[i][j] = tfn(C.saaty_tfn(s))
                fpcm[j][i] = fpcm[i][j].inv()
            else:
                s = int(np.clip(round(1.0 / ratio), 1, 9))
                crisp[i][j] = 1.0 / s; crisp[j][i] = s
                fpcm[j][i] = tfn(C.saaty_tfn(s))
                fpcm[i][j] = fpcm[j][i].inv()
    for i in range(n):
        fpcm[i][i] = TFN(1, 1, 1)
    return fpcm, crisp


def derive_weights(importance: pd.Series):
    fpcm, crisp = importance_to_fpcm(importance)
    w_chang = chang_extent_weights(fpcm)
    w_buckley_tfn = buckley_weights(fpcm)
    w_buckley = np.array([t.defuzz() for t in w_buckley_tfn])
    w_buckley = w_buckley / w_buckley.sum()
    cr = consistency_ratio(crisp)
    return {
        "w_chang": pd.Series(w_chang, index=C.CRIT_IDS),
        "w_buckley": pd.Series(w_buckley, index=C.CRIT_IDS),
        "w_buckley_tfn": w_buckley_tfn,
        "CR": cr,
        "fpcm": fpcm,
    }


# ---------------------------------------------------------------------------
# Fuzzify SME construct scores (1..5 mean -> nearest linguistic TFN)
# ---------------------------------------------------------------------------
def fuzzify_scores(scores: pd.DataFrame):
    """scores: (m x 12) construct means in [1,5]. Returns list-of-list TFN."""
    D = []
    for _, row in scores[C.CRIT_IDS].iterrows():
        D.append([tfn(C.LINGUISTIC_TFN[int(np.clip(round(x), 1, 5))]) for x in row.values])
    return D


# ---------------------------------------------------------------------------
# Weight agreement (Eq. 12)
# ---------------------------------------------------------------------------
def weight_agreement(w_ahp: pd.Series, w_shap: pd.Series):
    a = w_ahp.reindex(C.CRIT_IDS).values
    s = w_shap.reindex(C.CRIT_IDS).values
    rho, p_rho = spearmanr(a, s)
    tau, p_tau = kendalltau(a, s)
    delta = pd.Series(a - s, index=C.CRIT_IDS)
    return {"spearman": rho, "spearman_p": p_rho, "kendall": tau,
            "kendall_p": p_tau, "delta": delta}


# ---------------------------------------------------------------------------
# Sensitivity: OAT + Monte-Carlo Dirichlet
# ---------------------------------------------------------------------------
def oat_sensitivity(D, w_ahp: pd.Series, base_cc, perturb=(0.1, 0.2, 0.3)):
    """Perturb each weight by +/- p, renormalize, recompute ranks. Returns heatmap df."""
    base_rank = pd.Series(base_cc).rank(ascending=False).values
    n = len(C.CRIT_IDS)
    records = {}
    for p in perturb:
        for sign, tag in [(+1, f"+{int(p*100)}%"), (-1, f"-{int(p*100)}%")]:
            changes = []
            for k in range(n):
                w = w_ahp.values.astype(float).copy()
                w[k] *= (1 + sign * p)
                w = np.clip(w, 1e-6, None); w /= w.sum()
                cc = fuzzy_topsis(D, [TFN(x, x, x) for x in w])["CC"]
                r = pd.Series(cc).rank(ascending=False).values
                changes.append(np.mean(np.abs(r - base_rank)))
            records[tag] = changes
    hm = pd.DataFrame(records, index=C.CRIT_IDS)
    return hm


def monte_carlo(D, w_ahp: pd.Series, base_cc, base_tier, n_sim=1000, conc=50, seed=42):
    """Dirichlet-sample weights around w_AHP; report rank-correlation stability,
    tier-retention and top-k overlap (the standard MCDM stability battery)."""
    rng = np.random.default_rng(seed)
    base_s = pd.Series(base_cc)
    base_rank = base_s.rank(ascending=False)
    alpha = w_ahp.values.astype(float) * conc
    alpha = np.clip(alpha, 0.05, None)
    m = len(base_cc)
    rank_matrix = np.zeros((n_sim, m))
    top10_base = set(base_rank.sort_values().index[:10])
    top10_overlap, spearmans, tier_ret = [], [], []
    q1, q2 = np.quantile(base_cc, [1/3, 2/3])
    for s in range(n_sim):
        w = rng.dirichlet(alpha)
        cc = fuzzy_topsis(D, [TFN(x, x, x) for x in w])["CC"]
        r = pd.Series(cc).rank(ascending=False)
        rank_matrix[s] = r.values
        top10_overlap.append(len(top10_base & set(r.sort_values().index[:10])) / 10.0)
        spearmans.append(spearmanr(base_cc, cc)[0])
        new_tier = np.where(cc <= q1, "Low", np.where(cc <= q2, "Medium", "High"))
        tier_ret.append(np.mean(new_tier == base_tier))
    mean_rank_change = np.mean(np.abs(rank_matrix - base_rank.values[None, :]), axis=0)
    spearmans = np.array(spearmans)
    return {"mean_rank_change": mean_rank_change,
            "median_spearman": float(np.median(spearmans)),
            "pct_spearman_ge_090": float(np.mean(spearmans >= 0.90)),
            "mean_tier_retention": float(np.mean(tier_ret)),
            "mean_top10_overlap": float(np.mean(top10_overlap))}


# ---------------------------------------------------------------------------
# Fuzzy C-Means clustering + validity
# ---------------------------------------------------------------------------
def fcm_cluster(scores: pd.DataFrame, c_range=range(2, 7), seed=42):
    X = scores[C.CRIT_IDS].values
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    data = Xz.T
    results = {}
    for c in c_range:
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data, c, 2.0, error=1e-6, maxiter=1000, seed=seed)
        # Xie-Beni index
        um = u ** 2
        d2 = np.zeros((c, X.shape[0]))
        for k in range(c):
            d2[k] = np.sum((Xz - cntr[k]) ** 2, axis=1)
        num = np.sum(um * d2)
        min_center = np.min([np.sum((cntr[a] - cntr[b]) ** 2)
                             for a in range(c) for b in range(c) if a != b])
        xb = num / (X.shape[0] * min_center)
        results[c] = {"fpc": fpc, "xb": xb, "u": u, "centers": cntr}
    # choose c*: maximize fpc, minimize xb (rank sum)
    cs = list(results)
    fpc_rank = pd.Series({c: results[c]["fpc"] for c in cs}).rank(ascending=False)
    xb_rank = pd.Series({c: results[c]["xb"] for c in cs}).rank(ascending=True)
    score = fpc_rank + xb_rank
    c_star = int(score.idxmin())
    return c_star, results, Xz


# ---------------------------------------------------------------------------
# Prescriptive gap decomposition  (revised: peer-relative weighted deficit)
# ---------------------------------------------------------------------------
# v1 defined Gap_ij = w_j * d(r_ij (x) w_j, (1,1,1)). That formulation applied the
# weight TWICE (once inside the weighted value v_ij, once as the multiplier) and
# measured distance to the crisp ideal (1,1,1), which the constant 1 dominates
# because every weighted TFN component is <= max_j w_j. The resulting priority list
# was therefore driven largely by the global weight vector rather than by the firm's
# own capability profile. The revision below separates the two concerns:
#
#   deficit_ij  = (s_max - x_ij) / (s_max - s_min)        firm-specific shortfall in [0,1]
#   Gap_ij      = w_j * (deficit_ij - mean_i deficit_ij)  peer-relative, weight-scaled
#
# Subtracting the cohort-mean deficit per criterion removes the criterion-level scale
# confound, so a criterion is prioritised only where THIS firm lags its peers. The
# absolute variant is retained for comparison/reporting.
def gap_decomposition(scores: pd.DataFrame, w_ahp: pd.Series, top_k=3,
                      s_min: float = 1.0, s_max: float = 5.0, mode: str = "relative"):
    """Per-SME intervention priorities from a weighted capability deficit.

    scores : (m x n) construct scores on the native [s_min, s_max] Likert scale.
    mode   : "relative" (default, peer-centred) or "absolute".
    Returns (gap_df, top_list, extras) where extras carries both variants and the
    cohort deficit profile used for centring.
    """
    ids = [c for c in C.CRIT_IDS]
    X = scores[ids].values.astype(float)
    w = w_ahp.reindex(ids).values.astype(float)
    deficit = (s_max - X) / (s_max - s_min)          # in [0,1], higher = weaker
    peer = deficit.mean(axis=0)                      # cohort deficit per criterion
    gap_abs = w * deficit
    gap_rel = w * (deficit - peer)
    G = gap_rel if mode == "relative" else gap_abs
    gapdf = pd.DataFrame(G, columns=ids, index=scores.index)
    absdf = pd.DataFrame(gap_abs, columns=ids, index=scores.index)
    top = [gapdf.iloc[i].sort_values(ascending=False).index[:top_k].tolist()
           for i in range(len(gapdf))]
    extras = {"gap_absolute": absdf,
              "peer_deficit": pd.Series(peer, index=ids),
              "deficit": pd.DataFrame(deficit, columns=ids, index=scores.index)}
    return gapdf, top, extras


# ---------------------------------------------------------------------------
# Cost-aware ROI prioritisation
# ---------------------------------------------------------------------------
# Reviewers asked for implementation cost in the prioritisation. No cost variable
# exists in any of the three open datasets, so we expose an explicit, auditable
# relative-cost vector derived from the intervention TYPE each criterion implies,
# and rank by benefit-per-unit-cost. The vector is an assumption, not a measurement;
# roi_sensitivity() below quantifies how far conclusions move when it is perturbed.
#
# Cost tiers (relative units, 1 = cheapest):
#   process / policy change        1.0   (procedural, low capital)
#   training & HR practice         1.5   (recurring staff time)
#   software / integration         2.5   (licences + integration effort)
#   physical infrastructure        4.0   (capital expenditure)
COST_TIER = {"process": 1.0, "training": 1.5, "software": 2.5, "infrastructure": 4.0}

CRITERION_COST_TYPE = {
    "C1":  "software",        # data governance platform & repository
    "C2":  "software",        # device interoperability / middleware
    "C3":  "process",         # business-criticality alignment (strategy work)
    "C4":  "training",        # workforce training provision
    "C5":  "training",        # self-efficacy (coaching, enablement)
    "C6":  "software",        # visualization / interaction tooling
    "C7":  "process",         # adaptation to future demands (planning)
    "C8":  "process",         # process optimization
    "C9":  "infrastructure",  # fault tolerance of machinery
    "C10": "process",         # process capability
    "C11": "infrastructure",  # resource accessibility (capacity)
    "C12": "process",         # management maturity
    "C13": "process",         # governance & control
    "C14": "process",         # participatory job design
    "C15": "process",         # performance appraisal
    "C16": "training",        # compensation & recognition
}


def cost_vector(cost_tier: dict | None = None) -> pd.Series:
    """Relative implementation cost per criterion (see COST_TIER)."""
    tiers = cost_tier or COST_TIER
    return pd.Series({c: tiers[CRITERION_COST_TYPE[c]] for c in C.CRIT_IDS})


def roi_prioritisation(gapdf: pd.DataFrame, cost: pd.Series | None = None, top_k=3):
    """Rank interventions by weighted gap per unit implementation cost.

    ROI_ij = Gap_ij / cost_j. Returns (roi_df, top_list).
    """
    cost = cost_vector() if cost is None else cost.reindex(C.CRIT_IDS)
    roi = gapdf[C.CRIT_IDS].div(cost, axis=1)
    top = [roi.iloc[i].sort_values(ascending=False).index[:top_k].tolist()
           for i in range(len(roi))]
    return roi, top


def roi_sensitivity(gapdf: pd.DataFrame, n_sim: int = 1000, spread: float = 0.5,
                    seed: int = 42):
    """How stable is the ROI top-1 when the assumed cost vector is perturbed?

    Each criterion cost is multiplied by a lognormal factor with the given spread
    (so a 0.5 spread spans roughly a 2.7-fold cost error either way). Reports the
    share of SMEs whose top-1 ROI intervention is unchanged, averaged over draws.
    """
    rng = np.random.default_rng(seed)
    base_cost = cost_vector()
    base_top1 = roi_prioritisation(gapdf, base_cost, top_k=1)[1]
    base_top1 = np.array([t[0] for t in base_top1])
    retention, rank_rho = [], []
    base_order = (gapdf[C.CRIT_IDS].div(base_cost, axis=1)).mean(axis=0).rank(ascending=False)
    for _ in range(n_sim):
        c = base_cost * np.exp(rng.normal(0.0, spread, size=len(base_cost)))
        t1 = np.array([t[0] for t in roi_prioritisation(gapdf, c, top_k=1)[1]])
        retention.append(float(np.mean(t1 == base_top1)))
        order = (gapdf[C.CRIT_IDS].div(c, axis=1)).mean(axis=0).rank(ascending=False)
        rank_rho.append(spearmanr(base_order.values, order.values)[0])
    return {"mean_top1_retention": float(np.mean(retention)),
            "p05_top1_retention": float(np.percentile(retention, 5)),
            "median_cohort_rank_rho": float(np.median(rank_rho)),
            "spread": spread, "n_sim": n_sim}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for the elicited weights
# ---------------------------------------------------------------------------
def bootstrap_weights(d1_item_scores: pd.DataFrame, n_boot: int = 1000,
                      seed: int = 42, ci: float = 95.0):
    """Nonparametric bootstrap CI for each criterion weight.

    Resamples the D1 respondents with replacement, re-derives the whole importance ->
    pairwise -> Buckley weight chain on each draw, and reports percentile intervals.
    This quantifies how much of the weight vector is pinned down by 150 respondents
    rather than treating the point estimate as exact -- the elicitation uncertainty
    that a single reported weight hides.

    d1_item_scores : (respondents x criteria) numeric importance ratings.
    Returns DataFrame indexed by criterion with mean/lo/hi/sd and rank stability.
    """
    rng = np.random.default_rng(seed)
    ids = [c for c in C.CRIT_IDS if c in d1_item_scores.columns]
    X = d1_item_scores[ids].astype(float).values
    m = X.shape[0]
    lo_q, hi_q = (100 - ci) / 2.0, 100 - (100 - ci) / 2.0
    draws, ranks = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        imp = pd.Series(np.nanmean(X[idx], axis=0), index=ids)
        try:
            w = derive_weights(imp.reindex(C.CRIT_IDS))["w_buckley"]
        except Exception:
            continue
        draws.append(w.values)
        ranks.append(w.rank(ascending=False).values)
    Wb = np.vstack(draws)
    Rb = np.vstack(ranks)
    base = derive_weights(pd.Series(np.nanmean(X, axis=0), index=ids)
                          .reindex(C.CRIT_IDS))["w_buckley"]
    out = pd.DataFrame({
        "w": base.values,
        "boot_mean": Wb.mean(axis=0),
        "boot_sd": Wb.std(axis=0),
        "ci_lo": np.percentile(Wb, lo_q, axis=0),
        "ci_hi": np.percentile(Wb, hi_q, axis=0),
        "rank_mode": pd.DataFrame(Rb).mode(axis=0).iloc[0].values,
        "rank_sd": Rb.std(axis=0),
    }, index=C.CRIT_IDS)
    out["ci_width"] = out["ci_hi"] - out["ci_lo"]
    return out, {"n_boot": int(Wb.shape[0]), "ci": ci,
                 "mean_ci_width": float(out["ci_width"].mean()),
                 "mean_rank_sd": float(out["rank_sd"].mean())}
