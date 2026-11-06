"""End-to-end pipeline: data -> psychometrics -> Fuzzy AHP -> Fuzzy TOPSIS ->
ML+SHAP -> weight agreement -> sensitivity -> FCM clustering -> gap decomposition.

Saves every artifact needed by figures.py / tables.py into data/processed and reports/.
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dtreadiness import config as C
from dtreadiness import data as D
from dtreadiness import psychometrics as P
from dtreadiness import ml as ML
from dtreadiness import analysis as A
from dtreadiness import validity as V
from dtreadiness import hybrid_weights as HW
from dtreadiness.fuzzy import fuzzy_topsis, fuzzy_vikor, TFN
from scipy.stats import spearmanr, kendalltau

np.random.seed(C.SEED)
os.makedirs(C.DATA_PROC, exist_ok=True)
PROC = C.DATA_PROC
RES = {}   # scalar results dict -> results.json


def save_csv(df, name):
    df.to_csv(os.path.join(PROC, name))
    print(f"  saved {name}  shape={getattr(df,'shape',None)}")


print("[1/9] Loading data & construct scores ...")
scores = D.d2_construct_scores()                 # 321 x 12 (+demog)
item_frame = D.d2_item_frame()
importance = D.d1_construct_importance()          # C1..C12 expert-proxy importance
save_csv(scores, "sme_construct_scores.csv")
importance.to_csv(os.path.join(PROC, "d1_importance.csv"))
RES["n_sme"] = int(scores.shape[0])
RES["n_dt_respondents"] = 150
RES["n_criteria"] = len(C.CRIT_IDS)
RES["combined_n"] = 150 + int(scores.shape[0])

print("[2/11] Psychometrics, discriminant validity & common-method bias ...")
item_map = D.construct_item_map()
rel = P.reliability_table(item_map, item_frame)
kmo, chi2, pval = P.kmo_bartlett(item_frame)
save_csv(rel.set_index("Criterion"), "reliability.csv")
RES["KMO"] = kmo; RES["Bartlett_chi2"] = chi2; RES["Bartlett_p"] = pval
RES["alpha_mean"] = float(rel["Cronbach_alpha"].mean())
RES["alpha_min"] = float(rel["Cronbach_alpha"].min())
RES["n_alpha_ge_070"] = int((rel["Cronbach_alpha"] >= 0.70).sum())
RES["CR_min"] = float(rel["CR"].min())
RES["AVE_min"] = float(rel["AVE"].min())
RES["n_AVE_ge_050"] = int((rel["AVE"] >= 0.50).sum())

# --- discriminant validity (HTMT) ---
H, htsum = V.htmt(item_frame, item_map)
save_csv(H, "htmt_matrix.csv")
RES["HTMT_max"] = htsum["max"]; RES["HTMT_mean"] = htsum["mean"]
RES["HTMT_violations"] = htsum["n_violations"]; RES["HTMT_passes"] = htsum["passes"]

# --- independent firm-performance outcome (never used to build a criterion) ---
outcome = D.d2_outcome_score()
outcome.to_csv(os.path.join(PROC, "outcome_performance.csv"))

# --- common-method bias: Harman + Kock VIF + marker variable ---
cmb_items = pd.concat([item_frame, D.load_d2()[[c for p in C.OUTCOME_D2
                       for c in D._group_cols(D.load_d2(), p)]].apply(pd.to_numeric,
                       errors="coerce")], axis=1)
harman = V.harman_single_factor(cmb_items)
vifres = V.full_collinearity_vif(scores, C.CRIT_IDS)
marker = V.marker_variable_test(scores, outcome, C.CRIT_IDS)
save_csv(vifres["vif"].rename("VIF").to_frame(), "cmb_vif.csv")
save_csv(marker["by_marker"].rename("partialled_rho").to_frame(), "cmb_marker.csv")
RES["CMB"] = {
    "harman_first_factor_pct": harman["first_factor_pct"],
    "harman_n_eigen_gt1": harman["n_eigen_gt1"],
    "harman_passes": harman["passes"],
    "vif_max": vifres["max_vif"], "vif_argmax": vifres["argmax"],
    "vif_passes": vifres["passes"],
    "marker_raw_rho": marker["raw_rho"],
    "marker_min": marker["min_partialled"], "marker_max": marker["max_partialled"],
    "marker_median": marker["median_partialled"], "marker_passes": marker["passes"],
}

print("[3/9] Fuzzy AHP weights (Chang extent + Buckley) ...")
wres = A.derive_weights(importance)
# Operational weights: Buckley geometric-mean (robust; Chang extent can zero a
# criterion -- see C4 -- a documented limitation). Chang reported for comparison (T5).
w_ahp = wres["w_buckley"]
wtab = pd.DataFrame({
    "importance": importance.reindex(C.CRIT_IDS),
    "w_chang": wres["w_chang"], "w_buckley": wres["w_buckley"],
    "buckley_l": [t.l for t in wres["w_buckley_tfn"]],
    "buckley_m": [t.m for t in wres["w_buckley_tfn"]],
    "buckley_u": [t.u for t in wres["w_buckley_tfn"]],
})
# Rank on the OPERATIONAL weights (Buckley). Chang extent analysis is reported for
# comparison only: it degenerates here, assigning exactly zero to C4/C5 and tying
# several criteria, the known failure mode documented by Wang et al. (2008).
wtab["rank"] = wtab["w_buckley"].rank(ascending=False).astype(int)
RES["chang_n_zero"] = int((wtab["w_chang"] <= 1e-12).sum())
RES["chang_n_distinct"] = int(wtab["w_chang"].nunique())
save_csv(wtab, "ahp_weights.csv")
RES["AHP_CR"] = float(wres["CR"])
RES["chang_buckley_rho"] = float(spearmanr(wres["w_chang"], wres["w_buckley"])[0])

# --- Best-Worst Method: independent second elicitation method ---
from dtreadiness.fuzzy import bwm_weights
w_bwm_arr, bwm_cr = bwm_weights(importance.reindex(C.CRIT_IDS).values)
w_bwm = pd.Series(w_bwm_arr, index=C.CRIT_IDS)
wtab["w_bwm"] = w_bwm
RES["BWM_CR"] = float(bwm_cr)
RES["AHP_BWM_spearman"] = float(spearmanr(w_ahp, w_bwm)[0])
RES["AHP_BWM_kendall"] = float(kendalltau(w_ahp, w_bwm)[0])

# --- bootstrap CIs over D1 respondents: elicitation uncertainty ---
d1_resp = D.d1_respondent_importance()
boot, bootsum = A.bootstrap_weights(d1_resp, n_boot=1000, seed=C.SEED)
save_csv(boot, "weight_bootstrap.csv")
RES["boot_n"] = bootsum["n_boot"]
RES["boot_mean_ci_width"] = bootsum["mean_ci_width"]
RES["boot_mean_rank_sd"] = bootsum["mean_rank_sd"]
save_csv(wtab, "ahp_weights.csv")

print("[4/9] Fuzzy TOPSIS ranking ...")
Dfz = A.fuzzify_scores(scores)
w_tfn = [TFN(x, x, x) for x in w_ahp.values]
topsis = fuzzy_topsis(Dfz, w_tfn)
CC = topsis["CC"]
scores_out = scores.copy()
scores_out["CC"] = CC
scores_out["rank"] = pd.Series(CC, index=scores.index).rank(ascending=False).astype(int)
# tercile tiers
q1, q2 = np.quantile(CC, [1/3, 2/3])
tier = np.where(CC <= q1, "Low", np.where(CC <= q2, "Medium", "High"))
scores_out["tier"] = tier
# min-max normalized readiness index (0-100) for interpretability
RI_idx = 100 * (CC - CC.min()) / (CC.max() - CC.min())
scores_out["readiness_index"] = RI_idx
save_csv(scores_out, "topsis_scores.csv")
RES["CC_mean"] = float(np.mean(CC)); RES["CC_std"] = float(np.std(CC))
RES["CC_min"] = float(np.min(CC)); RES["CC_max"] = float(np.max(CC))
RES["tier_counts"] = {t: int((tier == t).sum()) for t in C.TIERS}

# cross-method VIKOR
Q = fuzzy_vikor(Dfz, w_tfn)
tau_tv = kendalltau(pd.Series(CC).rank(ascending=False),
                    pd.Series(-Q).rank(ascending=False))[0]
RES["topsis_vikor_kendall"] = float(tau_tv)

# baselines: equal-weight TOPSIS, crisp (defuzz) TOPSIS
nC = len(C.CRIT_IDS)
w_eq = [TFN(1/nC, 1/nC, 1/nC)] * nC
CC_eq = fuzzy_topsis(Dfz, w_eq)["CC"]
RES["equalweight_kendall"] = float(kendalltau(
    pd.Series(CC).rank(ascending=False), pd.Series(CC_eq).rank(ascending=False))[0])

print("[5/9] ML validation + SHAP ...")
X = scores[C.CRIT_IDS].copy()
y = pd.Categorical(tier, categories=C.TIERS, ordered=True).codes
ml_metrics, oof = ML.evaluate_cv(X, y, n_splits=5)
mldf = pd.DataFrame(ml_metrics).T
save_csv(mldf, "ml_metrics.csv")
best_model = mldf["macroF1"].idxmax()
RES["best_model"] = best_model
RES["best_model_acc"] = float(mldf.loc[best_model, "Accuracy"])
RES["best_model_f1"] = float(mldf.loc[best_model, "macroF1"])
RES["best_model_auc"] = float(mldf.loc[best_model, "AUC"])
RES["best_model_kappa"] = float(mldf.loc[best_model, "Kappa"])

# SHAP on the best-performing tree ensemble (LightGBM); TreeExplainer requires a tree model.
shap_model = "LightGBM"
RES["shap_model"] = shap_model
RES["shap_model_f1"] = float(mldf.loc[shap_model, "macroF1"])
w_shap, fitted, explainer, shap_vals = ML.shap_importance(X, y, shap_model)
w_shap = w_shap.reindex(C.CRIT_IDS)
w_shap.to_csv(os.path.join(PROC, "shap_importance.csv"))
np.save(os.path.join(PROC, "shap_values.npy"), np.array(shap_vals, dtype=object), allow_pickle=True)
X.to_csv(os.path.join(PROC, "ml_X.csv"), index=False)

print("[6/9] Weight agreement (AHP vs SHAP) ...")
agree = A.weight_agreement(w_ahp, w_shap)
agdf = pd.DataFrame({"w_AHP": w_ahp, "w_SHAP": w_shap, "delta": agree["delta"]})
save_csv(agdf, "weight_agreement.csv")
RES["AHP_SHAP_spearman"] = float(agree["spearman"])
RES["AHP_SHAP_spearman_p"] = float(agree["spearman_p"])
RES["AHP_SHAP_kendall"] = float(agree["kendall"])

print("[7/9] Sensitivity (OAT + Monte-Carlo) ...")
hm = A.oat_sensitivity(Dfz, w_ahp, CC)
save_csv(hm, "sensitivity_oat.csv")
mc = A.monte_carlo(Dfz, w_ahp, CC, tier, n_sim=1000, conc=50, seed=C.SEED)
RES["mc_median_spearman"] = mc["median_spearman"]
RES["mc_pct_spearman_ge_090"] = mc["pct_spearman_ge_090"]
RES["mc_mean_tier_retention"] = mc["mean_tier_retention"]
RES["mc_mean_top10_overlap"] = mc["mean_top10_overlap"]
RES["mc_mean_rank_change"] = float(np.mean(mc["mean_rank_change"]))
np.save(os.path.join(PROC, "mc_mean_rank_change.npy"), mc["mean_rank_change"])

print("[8/9] FCM clustering ...")
c_star, clus, Xz = A.fcm_cluster(scores, seed=C.SEED)
u = clus[c_star]["u"]; centers = clus[c_star]["centers"]
labels = np.argmax(u, axis=0)
scores_out["cluster"] = labels
save_csv(scores_out[["CC", "rank", "tier", "cluster"]], "sme_final.csv")
np.save(os.path.join(PROC, "fcm_centers.npy"), centers)
np.save(os.path.join(PROC, "fcm_u.npy"), u)
validity = pd.DataFrame({c: {"fpc": clus[c]["fpc"], "xb": clus[c]["xb"]} for c in clus}).T
save_csv(validity, "fcm_validity.csv")
RES["c_star"] = int(c_star)
RES["fpc_cstar"] = float(clus[c_star]["fpc"])
RES["xb_cstar"] = float(clus[c_star]["xb"])
RES["cluster_sizes"] = {int(k): int((labels == k).sum()) for k in range(c_star)}

# --- is the partition real, or a stratification of one gradient? ---
cstruct = V.cluster_structure_test(scores, C.CRIT_IDS, c_range=range(2, 7),
                                   n_null=200, seed=C.SEED)
RES["cluster_structure"] = cstruct
print(f"    cluster structure: {cstruct['verdict']} (PC1 share "
      f"{cstruct['pc1_variance_share']:.3f})")
# cluster centroid profile (destandardized to 1..5)
mu = scores[C.CRIT_IDS].mean().values; sd = scores[C.CRIT_IDS].std().values
cent_real = pd.DataFrame(centers * sd + mu, columns=C.CRIT_IDS)
cent_real.index = [f"Cluster{k+1}" for k in range(c_star)]
save_csv(cent_real, "cluster_centroids.csv")

print("[9/11] Prescriptive gap decomposition (peer-relative weighted deficit) ...")
gapdf, top3, gx = A.gap_decomposition(scores, w_ahp, top_k=3, mode="relative")
gap_abs = gx["gap_absolute"]
_, top3_abs, _ = A.gap_decomposition(scores, w_ahp, top_k=3, mode="absolute")
save_csv(gapdf, "gap_relative.csv")
save_csv(gap_abs, "gap_absolute.csv")
save_csv(gx["peer_deficit"].rename("peer_deficit").to_frame(), "peer_deficit.csv")

# --- cost-aware ROI prioritisation ---
cost = A.cost_vector()
roi, top3_roi = A.roi_prioritisation(gapdf, cost, top_k=3)
save_csv(roi, "roi_scores.csv")
save_csv(cost.rename("rel_cost").to_frame(), "cost_vector.csv")
roi_sens = A.roi_sensitivity(gapdf, n_sim=500, spread=0.5, seed=C.SEED)
RES["ROI_sensitivity"] = roi_sens

presc = pd.DataFrame({"SME": scores.index, "CC": CC, "tier": tier,
                      "cluster": labels,
                      "top1": [t[0] for t in top3],
                      "top2": [t[1] for t in top3],
                      "top3": [t[2] for t in top3],
                      "top1_abs": [t[0] for t in top3_abs],
                      "top1_roi": [t[0] for t in top3_roi],
                      "top2_roi": [t[1] for t in top3_roi],
                      "top3_roi": [t[2] for t in top3_roi]})
save_csv(presc.set_index("SME"), "prescriptions.csv")

# how firm-specific is the roadmap? (reviewer concern: does it just echo the weights?)
wb = np.tile(w_ahp.reindex(C.CRIT_IDS).values, (len(gapdf), 1))
RES["gap_diagnostics"] = {
    "relative_corr_with_weight": float(np.corrcoef(gapdf.values.ravel(), wb.ravel())[0, 1]),
    "absolute_corr_with_weight": float(np.corrcoef(gap_abs.values.ravel(), wb.ravel())[0, 1]),
    "relative_n_distinct_top1": int(pd.Series([t[0] for t in top3]).nunique()),
    "absolute_n_distinct_top1": int(pd.Series([t[0] for t in top3_abs]).nunique()),
    "relative_max_top1_share": float(pd.Series([t[0] for t in top3]).value_counts(normalize=True).iloc[0]),
    "absolute_max_top1_share": float(pd.Series([t[0] for t in top3_abs]).value_counts(normalize=True).iloc[0]),
    "roi_n_distinct_top1": int(pd.Series([t[0] for t in top3_roi]).nunique()),
    "roi_max_top1_share": float(pd.Series([t[0] for t in top3_roi]).value_counts(normalize=True).iloc[0]),
}
RES["top1_counts"] = pd.Series([t[0] for t in top3]).value_counts().to_dict()
RES["top1_roi_counts"] = pd.Series([t[0] for t in top3_roi]).value_counts().to_dict()

# most common leverage criterion per cluster
clus_gap, clus_gap_roi = {}, {}
for k in range(c_star):
    idx = set(np.where(labels == k)[0])
    tc = pd.Series([t[0] for i, t in enumerate(top3) if i in idx]).value_counts()
    tr = pd.Series([t[0] for i, t in enumerate(top3_roi) if i in idx]).value_counts()
    clus_gap[int(k)] = tc.index[0] if len(tc) else None
    clus_gap_roi[int(k)] = tr.index[0] if len(tr) else None
RES["cluster_top_gap"] = clus_gap
RES["cluster_top_gap_roi"] = clus_gap_roi
RES["mean_gap_per_criterion"] = gapdf.mean().to_dict()

print("[10/11] Outcome-prediction task (non-circular ML) ...")
ml_out, n_out = ML.evaluate_outcome_task(scores, outcome, C.CRIT_IDS)
save_csv(ml_out, "ml_outcome_metrics.csv")
RES["ml_outcome"] = {
    "n": n_out,
    "best_model": str(ml_out["AUC"].idxmax()),
    "best_auc": float(ml_out["AUC"].max()),
    "best_auc_sd": float(ml_out.loc[ml_out["AUC"].idxmax(), "AUC_sd"]),
    "best_f1": float(ml_out.loc[ml_out["AUC"].idxmax(), "macroF1"]),
    "best_acc": float(ml_out.loc[ml_out["AUC"].idxmax(), "Accuracy"]),
}

print("[10b/11] Hybrid (elicited + outcome-learned) weighting, nested CV ...")
hyb = HW.nested_cv_benchmark(scores, outcome, C.CRIT_IDS, w_ahp, seed=C.SEED)
save_csv(pd.DataFrame({"w_hybrid": hyb["w_hybrid"], "w_learned": hyb["w_learned"],
                       "w_elicited": w_ahp.reindex(C.CRIT_IDS)}), "hybrid_weights.csv")
RES["hybrid_weighting"] = {k: v for k, v in hyb.items()
                           if k not in ("w_hybrid", "w_learned")}

print("[11/11] Weight-scheme benchmark against the independent outcome ...")
wbench = V.weight_scheme_benchmark(scores, outcome, C.CRIT_IDS, w_ahp,
                                   n_null=2000, seed=C.SEED)
wbench.pop("by_marker", None)
RES["weight_benchmark"] = wbench

print("[11b/11] Criterion validity of the ranked readiness score ...")
from scipy.stats import spearmanr as _sp, kendalltau as _kt
_idx = scores.index.intersection(outcome.dropna().index)
RES["criterion_validity_ranked"] = {
    "spearman_CC_vs_performance": float(_sp(pd.Series(CC, index=scores.index).loc[_idx],
                                            outcome.loc[_idx])[0]),
    "n": int(len(_idx)),
}

print("[11c] Reviewer-requested baselines: unweighted mean + equal-weight ...")
# R4: report the unweighted arithmetic-mean baseline explicitly, alongside
# equal-weight TOPSIS, so the elicited/learned weighting is judged against both
# naive schemes rather than only against each other.
_um_full = scores[C.CRIT_IDS].mean(axis=1)                       # unweighted mean of raw scores
_ccser = pd.Series(CC, index=scores.index)
_Zb = (scores.loc[_idx, C.CRIT_IDS] - scores.loc[_idx, C.CRIT_IDS].mean()) \
      / scores.loc[_idx, C.CRIT_IDS].std()
_umz = _Zb.mean(axis=1)
RES["baselines"] = {
    "unweighted_mean_vs_topsis_spearman": float(_sp(_um_full, _ccser)[0]),
    "unweighted_mean_vs_topsis_kendall": float(_kt(_um_full.rank(ascending=False),
                                                   _ccser.rank(ascending=False))[0]),
    "unweighted_mean_vs_perf_spearman": float(_sp(_umz, outcome.loc[_idx])[0]),
    "equalweight_topsis_kendall": float(RES.get("equalweight_kendall", float("nan"))),
    "equalweight_vs_perf_spearman": float(wbench["rho_equal"]),
    "elicited_vs_perf_spearman": float(wbench["rho_elicited"]),
    "fitted_oos_vs_perf_spearman": float(wbench["rho_fitted_oos"]),
}

print("[11d] Sustainability / resource-efficiency metric ...")
# R4: operationalise the Industry-5.0 sustainability pillar. We quantify the
# resource efficiency of cost-aware over cost-blind prioritisation as the gap
# (readiness improvement potential) closed per unit of implementation cost.
# For each SME, compare the top-3 interventions chosen by (a) weighted gap alone
# vs (b) cost-aware ROI: the ROI set closes a similar gap at lower total cost,
# the avoided cost being the stranded-capital (waste) that is averted.
try:
    import numpy as _np
    cost_v = cost.reindex(C.CRIT_IDS)
    gaprel = gapdf[C.CRIT_IDS]                       # peer-relative weighted gap
    roidf = roi[C.CRIT_IDS]
    gap_top = gaprel.apply(lambda r: r.sort_values(ascending=False).index[:3], axis=1)
    roi_top = roidf.apply(lambda r: r.sort_values(ascending=False).index[:3], axis=1)
    # total cost of each firm's top-3 set under each policy
    gap_cost = gap_top.apply(lambda ix: float(cost_v[list(ix)].sum()))
    roi_cost = roi_top.apply(lambda ix: float(cost_v[list(ix)].sum()))
    # readiness gap addressed (sum of |gap| over the chosen criteria)
    gap_addr = pd.Series({i: float(gaprel.loc[i, list(gap_top[i])].abs().sum())
                          for i in gaprel.index})
    roi_addr = pd.Series({i: float(gaprel.loc[i, list(roi_top[i])].abs().sum())
                          for i in gaprel.index})
    RES["sustainability"] = {
        "mean_cost_gap_policy": float(gap_cost.mean()),
        "mean_cost_roi_policy": float(roi_cost.mean()),
        "mean_cost_saving_pct": float(100 * (gap_cost.mean() - roi_cost.mean())
                                      / gap_cost.mean()),
        "roi_gap_addressed_ratio": float(roi_addr.mean() / gap_addr.mean()),
        "note": ("ROI policy closes " + f"{100*roi_addr.mean()/gap_addr.mean():.0f}"
                 + "% of the readiness gap the weighted-gap policy closes, at "
                 + f"{100*(gap_cost.mean()-roi_cost.mean())/gap_cost.mean():.0f}"
                 + "% lower implementation cost; the averted cost is stranded "
                 + "capital (resource waste) avoided."),
    }
except Exception as _e:
    RES["sustainability"] = {"error": str(_e)}

with open(os.path.join(PROC, "results.json"), "w") as f:
    json.dump(RES, f, indent=2, default=float)

print("\n===== KEY RESULTS =====")
for k in ["n_sme", "combined_n", "KMO", "Bartlett_p", "alpha_mean", "n_alpha_ge_070",
          "AHP_CR", "CC_mean", "tier_counts", "best_model", "best_model_acc",
          "best_model_f1", "best_model_auc", "AHP_SHAP_spearman", "AHP_SHAP_kendall",
          "mc_median_spearman", "mc_pct_spearman_ge_090", "mc_mean_tier_retention",
          "mc_mean_top10_overlap", "topsis_vikor_kendall",
          "c_star", "fpc_cstar", "cluster_sizes"]:
    print(f"  {k}: {RES.get(k)}")
print("\nTop-5 AHP weights:\n", wtab.sort_values('w_chang', ascending=False)[['w_chang','rank']].head())
print("\nDone. Artifacts in", PROC)
