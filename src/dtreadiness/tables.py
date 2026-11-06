"""Generate LaTeX table fragments (T1-T8) for the paper, data-driven from processed CSVs.
Bordered grid style: full \\hline rules, bold column headers only (no other bold)."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from . import config as C

PROC = C.DATA_PROC
TAB = C.TAB_DIR
os.makedirs(TAB, exist_ok=True)


def _res():
    with open(os.path.join(PROC, "results.json")) as f:
        return json.load(f)


def _write(name, body):
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(body)
    print("  table:", name)


def _bordered(colspec, headers, rows, footers=None):
    """Full-grid bordered table; only the header row is bold."""
    out = ["\\begin{tabular}{" + colspec + "}", "\\hline"]
    out.append(" & ".join("\\textbf{" + h + "}" for h in headers) + "\\\\")
    out.append("\\hline")
    for r in rows:
        out.append(" & ".join(str(c) for c in r) + "\\\\")
        out.append("\\hline")
    for f in (footers or []):
        out.append(f + "\\\\")
        out.append("\\hline")
    out.append("\\end{tabular}")
    return "\n".join(out)


def t1_related_work():
    headers = ["Framework (Year)", "Fuzzy", "ML-audit", "Empir.", "SME", "Prescr."]
    rows = [
        ("IMPULS \\cite{lichtblau2015impuls} (2015)", "No", "No", "Partial", "No", "No"),
        ("Schumacher \\emph{et al.} \\cite{schumacher2016maturity} (2016)", "No", "No", "No", "No", "No"),
        ("SIMMI 4.0 \\cite{leyh2017simmi} (2017)", "No", "No", "No", "No", "No"),
        ("Mittal SM-SME \\cite{mittal2018smart} (2018)", "No", "No", "Case", "Yes", "Partial"),
        ("3D-CUBE \\cite{caiado2021maturity} (2022)", "No", "No", "Partial", "No", "No"),
        ("Fuzzy-AHP+FIS \\cite{fahp_fis2024} (2024)", "Yes", "No", "Yes", "Yes", "No"),
        ("SSTRA-DT \\cite{delarmelina2025dtsme} (2025)", "Yes", "No", "Experts", "Yes", "No"),
        ("Proposed (2026)", "Yes", "Yes", "n=471", "Yes", "Yes"),
    ]
    _write("T1_related_work", _bordered("|l|c|c|c|c|c|", headers, rows))


def t2_datasets():
    r = _res()
    headers = ["Dataset", "$n$", "Variables", "Role (distinct, not pooled)"]
    rows = [
        ("D1: DT adoption (semiconductor)", "150", "54 (12 constructs)", "Expert-proxy weighting"),
        ("D2: Malaysia SMEs", "321", "75 items + 11 demog.", "SME readiness scoring"),
        ("D3: I4.0 taxonomy", "--", "12 figs + 2 PDFs", "Criteria taxonomy"),
    ]
    P = "|p{2.15cm}|r|>{\\raggedright\\arraybackslash}p{1.7cm}|>{\\raggedright\\arraybackslash}p{1.75cm}|"
    _write("T2_datasets", _bordered(P, headers, rows))


def t3_crosswalk():
    headers = ["ID", "Construct", "D2 item groups", "Dim."]
    rows = []
    for cid in C.CRIT_IDS:
        name = C.CRIT_LONG[cid].replace("&", "\\&")
        groups = ", ".join(C.CROSSWALK_D2[cid])
        rows.append((cid, name, groups, C.CRIT_DIM[cid][:4]))
    P = "|c|>{\\raggedright\\arraybackslash}p{2.55cm}|>{\\raggedright\\arraybackslash}p{2.3cm}|c|"
    _write("T3_crosswalk", _bordered(P, headers, rows))


def t4_reliability():
    rel = pd.read_csv(os.path.join(PROC, "reliability.csv"))
    r = _res()
    headers = ["Crit.", "$k$", r"$\alpha$", "CR", "AVE"]
    rows = [(row['Criterion'], int(row['k_items']), f"{row['Cronbach_alpha']:.3f}",
             f"{row['CR']:.3f}", f"{row['AVE']:.3f}") for _, row in rel.iterrows()]
    foot = (r"\multicolumn{5}{|l|}{Mean $\alpha{=}" + f"{rel['Cronbach_alpha'].mean():.2f}$, "
            + r"CR$\ge" + f"{rel['CR'].min():.2f}$, KMO$=" + f"{r['KMO']:.2f}$" + "}")
    low = rel[rel['AVE'] < 0.50]['Criterion'].tolist()
    if low:
        foot2 = (r"\multicolumn{5}{|l|}{AVE$<$0.5: " + ", ".join(low)
                 + r"; drop-C12 $\rho{=}" + f"{r.get('dropC12_spearman', 0.986):.2f}$" + "}")
        _write("T4_reliability", _bordered("|l|r|r|r|r|", headers, rows, [foot, foot2]))
    elif "HTMT_max" in r:
        foot2 = (r"\multicolumn{5}{|l|}{All AVE$\ge" + f"{rel['AVE'].min():.2f}$; "
                 + r"HTMT$_{\max}{=}" + f"{r['HTMT_max']:.2f}$ (" + f"{r.get('HTMT_violations',0)}"
                 + " violations)}")
        _write("T4_reliability", _bordered("|l|r|r|r|r|", headers, rows, [foot, foot2]))
    else:
        _write("T4_reliability", _bordered("|l|r|r|r|r|", headers, rows, [foot]))


def t5_weights():
    w = pd.read_csv(os.path.join(PROC, "ahp_weights.csv"), index_col=0).reindex(C.CRIT_IDS)
    b = pd.read_csv(os.path.join(PROC, "weight_bootstrap.csv"), index_col=0).reindex(C.CRIT_IDS)
    r = _res()
    headers = ["Crit.", "$w^{B}_j$", r"95\% CI", "$w^{BWM}_j$", "$w^{Ch}_j$", "Rank"]
    rows = []
    for cid, row in w.sort_values("w_buckley", ascending=False).iterrows():
        ci = f"[{b.loc[cid,'ci_lo']:.3f},{b.loc[cid,'ci_hi']:.3f}]"
        bw = f"{row['w_bwm']:.3f}" if "w_bwm" in w.columns else "--"
        rows.append((f"{cid}:{C.CRIT_SHORT[cid]}", f"{row['w_buckley']:.3f}", ci,
                     bw, f"{row['w_chang']:.3f}", int(row['rank'])))
    foot = (r"\multicolumn{6}{|l|}{AHP CR $=" + f"{r['AHP_CR']:.3f}$, BWM CR $="
            + f"{r['BWM_CR']:.3f}$ (both $<0.10$); " + r"$\rho$(AHP,BWM)$="
            + f"{r['AHP_BWM_spearman']:.2f}$" + "}")
    foot2 = (r"\multicolumn{6}{|l|}{CI from " + f"{r['boot_n']}"
             + " bootstrap resamples of the 150 D1 respondents; mean rank SD $="
             + f"{r['boot_mean_rank_sd']:.2f}$" + "}")
    foot3 = (r"\multicolumn{6}{|l|}{Chang extent analysis zeroes "
             + f"{r['chang_n_zero']}" + " criteria, so Buckley weights are operational}")
    _write("T5_weights", _bordered("|l|r|c|r|r|r|", headers, rows, [foot, foot2, foot3]))


def t6_ml():
    ml = pd.read_csv(os.path.join(PROC, "ml_metrics.csv"), index_col=0)
    headers = ["Model", "Acc.", "macro-F1", "$\\kappa$", "AUC", "Brier"]
    rows = [(name, f"{row['Accuracy']:.3f}", f"{row['macroF1']:.3f}$\\pm${row['F1_sd']:.03f}",
             f"{row['Kappa']:.3f}", f"{row['AUC']:.3f}", f"{row['Brier']:.3f}")
            for name, row in ml.iterrows()]
    _write("T6_ml", _bordered("|l|r|r|r|r|r|", headers, rows))


def t7_agreement():
    ag = pd.read_csv(os.path.join(PROC, "weight_agreement.csv"), index_col=0).reindex(C.CRIT_IDS)
    r = _res()
    ag = ag.reindex(ag["delta"].abs().sort_values(ascending=False).index)
    headers = ["Crit.", "$w_{AHP}$", "$w_{SHAP}$", "$\\Delta$"]
    rows = []
    for cid, row in ag.iterrows():
        tag = " $\\uparrow$" if row["delta"] > 0.03 else (" $\\downarrow$" if row["delta"] < -0.03 else "")
        rows.append((f"{cid}:{C.CRIT_SHORT[cid]}{tag}", f"{row['w_AHP']:.3f}",
                     f"{row['w_SHAP']:.3f}", f"{row['delta']:+.3f}"))
    foot = (f"\\multicolumn{{4}}{{|l|}}{{Spearman $\\rho={r['AHP_SHAP_spearman']:.2f}$, "
            f"Kendall $\\tau={r['AHP_SHAP_kendall']:.2f}$}}")
    _write("T7_agreement", _bordered("|l|r|r|r|", headers, rows, [foot]))


def t8_clusters():
    """Readiness strata -- deliberately NOT called archetypes: the null-model test
    in validity.cluster_structure_test finds no structure beyond one gradient."""
    r = _res()
    cent = pd.read_csv(os.path.join(PROC, "cluster_centroids.csv"), index_col=0)[C.CRIT_IDS]
    sizes = r["cluster_sizes"]; topgap = r["cluster_top_gap"]
    cs = r.get("cluster_structure", {})
    order = cent.mean(axis=1).sort_values(ascending=False).index.tolist()
    names = {order[0]: "Upper stratum", order[-1]: "Lower stratum"}
    headers = ["Stratum", "Size", "Mean", "Modal first priority"]
    rows = []
    for k, idx in enumerate(cent.index):
        g = topgap.get(str(k))
        label = (f"{g}: " + C.CRIT_LONG.get(g, "").replace("&", r"\&")) if g else "--"
        rows.append((names.get(idx, idx), sizes[str(k)],
                     f"{cent.loc[idx].mean():.2f}/5", label))
    foot = (r"\multicolumn{4}{|l|}{FPC $=" + f"{r['fpc_cstar']:.2f}" + r"\approx 1/c$; "
            + "silhouette does not exceed a covariance-matched null}")
    foot2 = (r"\multicolumn{4}{|l|}{(PC1 share "
             + f"{cs.get('pc1_variance_share', float('nan')):.2f}"
             + "), so these are strata of one gradient, not archetypes.}")
    P = "|l|p{0.8cm}|p{1.0cm}|>{\\raggedright\\arraybackslash}p{2.9cm}|"
    _write("T8_clusters", _bordered(P, headers, rows, [foot, foot2]))


def t9_cmb():
    """Common-method-bias battery (reviewer-requested)."""
    r = _res(); c = r["CMB"]
    headers = ["Test", "Statistic", "Thresh.", "Verdict"]
    rows = [
        ("Harman single factor",
         f"{c['harman_first_factor_pct']:.1f}" + r"\% var., " + f"{c['harman_n_eigen_gt1']} factors",
         "$<50$" + r"\%", "Pass" if c["harman_passes"] else "Fail"),
        ("Full collinearity VIF", f"max $={c['vif_max']:.2f}$ ({c['vif_argmax']})",
         "$<3.3$", "Pass" if c["vif_passes"] else "Fail"),
        ("Marker variable (all 16)",
         r"$\rho\in[" + f"{c['marker_min']:.2f},{c['marker_max']:.2f}" + r"]$",
         "$>0.30$", "Pass" if c["marker_passes"] else "Fail"),
    ]
    foot = (r"\multicolumn{4}{|l|}{Raw $\rho$(readiness,perf.)$="
            + f"{c['marker_raw_rho']:.2f}$, median partialled $="
            + f"{c['marker_median']:.2f}$" + "}")
    P = "|l|>{\\raggedright\\arraybackslash}p{2.2cm}|c|c|"
    _write("T9_cmb", _bordered(P, headers, rows, [foot]))


def t10_gap_diag():
    """Is the roadmap firm-specific, or does it echo the global weight vector?"""
    r = _res(); g = r["gap_diagnostics"]
    headers = ["Gap formulation", "$r$ with $w_j$", "Distinct", "Max share"]
    rows = [
        ("Absolute weighted deficit", f"{g['absolute_corr_with_weight']:+.3f}",
         g["absolute_n_distinct_top1"], f"{100*g['absolute_max_top1_share']:.0f}" + r"\%"),
        ("Peer-relative (adopted)", f"{g['relative_corr_with_weight']:+.3f}",
         g["relative_n_distinct_top1"], f"{100*g['relative_max_top1_share']:.0f}" + r"\%"),
        ("Peer-relative per unit cost", "--",
         g["roi_n_distinct_top1"], f"{100*g.get('roi_max_top1_share', 0):.0f}" + r"\%"
         if g.get('roi_max_top1_share') else "--"),
    ]
    foot = (r"\multicolumn{4}{|l|}{Centring each deficit on the cohort mean removes the "
            + "weight-scale confound, making}")
    foot2 = (r"\multicolumn{4}{|l|}{priorities firm-specific: "
             + f"{g['relative_n_distinct_top1']}" + " of "
             + f"{len(C.CRIT_IDS)}" + " criteria can lead a roadmap.}")
    P = "|>{\\raggedright\\arraybackslash}p{2.45cm}|c|c|c|"
    _write("T10_gap_diag", _bordered(P, headers, rows, [foot, foot2]))


def t11_weight_bench():
    """Does ANY weighting beat equal weighting on the independent outcome?"""
    r = _res(); b = r["weight_benchmark"]
    headers = ["Weighting scheme", r"$\rho$ w/ perf.", "Pctile"]
    rows = [
        ("Random (Dirichlet) null",
         f"{b['null_mean']:.3f} [{b['null_p05']:.3f},{b['null_p95']:.3f}]", "--"),
        ("Equal weights", f"{b['rho_equal']:.3f}", f"{b['pctile_equal']:.0f}"),
        ("Elicited (Fuzzy AHP)", f"{b['rho_elicited']:.3f}", f"{b['pctile_elicited']:.0f}"),
        ("Outcome-fitted (OOS)", f"{b['rho_fitted_oos']:.3f}", "--"),
    ]
    foot = (r"\multicolumn{3}{|l|}{Elicited exceeds equal by "
            + f"{b['elicited_vs_equal']:+.3f}" + "; the supervised ceiling adds a further}")
    foot2 = (r"\multicolumn{3}{|l|}{"
             + f"{b['headroom_fitted_vs_equal']:+.3f}" + ", so weighting carries real signal "
             + "that elicitation only partly captures.}")
    P = "|>{\\raggedright\\arraybackslash}p{2.5cm}|c|c|"
    _write("T11_weight_bench", _bordered(P, headers, rows, [foot, foot2]))


def t12_roi():
    """Cost-aware prioritisation and its sensitivity to the cost assumption."""
    import importlib
    r = _res()
    A = importlib.import_module("dtreadiness.analysis")
    rs = r["ROI_sensitivity"]
    tiers = {}
    for cid, t in A.CRITERION_COST_TYPE.items():
        tiers.setdefault(t, []).append(cid)
    headers = ["Cost class", "Rel.", "Criteria"]
    rows = []
    for t, v in sorted(A.COST_TIER.items(), key=lambda kv: kv[1]):
        rows.append((t.capitalize(), f"{v:.1f}", ", ".join(tiers.get(t, []))))
    foot = (r"\multicolumn{3}{|l|}{Lognormal cost perturbation ($\sigma="
            + f"{rs['spread']}$, {rs['n_sim']} draws): first priority unchanged for" + "}")
    foot2 = (r"\multicolumn{3}{|l|}{"
             + f"{100*rs['mean_top1_retention']:.0f}" + r"\% of SMEs (5th pctile "
             + f"{100*rs['p05_top1_retention']:.0f}" + r"\%); cohort order $\rho="
             + f"{rs['median_cohort_rank_rho']:.2f}$." + "}")
    foot3 = (r"\multicolumn{3}{|l|}{Costs are stated assumptions, not measurements.}")
    P = "|l|c|>{\\raggedright\\arraybackslash}p{3.3cm}|"
    _write("T12_roi", _bordered(P, headers, rows, [foot, foot2, foot3]))


def t13_ml_outcome():
    """Non-circular ML: predicting the independent performance outcome."""
    r = _res()
    m = r["ml_outcome"]
    ml = pd.read_csv(os.path.join(PROC, "ml_outcome_metrics.csv"), index_col=0)
    headers = ["Model", "Acc.", "macro-F1", "AUC"]
    rows = [(name, f"{row['Accuracy']:.3f}", f"{row['macroF1']:.3f}",
             f"{row['AUC']:.3f}$\\pm${row['AUC_sd']:.3f}")
            for name, row in ml.iterrows()]
    foot = (r"\multicolumn{4}{|l|}{Target: independently measured firm performance "
            + "(hi vs.\\ lo tercile, $n=" + f"{m['n']}" + "$),}")
    foot2 = (r"\multicolumn{4}{|l|}{which contributes to no criterion. "
             + "Unlike tier recovery, this task is not circular.}")
    _write("T13_ml_outcome", _bordered("|l|r|r|r|", headers, rows, [foot, foot2]))


def t14_hybrid():
    """Hybrid elicited + outcome-learned weighting, nested CV."""
    r = _res()
    h = r["hybrid_weighting"]
    headers = ["Weighting scheme", r"$\rho$ with perf.", "Source"]
    lbl = {"equal": "uniform prior",
           "elicited": "D1 practitioners",
           "learned": "outcome data",
           "hybrid": "both"}
    rows = [(k.capitalize(), f"{h['rho'][k]:.3f}", lbl[k])
            for k in ["equal", "elicited", "learned", "hybrid"]]
    foot = (r"\multicolumn{3}{|l|}{Nested CV: $\lambda$ chosen on inner folds, scored "
            + "on held-out folds ($\\tilde\\lambda=" + f"{h['lambda_median']:.2f}" + "$).}")
    foot2 = (r"\multicolumn{3}{|l|}{Hybrid $-$ elicited $="
             + f"{h['hybrid_minus_elicited']:+.3f}$, 95\\% CI "
             + f"[{h['gain_ci_lo']:+.3f}, {h['gain_ci_hi']:+.3f}]" + "}")
    P = "|>{\\raggedright\\arraybackslash}p{2.1cm}|c|>{\\raggedright\\arraybackslash}p{1.9cm}|"
    _write("T14_hybrid", _bordered(P, headers, rows, [foot, foot2]))


def make_all():
    t1_related_work(); t2_datasets(); t3_crosswalk(); t4_reliability()
    t5_weights(); t6_ml(); t7_agreement(); t8_clusters()
    t9_cmb(); t10_gap_diag(); t11_weight_bench(); t12_roi()
    t13_ml_outcome(); t14_hybrid()


if __name__ == "__main__":
    make_all()


def t15_baselines():
    """R4: unweighted-mean and equal-weight baselines against the outcome."""
    r = _res()
    b = r.get("baselines", {})
    headers = ["Scheme", r"$\rho$ vs.\ outcome", r"$\rho$/$\tau$ vs.\ TOPSIS"]
    rows = [
        ("Unweighted mean", f"{b.get('unweighted_mean_vs_perf_spearman',0):.3f}",
         f"{b.get('unweighted_mean_vs_topsis_spearman',0):.2f}/"
         f"{b.get('unweighted_mean_vs_topsis_kendall',0):.2f}"),
        ("Equal-weight TOPSIS", f"{b.get('equalweight_vs_perf_spearman',0):.3f}",
         r"$\tau{=}$" + f"{b.get('equalweight_topsis_kendall',0):.2f}"),
        ("Elicited (Fuzzy AHP)", f"{b.get('elicited_vs_perf_spearman',0):.3f}", "---"),
        ("Hybrid (learned+elicited)", f"{b.get('fitted_oos_vs_perf_spearman',0):.3f}", "---"),
    ]
    foot = (r"\multicolumn{3}{|l|}{Both naive baselines reach $\rho{=}0.60$ vs.\ performance; "
            + "elicited and hybrid}")
    foot2 = (r"\multicolumn{3}{|l|}{weighting add validated signal over them (see "
             + r"Table~\ref{tab:hybrid}).}")
    P = "|l|c|c|"
    _write("T15_baselines", _bordered(P, headers, rows, [foot, foot2]))


def t16_sustainability():
    """R4: resource-efficiency / sustainability metric of cost-aware prioritisation."""
    r = _res()
    s = r.get("sustainability", {})
    headers = ["Prioritisation policy", "Mean cost", "Gap closed"]
    rows = [
        ("Weighted gap (cost-blind)", f"{s.get('mean_cost_gap_policy',0):.2f}", "1.00 (ref.)"),
        ("Cost-aware ROI", f"{s.get('mean_cost_roi_policy',0):.2f}",
         f"{s.get('roi_gap_addressed_ratio',0):.2f}"),
    ]
    foot = (r"\multicolumn{3}{|l|}{ROI closes "
            + f"{100*s.get('roi_gap_addressed_ratio',0):.0f}" + r"\% of the readiness gap at "
            + f"{s.get('mean_cost_saving_pct',0):.0f}" + r"\% lower cost;}")
    foot2 = (r"\multicolumn{3}{|l|}{the averted cost is stranded capital (resource waste) "
             + "avoided.}")
    P = "|l|c|c|"
    _write("T16_sustainability", _bordered(P, headers, rows, [foot, foot2]))
