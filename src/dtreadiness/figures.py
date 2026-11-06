"""Publication figures (F1-F9 + dataset demographic pies).

Colorblind-safe (Okabe-Ito) palette, serif fonts, exported as vector PDF + 300 dpi PNG.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from math import pi
from . import config as C
from . import data as D

# ---- style -----------------------------------------------------------------
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
      "#56B4E9", "#F0E442", "#000000", "#999999"]
DIM_COLOR = {"Technological": "#0072B2", "Strategic": "#E69F00",
             "Organizational": "#009E73", "Human": "#D55E00"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 120, "savefig.bbox": "tight",
    # Embed Type 42 (TrueType) rather than matplotlib default Type 3 bitmap fonts:
    # IEEE PDF eXpress and most journals require fully embedded vector fonts.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # mathtext defaults to matplotlib Computer Modern, which embeds as Type 3;
    # use the regular serif family so math labels embed as TrueType too.
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
})
PROC = C.DATA_PROC
FIG = C.FIG_DIR
os.makedirs(FIG, exist_ok=True)


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print("  figure:", name)


def _res():
    with open(os.path.join(PROC, "results.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# F1 - Framework architecture: full-width multi-stage flowchart, distinct shapes
# ---------------------------------------------------------------------------
from matplotlib.patches import Polygon, Ellipse, RegularPolygon, Circle

def _hexagon(ax, cx, cy, w, h, color, label, fs=6.6):
    pts = [(cx-w/2, cy), (cx-w/4, cy+h/2), (cx+w/4, cy+h/2),
           (cx+w/2, cy), (cx+w/4, cy-h/2), (cx-w/4, cy-h/2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color+"22",
                 edgecolor=color, linewidth=1.3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs)

def _document(ax, cx, cy, w, h, color, label, fs=6.6):
    # rectangle with a wavy bottom (report/output icon)
    import numpy as _np
    xs = _np.linspace(cx-w/2, cx+w/2, 40)
    wave = cy - h/2 + 0.9*_np.sin((xs-cx)/w*4*_np.pi)
    verts = [(cx-w/2, cy+h/2), (cx+w/2, cy+h/2)] + list(zip(xs[::-1], wave[::-1]))
    ax.add_patch(Polygon(verts, closed=True, facecolor=color+"33",
                 edgecolor=color, linewidth=1.3))
    ax.text(cx, cy+0.6, label, ha="center", va="center", fontsize=fs, fontweight="bold")

def _gear(ax, cx, cy, r, color):
    ax.add_patch(RegularPolygon((cx, cy), numVertices=8, radius=r*1.35,
                 orientation=np.pi/8, facecolor=color, edgecolor=color, linewidth=0))
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=color, linewidth=1.0))
    ax.add_patch(Circle((cx, cy), r*0.4, facecolor=color, edgecolor=color))

def _bars_icon(ax, cx, cy, s, color):
    for k, hh in enumerate([0.5, 1.0, 0.7]):
        ax.add_patch(plt.Rectangle((cx-1.3*s+k*s, cy-0.6*s), 0.6*s, hh*s,
                     facecolor=color, edgecolor="none"))

def _dbicon(ax, cx, cy, s, color):
    for dy in (0.5, 0, -0.5):
        ax.add_patch(Ellipse((cx, cy+dy*s), 1.4*s, 0.5*s, facecolor="none",
                     edgecolor=color, linewidth=0.9))

def _cylinder(ax, cx, cy, w, h, color, label):
    er = h * 0.16
    ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2 + er), w, h - 2*er,
                 facecolor=color + "33", edgecolor=color, linewidth=1.2))
    ax.add_patch(Ellipse((cx, cy + h/2 - er), w, 2*er, facecolor=color + "55",
                 edgecolor=color, linewidth=1.2))
    ax.add_patch(Ellipse((cx, cy - h/2 + er), w, 2*er, facecolor=color + "33",
                 edgecolor=color, linewidth=1.2))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=7, fontweight="bold")

def _proc(ax, cx, cy, w, h, color, label, fs=7.2):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.15,rounding_size=1.0",
                 facecolor=color + "22", edgecolor=color, linewidth=1.3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs)

def _para(ax, cx, cy, w, h, color, label):
    sk = w * 0.16
    pts = [(cx - w/2 + sk, cy - h/2), (cx + w/2, cy - h/2),
           (cx + w/2 - sk, cy + h/2), (cx - w/2, cy + h/2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color + "33",
                 edgecolor=color, linewidth=1.3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=7, fontweight="bold")

def _diamond(ax, cx, cy, w, h, color, label):
    pts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color + "22",
                 edgecolor=color, linewidth=1.3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=6.6)

def _arrow(ax, p1, p2, color="#333333", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                 linewidth=1.2, color=color, shrinkA=1, shrinkB=1))

def fig1_architecture():
    """Single-column vertical flowchart with varied shapes + drawn icons."""
    fig, ax = plt.subplots(figsize=(3.45, 4.35))
    ax.axis("off"); ax.set_xlim(0, 60); ax.set_ylim(0, 100)
    cx = 21  # main spine x
    ox = 47  # side-output x
    bands = [("A", "Data", OI[0], 74, 100),
             ("B", "Weighting", OI[1], 58, 74),
             ("C", "Ranking", OI[2], 44, 58),
             ("D", "Validation &\nPrescription", OI[3], 1, 44)]
    for tag, name, col, y0, y1 in bands:
        ax.add_patch(FancyBboxPatch((1, y0), 58, y1-y0-0.6,
                     boxstyle="round,pad=0.1,rounding_size=1.2",
                     facecolor=col+"0d", edgecolor=col+"55", linewidth=0.8, linestyle=(0,(4,2))))
        ax.text(3.2, (y0+y1)/2, tag, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=col, rotation=0)
        ax.text(58, y1-2.2, name, ha="right", va="top", fontsize=6.0, color=col, style="italic")
    # A: three dataset cylinders -> construct matrix
    for i, (lab) in enumerate(["D1", "D2", "D3"]):
        _cylinder(ax, 12+i*15, 91, 12, 7.5, OI[0], f"{lab}")
    _proc(ax, cx, 79, 26, 8.5, OI[8], "Construct feature matrix (12 crit.)", 6.6)
    for i in range(3):
        _arrow(ax, (12+i*15, 87), (cx+(i-1)*3, 83.5))
    # B: AHP hexagon -> weights parallelogram
    _hexagon(ax, cx, 66, 26, 11, OI[1], "Fuzzy AHP\n(Chang+Buckley)")
    _para(ax, ox, 66, 18, 8, OI[1], "$w_j$")
    _arrow(ax, (cx, 74.7), (cx, 71.7))
    _arrow(ax, (cx+13, 66), (ox-9, 66), color="#888")
    # C: TOPSIS box -> CC parallelogram
    _proc(ax, cx, 51, 26, 9, OI[2], "Fuzzy TOPSIS (Chen)", 6.8)
    _para(ax, ox, 51, 18, 8, OI[2], "$CC_i$, tiers")
    _arrow(ax, (cx, 60.4), (cx, 55.6))
    _arrow(ax, (cx+13, 51), (ox-9, 51), color="#888")
    # D: ML+SHAP hexagon(w/gear) -> decision diamond -> FCM -> roadmap doc(w/bars)
    _hexagon(ax, cx, 37, 27, 10, OI[3], "  ML + SHAP audit", 6.6); _gear(ax, cx-9.5, 37, 1.8, OI[3])
    _diamond(ax, cx, 24, 22, 11, OI[3], "AHP $\\approx$\nSHAP?")
    _proc(ax, cx, 12.5, 26, 8, OI[3], "FCM + Gap decomposition", 6.4)
    _document(ax, ox, 12.5, 20, 12, OI[3], "SME\nroadmap", 6.0)
    _bars_icon(ax, ox, 8.2, 0.9, OI[3])
    _arrow(ax, (cx, 46.5), (cx, 42.2))
    _arrow(ax, (cx, 31.9), (cx, 29.6))
    _arrow(ax, (cx, 18.4), (cx, 16.6))
    _arrow(ax, (cx+13, 12.5), (ox-9, 12.5), color="#888")
    ax.text(30, 0.0, "Industry 5.0 human-in-the-loop decision support",
            ha="center", va="bottom", fontsize=5.8, style="italic", color="#666")
    _save(fig, "F1_architecture")


# ---------------------------------------------------------------------------
# F2 - Criteria hierarchy tree
# ---------------------------------------------------------------------------
def fig2_hierarchy():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(50, 94, "Goal: DT Readiness", ha="center", fontweight="bold", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#eeeeee", ec="#333"))
    dims = ["Technological", "Strategic", "Organizational", "Human"]
    dx = {d: 12.5 + i*25 for i, d in enumerate(dims)}
    for d in dims:
        ax.plot([50, dx[d]], [90, 66], color="#999", lw=0.8)
        ax.text(dx[d], 62, d, ha="center", fontweight="bold", fontsize=8.4,
                color=DIM_COLOR[d],
                bbox=dict(boxstyle="round,pad=0.3", fc=DIM_COLOR[d]+"22", ec=DIM_COLOR[d]))
    # constructs under each dimension
    from collections import defaultdict
    groups = defaultdict(list)
    for cid in C.CRIT_IDS:
        groups[C.CRIT_DIM[cid]].append(cid)
    for d in dims:
        cids = groups[d]
        n = len(cids)
        ys = np.linspace(48, 6, n)
        for k, cid in enumerate(cids):
            ax.plot([dx[d], dx[d]-9], [58, ys[k]], color="#ccc", lw=0.6)
            ax.text(dx[d]-9, ys[k], f"{cid}: {C.CRIT_SHORT[cid]}", ha="center", va="center",
                    fontsize=7, color=DIM_COLOR[d],
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=DIM_COLOR[d], lw=0.6))
    ax.text(50, 1, "12 constructs operationalized by 22 Malaysian-SME item groups (75 items)",
            ha="center", fontsize=7.2, style="italic", color="#555")
    _save(fig, "F2_hierarchy")


# ---------------------------------------------------------------------------
# F3 - Reliability bar + demographic pies
# ---------------------------------------------------------------------------
def fig3_psychometrics():
    rel = pd.read_csv(os.path.join(PROC, "reliability.csv"))
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.55, wspace=0.55)
    # (a) Cronbach alpha bars
    ax = fig.add_subplot(gs[0, :])
    cids = rel["Criterion"].tolist()
    colors = [DIM_COLOR[C.CRIT_DIM[c]] for c in cids]
    ax.bar(cids, rel["Cronbach_alpha"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.70, color=OI[3], ls="--", lw=1.1)
    ax.set_ylim(0.6, 1.02); ax.set_ylabel(r"Cronbach's $\alpha$")
    ax.set_title("(a) Internal consistency by construct", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DIM_COLOR[d]) for d in DIM_COLOR]
    ax.legend(handles + [plt.Line2D([0], [0], color=OI[3], ls='--')],
              list(DIM_COLOR.keys()) + [r"$\alpha=0.70$"], ncol=5, fontsize=6.8,
              loc="lower center", bbox_to_anchor=(0.5, 0.005),
              frameon=True, framealpha=0.9, edgecolor="#cccccc")
    # demographic pies (group slices <5% into "Other" to avoid label collisions)
    df = D.load_d2()
    def catmap(col, mapping=None, topn=None):
        s = df[col].dropna()
        if mapping:
            s = s.map(lambda x: mapping.get(int(x), f"{col} {int(x)}") if pd.notna(x) else x)
        vc = s.value_counts()
        if topn:
            vc = vc.head(topn)
        pct = 100 * vc / vc.sum()
        small = pct[pct < 5].index
        if len(small) > 1:
            vc = pd.concat([vc.drop(small), pd.Series({"Other": vc[small].sum()})])
        return vc
    emp = {1: "<5", 2: "5-30", 3: "30-75", 4: "75-200", 5: ">200"}
    own = {1: "Local", 2: "Foreign", 3: "Joint"}
    edu = {1: "SPM", 2: "Diploma", 3: "Degree", 4: "Master", 5: "PhD"}
    ind = {6: "Food/Bev", 4: "Textile", 5: "Wood", 12: "Metal", 1: "Electrical"}
    specs = [("Employee", emp, None, "(b) Firm size"),
             ("Ownership", own, None, "(c) Ownership"),
             ("Education", edu, None, "(d) Education"),
             ("Industry", ind, 5, "(e) Top industries")]
    for i, (col, mp, topn, title) in enumerate(specs):
        axp = fig.add_subplot(gs[1, i])
        vc = catmap(col, mp, topn)
        axp.pie(vc.values, labels=vc.index,
                autopct=lambda p: f"{p:.0f}%" if p >= 5 else "", startangle=90,
                colors=OI[:len(vc)], textprops={"fontsize": 6.2},
                pctdistance=0.72, labeldistance=1.12,
                wedgeprops={"edgecolor": "white", "linewidth": 0.6})
        axp.set_title(title, fontsize=8)
    _save(fig, "F3_psychometrics")


# ---------------------------------------------------------------------------
# F4 - Fuzzy AHP weights (Chang vs Buckley w/ Buckley TFN spread)
# ---------------------------------------------------------------------------
def fig4_weights():
    w = pd.read_csv(os.path.join(PROC, "ahp_weights.csv"), index_col=0).reindex(C.CRIT_IDS)
    order = w["w_buckley"].sort_values().index
    w = w.loc[order]
    y = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = [DIM_COLOR[C.CRIT_DIM[c]] for c in w.index]
    # buckley TFN spread as error bars
    xerr = np.vstack([w["w_buckley"]-w["buckley_l"], w["buckley_u"]-w["w_buckley"]])
    ax.barh(y, w["w_buckley"], color=colors, edgecolor="black", linewidth=0.4,
            xerr=np.abs(xerr), error_kw=dict(ecolor="#555", lw=0.8, capsize=2), label="Buckley (crisp)")
    ax.scatter(w["w_chang"], y, marker="D", color="black", s=18, zorder=5, label="Chang extent")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}: {C.CRIT_SHORT[c]}" for c in w.index])
    ax.set_xlabel("Criterion weight")
    ax.set_title("Fuzzy-AHP criterion weights (Buckley bars, Chang diamonds; TFN spread)")
    ax.legend(loc="lower right")
    _save(fig, "F4_weights")


# ---------------------------------------------------------------------------
# F5 - Fuzzy TOPSIS CC distribution
# ---------------------------------------------------------------------------
def fig5_cc_distribution():
    sc = pd.read_csv(os.path.join(PROC, "topsis_scores.csv"))
    cc = sc["CC"].values
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(cc, bins=30, color=OI[0]+"cc", edgecolor="white", linewidth=0.5, density=True)
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(cc); xs = np.linspace(cc.min(), cc.max(), 200)
    ax.plot(xs, kde(xs), color=OI[3], lw=1.6, label="KDE")
    q1, q2 = np.quantile(cc, [1/3, 2/3])
    for q, lab in [(q1, "Low|Med"), (q2, "Med|High")]:
        ax.axvline(q, color="black", ls="--", lw=1.0)
        ax.text(q, ax.get_ylim()[1]*0.92, lab, rotation=90, va="top", ha="right", fontsize=7)
    ax.set_xlabel("Closeness coefficient  $CC_i$"); ax.set_ylabel("Density")
    ax.set_title(f"Fuzzy-TOPSIS readiness distribution across {len(cc)} SMEs")
    ax.legend()
    _save(fig, "F5_cc_distribution")


# ---------------------------------------------------------------------------
# F6 - SHAP beeswarm (best tree model)
# ---------------------------------------------------------------------------
def fig6_shap():
    import shap
    from . import ml as ML
    X = pd.read_csv(os.path.join(PROC, "ml_X.csv"))
    sc = pd.read_csv(os.path.join(PROC, "topsis_scores.csv"))
    y = pd.Categorical(sc["tier"], categories=C.TIERS, ordered=True).codes
    _, _, explainer, sv = ML.shap_importance(X, y, "LightGBM")
    Xd = X.copy(); Xd.columns = [f"{c}:{C.CRIT_SHORT[c]}" for c in X.columns]
    # aggregate multiclass -> mean abs for beeswarm on the 'High' class (index 2)
    if isinstance(sv, list):
        vals = sv[-1]
    else:
        arr = np.asarray(sv)
        vals = arr[:, :, -1] if arr.ndim == 3 else arr
    fig = plt.figure(figsize=(6.4, 4.2))
    shap.summary_plot(vals, Xd, show=False, plot_size=None, color_bar=True, max_display=12)
    fig = plt.gcf(); fig.set_size_inches(6.4, 4.2)
    fig.suptitle("SHAP summary (LightGBM, High-readiness class)", fontsize=9, y=1.0)
    _save(fig, "F6_shap")


# ---------------------------------------------------------------------------
# F7 - AHP vs SHAP weight agreement scatter
# ---------------------------------------------------------------------------
def fig7_agreement():
    ag = pd.read_csv(os.path.join(PROC, "weight_agreement.csv"), index_col=0).reindex(C.CRIT_IDS)
    r = _res()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    colors = [DIM_COLOR[C.CRIT_DIM[c]] for c in ag.index]
    ax.scatter(ag["w_AHP"], ag["w_SHAP"], c=colors, s=52, edgecolor="black", linewidth=0.5, zorder=4)
    lim = max(ag["w_AHP"].max(), ag["w_SHAP"].max())*1.15
    ax.plot([0, lim], [0, lim], ls="--", color="#888", lw=1.0, label="45$^\\circ$ agreement")
    for c in ag.index:
        ax.annotate(c, (ag.loc[c, "w_AHP"], ag.loc[c, "w_SHAP"]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Fuzzy-AHP weight  $w_{AHP}$"); ax.set_ylabel("SHAP importance  $w_{SHAP}$")
    ax.set_title("Expert vs data-driven weights")
    ax.text(0.03, 0.95, f"Spearman $\\rho$={r['AHP_SHAP_spearman']:.2f}\n"
            f"Kendall $\\tau$={r['AHP_SHAP_kendall']:.2f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))
    ax.legend(loc="lower right")
    _save(fig, "F7_agreement")


# ---------------------------------------------------------------------------
# F8 - Sensitivity heatmap
# ---------------------------------------------------------------------------
def fig8_sensitivity():
    hm = pd.read_csv(os.path.join(PROC, "sensitivity_oat.csv"), index_col=0).reindex(C.CRIT_IDS)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(hm))); ax.set_yticklabels([f"{c}:{C.CRIT_SHORT[c]}" for c in hm.index])
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{hm.values[i,j]:.1f}", ha="center", va="center", fontsize=6,
                    color="black" if hm.values[i,j] < hm.values.max()*0.6 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean |rank change|", fontsize=8)
    ax.set_title("OAT weight-perturbation sensitivity")
    ax.grid(False)
    _save(fig, "F8_sensitivity")


# ---------------------------------------------------------------------------
# F9 - FCM cluster profile radar
# ---------------------------------------------------------------------------
def fig9_radar():
    cent = pd.read_csv(os.path.join(PROC, "cluster_centroids.csv"), index_col=0)[C.CRIT_IDS]
    labels = [C.CRIT_SHORT[c] for c in C.CRIT_IDS]
    N = len(labels)
    ang = [n / float(N) * 2 * pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(5.2, 4.8), subplot_kw=dict(polar=True))
    names = {"Cluster1": "Advancing SMEs", "Cluster2": "Emerging SMEs"}
    for i, (idx, row) in enumerate(cent.iterrows()):
        vals = row.values.tolist() + [row.values[0]]
        ax.plot(ang, vals, lw=1.8, color=OI[i], label=names.get(idx, idx))
        ax.fill(ang, vals, color=OI[i], alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(2.4, 3.3)
    ax.set_title("FCM cluster readiness signatures (c*=2)", y=1.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    _save(fig, "F9_radar")


# ---------------------------------------------------------------------------
# F1b - Layered software architecture (L1..L8), data flows downward
# ---------------------------------------------------------------------------
def fig1b_layers():
    layers = [
        ("L1", "I/O", "load_dt · load_sme · load_taxonomy", OI[0], "-"),
        ("L2", "Preprocess", "likert map · missing · construct scores", OI[0], "L1"),
        ("L3", "Psychometrics", r"$\alpha$ · CR/AVE · KMO · Bartlett", OI[5], "L2"),
        ("L4", "Fuzzy MCDM", "TFN · AHP (Chang/Buckley) · TOPSIS · VIKOR", OI[1], "L2,L3"),
        ("L5", "Learning", "RF · XGB · LGBM · MLP · SHAP", OI[4], "L2"),
        ("L6", "Clustering", "Fuzzy C-Means · PC / Xie-Beni", OI[2], "L2,L4"),
        ("L7", "Analysis", "sensitivity · weight-agreement · gap decomp.", OI[3], "L4,L5"),
        ("L8", "Presentation", "figures · LaTeX tables", OI[8], "L4-L7"),
    ]
    n = len(layers)
    fig, ax = plt.subplots(figsize=(3.45, 3.6))
    ax.axis("off"); ax.set_xlim(0, 60); ax.set_ylim(0, 100)
    h = 9.3; gap = 1.5; x0, w = 9, 49
    ys = [97 - i*(h+gap) for i in range(n)]
    for (tag, name, mods, col, dep), y in zip(layers, ys):
        ax.add_patch(FancyBboxPatch((x0, y-h), w, h,
                     boxstyle="round,pad=0.1,rounding_size=1.4",
                     facecolor=col+"22", edgecolor=col, linewidth=1.3))
        ax.text(x0+2.5, y-3.3, f"{tag}", fontsize=8.2, fontweight="bold", color=col, va="center")
        ax.text(x0+8.5, y-3.3, name, fontsize=7.6, fontweight="bold", va="center")
        ax.text(x0+2.5, y-7.6, mods, fontsize=5.9, va="center", color="#333")
    # downward data-flow spine on the left
    ax.annotate("", xy=(5, ys[-1]-h), xytext=(5, ys[0]),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6))
    ax.text(4.1, (ys[0]+ys[-1])/2, "data flow (top-down)", rotation=90,
            ha="center", va="center", fontsize=6.2, color="#555")
    # dependency arrows on the right (higher layer depends on lower)
    for i in range(1, n):
        ax.annotate("", xy=(x0+w+0.3, ys[i-1]-h+1), xytext=(x0+w+0.3, ys[i]-1),
                    arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.9,
                                    connectionstyle="arc3,rad=-0.35"))
    ax.text(58.5, (ys[0]+ys[-1])/2, "depends on", rotation=90, ha="center",
            va="center", fontsize=6.2, color="#999")
    _save(fig, "F1b_layers")


# ---------------------------------------------------------------------------
# F10 - Gap-formulation diagnostics: does the roadmap echo the global weights?
# ---------------------------------------------------------------------------
def fig10_gap_diag():
    """Left: priority concentration under each gap formulation. Right: the
    weight-vs-gap scatter that motivated the revision."""
    r = _res()
    g = r["gap_diagnostics"]
    rel = pd.read_csv(os.path.join(PROC, "gap_relative.csv"), index_col=0)[C.CRIT_IDS]
    ab = pd.read_csv(os.path.join(PROC, "gap_absolute.csv"), index_col=0)[C.CRIT_IDS]
    w = pd.read_csv(os.path.join(PROC, "ahp_weights.csv"),
                    index_col=0).reindex(C.CRIT_IDS)["w_buckley"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) share of SMEs for which each criterion is the first priority
    ax = axes[0]
    sa = ab.idxmax(axis=1).value_counts(normalize=True).reindex(C.CRIT_IDS).fillna(0)
    sr = rel.idxmax(axis=1).value_counts(normalize=True).reindex(C.CRIT_IDS).fillna(0)
    x = np.arange(len(C.CRIT_IDS))
    ax.bar(x - 0.2, 100 * sa.values, 0.4, label="absolute", color=OI[1],
           edgecolor="black", linewidth=0.3)
    ax.bar(x + 0.2, 100 * sr.values, 0.4, label="peer-relative", color=OI[2],
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(C.CRIT_IDS, rotation=90, fontsize=6)
    ax.set_ylabel("SMEs with this first priority (%)")
    ax.set_title("(a) Priority concentration", fontsize=9)
    ax.legend(fontsize=7)

    # (b) mean gap vs criterion weight
    ax = axes[1]
    ax.scatter(w.values, ab.mean().values, s=22, color=OI[1], edgecolor="black",
               linewidth=0.3, label=f"absolute (r={g['absolute_corr_with_weight']:+.2f})")
    ax.scatter(w.values, rel.mean().values, s=22, color=OI[2], marker="D",
               edgecolor="black", linewidth=0.3,
               label=f"peer-relative (r={g['relative_corr_with_weight']:+.2f})")
    ax.axhline(0, color="#777", lw=0.6, ls=":")
    ax.set_xlabel("Criterion weight w\u2c7c")
    ax.set_ylabel("Mean gap")
    ax.set_title("(b) Weight dependence", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, "F10_gap_diag")


# ---------------------------------------------------------------------------
# F11 - Weight-scheme benchmark against the independent outcome
# ---------------------------------------------------------------------------
def fig11_weight_bench():
    """Locates equal, elicited and outcome-fitted weights in the random-weight null."""
    r = _res()
    b = r["weight_benchmark"]
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    lo, hi = b["null_p05"], b["null_p95"]
    ax.axvspan(lo, hi, color=OI[8], alpha=0.30,
               label="random-weight null (5–95th pct)")
    ax.axvline(b["null_mean"], color=OI[8], lw=1.0, ls="--")
    for val, lab, col, mk in [
            (b["rho_equal"], "equal", OI[0], "o"),
            (b["rho_elicited"], "elicited (Fuzzy AHP)", OI[3], "s"),
            (b["rho_fitted_oos"], "outcome-fitted (OOS ceiling)", OI[2], "D")]:
        ax.scatter([val], [0], s=70, color=col, marker=mk, edgecolor="black",
                   linewidth=0.5, zorder=5, label=f"{lab}: {val:.3f}")
    ax.set_yticks([])
    ax.set_xlabel("Spearman \u03c1 with independently measured firm performance")
    ax.set_title("Criterion validity of competing weighting schemes ($n$=321)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, "F11_weight_bench")


# ---------------------------------------------------------------------------
# F12 - Where outcome data corrects expert elicitation
# ---------------------------------------------------------------------------
def fig12_hybrid():
    """Left: elicited vs outcome-learned weight per criterion. Right: nested-CV
    criterion validity of each weighting scheme."""
    r = _res()
    h = r["hybrid_weighting"]
    w = pd.read_csv(os.path.join(PROC, "hybrid_weights.csv"), index_col=0)
    w = w.reindex(C.CRIT_IDS)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    # (a) per-criterion correction, sorted by size of change
    ax = axes[0]
    delta = (w["w_learned"] - w["w_elicited"]).sort_values()
    yy = np.arange(len(delta))
    cols = [OI[2] if v > 0 else OI[3] for v in delta.values]
    ax.barh(yy, delta.values, color=cols, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="#444", lw=0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{c}: {C.CRIT_SHORT[c]}" for c in delta.index], fontsize=6.5)
    ax.set_xlabel("learned \u2212 elicited weight")
    ax.set_title("(a) Where outcome data corrects experts", fontsize=9)

    # (b) nested-CV validity of each scheme
    ax = axes[1]
    order = ["equal", "elicited", "learned", "hybrid"]
    vals = [h["rho"][k] for k in order]
    cols = [OI[8], OI[3], OI[0], OI[2]]
    xx = np.arange(len(order))
    ax.bar(xx, vals, color=cols, edgecolor="black", linewidth=0.4, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xx)
    ax.set_xticklabels(order, rotation=20, fontsize=7.5)
    ax.set_ylim(0.55, max(vals) + 0.045)
    ax.set_ylabel("Spearman \u03c1 with performance")
    ax.set_title("(b) Nested-CV criterion validity", fontsize=9)
    fig.tight_layout()
    _save(fig, "F12_hybrid")


def make_all():
    fig1_architecture(); fig1b_layers(); fig2_hierarchy(); fig3_psychometrics(); fig4_weights()
    fig5_cc_distribution(); fig6_shap(); fig7_agreement(); fig8_sensitivity(); fig9_radar()


if __name__ == "__main__":
    make_all()
