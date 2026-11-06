"""Global configuration: paths, criteria taxonomy, construct crosswalk, fuzzy scales.

A Hybrid Fuzzy-AHP-TOPSIS Framework with Machine-Learning Validation for
Digital Twin Readiness Assessment in Resource-Constrained Manufacturing SMEs.
"""
from __future__ import annotations
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Set DT_READINESS_DATA_DIR to a folder containing the source-data subfolders.
DATA_RAW = os.path.abspath(os.environ.get(
    "DT_READINESS_DATA_DIR", os.path.join(ROOT, "data", "raw")))
DATA_PROC = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
TAB_DIR = os.path.join(ROOT, "reports", "tables")

D1_PATH = os.path.join(
    DATA_RAW, "d1", "Digital-Twin Model Adoption in Malaysian Semicondusctor Industry.xlsx")
D2_PATH = os.path.join(
    DATA_RAW, "d2", "Raw data.xlsx")

SEED = 42

# ----------------------------------------------------------------------------
# Criteria taxonomy: Goal -> 4 dimensions -> 12 constructs (criteria)
# ----------------------------------------------------------------------------
# Every criterion is a *benefit* criterion (higher construct score => more ready).
CRITERIA = [
    # id,   short,     long name,                              dimension
    ("C1",  "DataGov", "Data Governance & Repository",         "Technological"),
    ("C2",  "Interop", "Device Interoperability",              "Technological"),
    ("C3",  "BizCrit", "Business Criticality",                 "Strategic"),
    ("C4",  "Knowl",   "Knowledge Capacity",                   "Human"),
    ("C5",  "Visual",  "Visualization & Interaction",          "Technological"),
    ("C6",  "Adapt",   "Adaptation to Future Demands",         "Strategic"),
    ("C7",  "ProcOpt", "Process Optimization",                 "Organizational"),
    ("C8",  "Fault",   "Fault Tolerance & Reliability",        "Technological"),
    ("C9",  "ProdPerf","Production & Performance",             "Organizational"),
    ("C10", "Resource","Resource Accessibility",               "Strategic"),
    ("C11", "Infra",   "Infrastructure & Mgmt Maturity",       "Organizational"),
    ("C12", "HumanOrg","Human-Centric Organization",           "Human"),
]
CRIT_IDS = [c[0] for c in CRITERIA]
CRIT_SHORT = {c[0]: c[1] for c in CRITERIA}
CRIT_LONG = {c[0]: c[2] for c in CRITERIA}
CRIT_DIM = {c[0]: c[3] for c in CRITERIA}

# ----------------------------------------------------------------------------
# Construct crosswalk (Table T3):  criterion  ->  D2 item-group prefixes
# Documented, taxonomy-justified mapping of the 12 DT-readiness constructs onto
# the Malaysian-SME survey (D2) item groups.  Every D2 group is used exactly once.
# ----------------------------------------------------------------------------
CROSSWALK_D2 = {
    "C1":  ["HCIS"],                 # communication/information systems  -> data governance
    "C2":  ["OIDI"],                 # digital/technological innovation   -> interoperability
    "C3":  ["OPBG"],                 # business growth                    -> business criticality
    "C4":  ["HTR", "HSE"],           # training + self-efficacy           -> knowledge capacity
    "C5":  ["OICI"],                 # communication innovation           -> visualization/interaction
    "C6":  ["EINN"],                 # innovation orientation             -> adaptation to future
    "C7":  ["OIMI"],                 # managerial/process innovation      -> process optimization
    "C8":  ["ERT"],                  # reliability/technology robustness  -> fault tolerance
    "C9":  ["OPSF", "OPSNF", "EPRO"],# financial+non-financial perf+proc  -> production/performance
    "C10": ["OPRC"],                 # resource capacity                  -> resource accessibility
    "C11": ["MTM", "MTG"],           # management maturity + governance   -> infrastructure maturity
    "C12": ["HJD", "HPA", "HCO"],    # job design + appraisal + comp      -> human-centric organization
}

# D1 (Digital-Twin adoption survey) construct name prefixes, mapped to criteria.
# These give the expert-proxy *importance* used to seed the Fuzzy-AHP pairwise matrix.
CROSSWALK_D1 = {
    "C1":  "Data Repository",
    "C2":  "Devices Interoperability",
    "C3":  "Business' Criticality",
    "C4":  "Knowledge Capacity",
    "C5":  "Visualization and Interaction Process",
    "C6":  "Adaption of Future demands",
    "C7":  "Optimization of planned Layout",
    "C8":  "Fault tolerance of machinery",
    "C9":  "Optimized production and product",
    "C10": "Resource Accessibility",
    "C11": "Infrastructure Maturity",
    # C12 (human-centric organization) is the Industry-5.0 criterion with no direct
    # D1 antecedent; its importance is anchored on the overall adoption-salience block.
    "C12": "Adoption of Digital Twin in the industry",
}

# D1 Likert string -> integer (1..5)
LIKERT_MAP = {
    "completely disagree": 1, "strongly disagree": 1, "disagree": 2,
    "neutral": 3, "neither agree nor disagree": 3, "undecided": 3,
    "agree": 4, "strongly agree": 5, "completely agree": 5,
}

# ----------------------------------------------------------------------------
# Fuzzy linguistic scale for performance ratings (Chen 2000), on a 5-point base.
# VL(0,0,1) L(0,1,3) M(3,5,7) H(7,9,10) VH(9,10,10)
# ----------------------------------------------------------------------------
LINGUISTIC_TFN = {
    1: (0.0, 0.0, 1.0),   # Very Low
    2: (0.0, 1.0, 3.0),   # Low
    3: (3.0, 5.0, 7.0),   # Medium
    4: (7.0, 9.0, 10.0),  # High
    5: (9.0, 10.0, 10.0), # Very High
}

# Saaty fuzzy scale for AHP pairwise judgements: integer x -> TFN
def saaty_tfn(x: int):
    """Triangular fuzzification of a Saaty 1..9 judgement (or its reciprocal)."""
    if x == 1:
        return (1.0, 1.0, 2.0)
    if x >= 2:
        return (float(x - 1), float(x), float(min(x + 1, 9)))
    # reciprocal 1/k
    k = round(1.0 / x)
    l, m, u = saaty_tfn(k)
    return (1.0 / u, 1.0 / m, 1.0 / l)

# Readiness tier labels
TIERS = ["Low", "Medium", "High"]
