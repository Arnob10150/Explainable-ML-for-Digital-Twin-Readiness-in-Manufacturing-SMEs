"""One-command reproduction: pipeline -> figures -> tables.

Usage:  python scripts/run_all.py
Then compile the paper:  cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
"""
import sys, os, subprocess
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

print("=" * 60, "\n[1/3] Running analysis pipeline ...\n", "=" * 60)
subprocess.check_call([sys.executable, os.path.join(HERE, "run_pipeline.py")])

print("=" * 60, "\n[2/3] Generating figures ...\n", "=" * 60)
from dtreadiness import figures as F
F.make_all()

print("=" * 60, "\n[3/3] Generating tables ...\n", "=" * 60)
from dtreadiness import tables as T
T.make_all()

print("\nAll artifacts regenerated in reports/figures and reports/tables.")
