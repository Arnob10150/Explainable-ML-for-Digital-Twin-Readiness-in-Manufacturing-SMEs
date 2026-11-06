"""Run the optional independent-country cohort validation."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# The analysis module writes its result when imported.
from dtreadiness import external_validation  # noqa: F401,E402
