#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from run_reviewer_geometric_median_matched_v4332 import adapter_self_test
adapter_self_test()
print("GEOMETRIC MEDIAN MAPPING ADAPTER SELF TEST = PASS")
print("HISTORICAL MAX ITER = 60")
print("HISTORICAL TOLERANCE = 1e-7")
