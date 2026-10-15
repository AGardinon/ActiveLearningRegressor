"""compare.py <runA> <runB>: exact comparison of all CSV outputs of two benchmark runs.

Prints 'identical' or 'DIFFERENT' per file; for DIFFERENT files with the same
shape and columns, also prints the largest absolute difference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

a, b = Path(sys.argv[1]), Path(sys.argv[2])
fa = sorted(p.name for p in a.glob("*.csv"))
fb = sorted(p.name for p in b.glob("*.csv"))
if fa != fb:
    print("  FILE SETS DIFFER:", set(fa) ^ set(fb))

for f in fa:
    if f not in fb:
        continue
    da, db = pd.read_csv(a / f), pd.read_csv(b / f)
    same_layout = da.shape == db.shape and list(da.columns) == list(db.columns)
    same = same_layout and da.equals(db)
    msg = f"  {f:45s} {'identical' if same else 'DIFFERENT'}  {da.shape} vs {db.shape}"
    if same_layout and not same:
        num = da.select_dtypes("number").columns
        if len(num):
            msg += f"  max|diff| = {np.nanmax(np.abs(da[num].values - db[num].values)):.3g}"
    print(msg)
