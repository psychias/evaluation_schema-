#!/usr/bin/env python3
"""Validate LaTeX structure."""
import re, os

text = open("submission/main.tex").read()

# Brace balance
opens = text.count("{")
closes = text.count("}")
print(f"Braces: open={opens}, close={closes}, balanced={opens == closes}")

# Environment matching
begins = re.findall(r"\\begin\{(\w+)\}", text)
ends = re.findall(r"\\end\{(\w+)\}", text)
from collections import Counter
bc = Counter(begins)
ec = Counter(ends)
ok = True
for env in set(list(bc.keys()) + list(ec.keys())):
    if bc[env] != ec[env]:
        print(f"  MISMATCH: {env} begins={bc[env]} ends={ec[env]}")
        ok = False
if ok:
    print("All environments balanced:")
    for env, cnt in sorted(bc.items()):
        print(f"  {env}: {cnt}")

# Labels and refs
labels = re.findall(r"\\label\{([^}]+)\}", text)
refs = re.findall(r"\\ref\{([^}]+)\}", text)
print(f"\nLabels ({len(labels)}):")
for l in labels:
    print(f"  {l}")
missing = [r for r in set(refs) if r not in labels]
if missing:
    print(f"MISSING LABELS for refs: {missing}")
else:
    print("All refs have matching labels")

# Figures
figs = re.findall(r"includegraphics.*?\{([^}]+)\}", text)
print(f"\nFigures ({len(figs)}):")
for f in figs:
    path = os.path.join("submission", f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = f"{size:,} bytes" if exists else "MISSING!"
    print(f"  {f}: {status}")
