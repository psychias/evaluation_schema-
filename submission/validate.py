#!/usr/bin/env python3
import re, glob

tex = open('main.tex').read()

# Brace balance
opens = tex.count('{')
closes = tex.count('}')
print(f'Braces: {opens} open, {closes} close, balanced={opens==closes}')

# Labels and refs
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set(re.findall(r'\\ref\{([^}]+)\}', tex))
unresolved = refs - labels
print(f'Labels: {len(labels)}, Refs: {len(refs)}, Unresolved: {unresolved if unresolved else "none"}')

# Citations
cites = set()
for m in re.findall(r'\\cite[tp]?\{([^}]+)\}', tex):
    for c in m.split(','):
        cites.add(c.strip())
for m in re.findall(r'\\citealt\{([^}]+)\}', tex):
    for c in m.split(','):
        cites.add(c.strip())
print(f'Unique citations: {len(cites)}')

# Bib keys
bib_keys = set()
for bf in glob.glob('*.bib'):
    with open(bf) as f:
        for line in f:
            m2 = re.match(r'@\w+\{(\S+),', line)
            if m2:
                bib_keys.add(m2.group(1))
missing = cites - bib_keys
print(f'Bib keys: {len(bib_keys)}, Missing citations: {missing if missing else "none"}')
print(f'Lines: {len(tex.splitlines())}')
