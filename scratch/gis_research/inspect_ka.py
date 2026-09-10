import json
import re

fpath = 'scratch/gis_research/ka.geojson'
with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

matches = re.findall(r'"properties"\s*:\s*\{[^\}]+\}', text)
print('Total property blocks matched:', len(matches))
for i, m in enumerate(matches[:5]):
    print(f'Feature {i+1}: {m}')
