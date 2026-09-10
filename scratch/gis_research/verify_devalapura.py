import json
import re

fpath = 'scratch/gis_research/ka.geojson'

with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Search for all occurrences of Devala/Deva in property blocks
matches = re.findall(r'"properties"\s*:\s*(\{[^}]+\})', text)
print('Total property objects:', len(matches))

found_devala = []
for m in matches:
    if 'deval' in m.lower() or 'devlap' in m.lower():
        found_devala.append(m)

print(f'Devalapura matches found: {len(found_devala)}')
for f in found_devala[:10]:
    print('  -', f)
