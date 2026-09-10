import json

fpath = 'scratch/gis_research/ka_full.geojson'

with open(fpath, 'r', encoding='utf-8') as f:
    data = json.load(f)

features = data.get('features', [])

print('=== Pandavapura Villages with D ===')
for f in features:
    props = f.get('properties', {})
    dist = str(props.get('DISTRICT') or props.get('DIST_NAME') or '').lower()
    taluk = str(props.get('TALUK') or props.get('TALUKA_NAM') or '').lower()
    v_name = str(props.get('NAME') or props.get('VILL_NAME') or '')
    code = props.get('LOC_CODE') or props.get('VILL_CODE9')
    
    if 'mandya' in dist and 'pandavapura' in taluk and v_name.lower().startswith('d'):
        print(f'  Pandavapura: {v_name} (Code: {code})')

print('\n=== Maddur Villages with D ===')
for f in features:
    props = f.get('properties', {})
    dist = str(props.get('DISTRICT') or props.get('DIST_NAME') or '').lower()
    taluk = str(props.get('TALUK') or props.get('TALUKA_NAM') or '').lower()
    v_name = str(props.get('NAME') or props.get('VILL_NAME') or '')
    code = props.get('LOC_CODE') or props.get('VILL_CODE9')
    
    if 'mandya' in dist and 'maddur' in taluk and v_name.lower().startswith('d'):
        print(f'  Maddur: {v_name} (Code: {code})')
