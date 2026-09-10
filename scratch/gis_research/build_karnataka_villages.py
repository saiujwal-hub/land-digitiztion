import json
import os

fpath = 'scratch/gis_research/ka_full.geojson'
out_path = 'data/gis/karnataka/karnataka_villages.geojson'

print('Extracting Karnataka village geometries from raw DataMeet source...')
with open(fpath, 'r', encoding='utf-8') as f:
    data = json.load(f)

features = data.get('features', [])

karnataka_village_features = []

for feat in features:
    props = feat.get('properties', {})
    v_name = props.get('NAME') or props.get('VILL_NAME')
    t_name = props.get('TALUK') or props.get('TALUKA_NAM')
    d_name = props.get('DISTRICT') or props.get('DIST_NAME') or props.get('DISTRICT_N')
    loc_code = props.get('LOC_CODE') or props.get('VILL_CODE9')

    if v_name and d_name and t_name:
        karnataka_village_features.append({
            'type': 'Feature',
            'properties': {
                'level': 'village',
                'state': 'Karnataka',
                'district': str(d_name).strip(),
                'taluk': str(t_name).strip(),
                'village': str(v_name).strip(),
                'name': str(v_name).strip(),
                'census_loc_code': str(loc_code).strip() if loc_code else None,
                'source': 'DataMeet Indian Village Boundaries (ODbL / CC-BY 4.0)'
            },
            'geometry': feat.get('geometry')
        })

print(f'Extracted {len(karnataka_village_features)} real Karnataka village features.')

village_dataset = {
    'type': 'FeatureCollection',
    'state': 'KARNATAKA',
    'dataset_metadata': {
        'provider': 'DataMeet Indian Village Boundaries Community',
        'source_url': 'https://github.com/datameet/indian_village_boundaries/tree/master/ka',
        'license': 'Open Database License (ODbL 1.0) / CC-BY 4.0',
        'attribution': 'DataMeet Community (datameet.org)',
        'crs': 'EPSG:4326 (WGS 84)',
        'acquisition_date': '2026-09-03',
        'original_feature_count': len(features),
        'bundled_feature_count': len(karnataka_village_features)
    },
    'features': karnataka_village_features
}

os.makedirs('data/gis/karnataka', exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(village_dataset, f, indent=2)

print(f'Saved production Karnataka village dataset to {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB).')
