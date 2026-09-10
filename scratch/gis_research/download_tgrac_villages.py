"""
Download and validate official TGRAC Master Administrative Boundary Village Layer (Layer 5)
using deterministic OBJECTID range pagination and package into data/gis/telangana/telangana_villages.geojson.gz
"""

import json
import gzip
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://tgrac.telangana.gov.in/arcgis/rest/services/Master_Administrative_Folder/Master_Administrative_Boundary_test/FeatureServer/5/query"

OUT_DIR = Path("data/gis/telangana")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GZ_PATH = OUT_DIR / "telangana_villages.geojson.gz"

def esri_ring_to_geojson_polygon(rings):
    """Converts Esri geometry rings to GeoJSON Polygon or MultiPolygon coordinates."""
    if not rings:
        return None
    if len(rings) == 1:
        return {
            "type": "Polygon",
            "coordinates": rings
        }
    else:
        return {
            "type": "MultiPolygon",
            "coordinates": [[ring] for ring in rings]
        }

def fetch_all_tgrac_villages():
    print("Starting deterministic OBJECTID download of TGRAC Master_Village_Boundary (Layer 5)...", flush=True)
    last_oid = 0
    batch_size = 1000
    all_features = []
    seen_oids = set()

    fields = [
        "objectid", "district", "mandal", "village",
        "district_code", "subdistrict_code", "village_code",
        "district_name_", "subdistrict_name", "village_name_",
        "village_status", "census_2011_code", "census_2001_code",
        "old_dist", "old_mandal", "revenue_division"
    ]
    out_fields_str = ",".join(fields)

    while True:
        url = (
            f"{BASE_URL}?where=objectid%3E{last_oid}"
            f"&outFields={out_fields_str}"
            f"&returnGeometry=true"
            f"&outSR=4326"
            f"&orderByFields=objectid+ASC"
            f"&resultRecordCount={batch_size}"
            f"&f=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                features = data.get("features", [])
                if not features:
                    print("No more features returned. Fetch complete.", flush=True)
                    break

                for feat in features:
                    attrs = feat.get("attributes", {})
                    oid = attrs.get("objectid")
                    if oid in seen_oids:
                        continue
                    seen_oids.add(oid)
                    last_oid = max(last_oid, oid)

                    geom = feat.get("geometry", {})
                    rings = geom.get("rings")
                    geojson_geom = esri_ring_to_geojson_polygon(rings)

                    if not geojson_geom:
                        continue

                    # Clean attributes
                    dist_name = (attrs.get("district") or attrs.get("district_name_") or "").strip()
                    man_name = (attrs.get("mandal") or attrs.get("subdistrict_name") or "").strip()
                    vill_name = (attrs.get("village") or attrs.get("village_name_") or "").strip()

                    all_features.append({
                        "type": "Feature",
                        "properties": {
                            "state": "TELANGANA",
                            "district": dist_name,
                            "mandal": man_name,
                            "village": vill_name,
                            "district_code": attrs.get("district_code"),
                            "subdistrict_code": attrs.get("subdistrict_code"),
                            "village_code": attrs.get("village_code"),
                            "census_2011_code": attrs.get("census_2011_code"),
                            "old_dist": attrs.get("old_dist"),
                            "old_mandal": attrs.get("old_mandal"),
                            "revenue_division": attrs.get("revenue_division"),
                            "village_status": attrs.get("village_status"),
                            "objectid": oid
                        },
                        "geometry": geojson_geom
                    })

                print(f"Batch last_oid {last_oid}: fetched {len(features)} records ({time.time()-t0:.2f}s). Total unique so far: {len(all_features)}", flush=True)
                if len(features) < batch_size:
                    print("Reached final record page.", flush=True)
                    break
        except Exception as e:
            print(f"Error fetching after OID {last_oid}: {e}", flush=True)
            time.sleep(2)

    print(f"\nTotal Unique TGRAC Village Features Processed: {len(all_features)}", flush=True)

    geojson_doc = {
        "type": "FeatureCollection",
        "name": "TGRAC_Telangana_Master_Village_Boundaries",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "state": "TELANGANA",
        "source": "TGRAC (Telangana State Remote Sensing Applications Centre) Master_Village_Boundary (Layer 5)",
        "features": all_features
    }

    # Write Gzip compressed production asset
    print(f"Writing compressed asset to {GZ_PATH}...", flush=True)
    json_bytes = json.dumps(geojson_doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with gzip.open(GZ_PATH, "wb", compresslevel=9) as gz_out:
        gz_out.write(json_bytes)

    gz_size = GZ_PATH.stat().st_size / (1024 * 1024)
    raw_size = len(json_bytes) / (1024 * 1024)
    print(f"Done! Raw size: {raw_size:.2f} MB | Compressed size: {gz_size:.2f} MB (Compression: {(1 - gz_size/raw_size)*100:.1f}%)", flush=True)

if __name__ == "__main__":
    fetch_all_tgrac_villages()
