#!/usr/bin/env python3
"""
onboard_state.py - OneBhoomi CLI to Validate and Onboard New State GIS Datasets.

Validates new state GeoJSON datasets against the exact structural contract expected
by OneBhoomi's offline spatial GIS subsystem, and safely installs them into data/gis/<state>/.

Usage Examples:
    python onboard_state.py --state demo_state --input data/gis/demo_state/demo_state_villages.geojson
    python onboard_state.py --state demo_state --input data/gis/demo_state/ --layer both
    python onboard_state.py --state demo_state --input dataset.geojson --layer villages --force
"""

import argparse
import gzip
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data" / "gis"
ALLOWED_GEOMETRIES = {"Point", "Polygon", "MultiPolygon"}
ALLOWED_LEVELS = {"state", "district", "mandal", "taluk"}


def validate_geojson_feature(feature: Any, index: int, layer_type: str) -> List[str]:
    """Validates a single GeoJSON Feature against layer-specific schema rules."""
    errors: List[str] = []
    if not isinstance(feature, dict):
        return [f"ERROR: Feature {index} is not a valid JSON object."]

    if feature.get("type") != "Feature":
        errors.append(f"ERROR: Feature {index} has invalid type '{feature.get('type')}'; expected 'Feature'.")

    geom = feature.get("geometry")
    if not isinstance(geom, dict):
        errors.append(f"ERROR: Feature {index} has missing or invalid 'geometry' object.")
    else:
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype not in ALLOWED_GEOMETRIES:
            errors.append(f"ERROR: Feature {index} has unsupported geometry type '{gtype}'; expected one of {sorted(ALLOWED_GEOMETRIES)}.")
        if not coords:
            errors.append(f"ERROR: Feature {index} geometry has empty 'coordinates'.")

    props = feature.get("properties")
    if not isinstance(props, dict):
        errors.append(f"ERROR: Feature {index} has missing or non-dict 'properties'.")
        return errors

    if layer_type == "villages":
        has_village_name = bool(props.get("village") or props.get("name"))
        if not has_village_name:
            errors.append(f"ERROR: Feature {index} is missing required village name property ('village' or 'name').")

        has_subdistrict = bool(props.get("mandal") or props.get("taluk") or props.get("old_mandal"))
        if not has_subdistrict:
            errors.append(f"ERROR: Feature {index} is missing required subdistrict property ('mandal' or 'taluk').")

        has_district = bool(props.get("district") or props.get("old_dist"))
        if not has_district:
            errors.append(f"ERROR: Feature {index} is missing required district property ('district').")

    elif layer_type == "registry":
        level = (props.get("level") or "").lower()
        if level not in ALLOWED_LEVELS:
            errors.append(f"ERROR: Feature {index} property 'level'='{props.get('level')}' is invalid; expected one of {sorted(ALLOWED_LEVELS)}.")

        has_name = bool(props.get("name") or props.get("district") or props.get("mandal") or props.get("taluk"))
        if not has_name:
            errors.append(f"ERROR: Feature {index} is missing name identifier property ('name', 'district', or 'mandal').")

    return errors


def validate_geojson_content(content_bytes: bytes, filename: str, layer_type: str) -> Tuple[bool, List[str], int]:
    """Validates raw GeoJSON / Gzipped GeoJSON bytes."""
    errors: List[str] = []
    try:
        if filename.endswith(".gz"):
            text = gzip.decompress(content_bytes).decode("utf-8")
        else:
            text = content_bytes.decode("utf-8")
        data = json.loads(text)
    except Exception as e:
        return False, [f"ERROR: Failed to parse JSON from {filename}: {e}"], 0

    if not isinstance(data, dict):
        return False, [f"ERROR: {filename} root is not a JSON object."], 0

    if data.get("type") != "FeatureCollection":
        return False, [f"ERROR: {filename} root 'type' must be 'FeatureCollection', got '{data.get('type')}'."], 0

    features = data.get("features")
    if not isinstance(features, list):
        return False, [f"ERROR: {filename} 'features' property must be a list."], 0

    if len(features) == 0:
        return False, [f"ERROR: {filename} FeatureCollection contains 0 features."], 0

    seen_ids = set()
    for idx, f in enumerate(features):
        feat_errors = validate_geojson_feature(f, idx, layer_type)
        errors.extend(feat_errors)

        if isinstance(f, dict):
            p = f.get("properties") or {}
            fid = f.get("id") or p.get("village_code") or p.get("shapeID")
            if fid:
                if fid in seen_ids:
                    errors.append(f"ERROR: Duplicate feature identifier '{fid}' at index {idx}.")
                seen_ids.add(fid)

    return len(errors) == 0, errors, len(features)


def validate_dataset_file(file_path: Any, layer_type: str = "villages", layer: Optional[str] = None) -> Tuple[bool, List[str], int]:
    """Validates a dataset file on disk against the layer schema."""
    if layer is not None:
        layer_type = layer
    p = Path(file_path)
    if not p.exists():
        return False, [f"ERROR: File not found: {p}"], 0
    try:
        raw_b = p.read_bytes()
    except Exception as e:
        return False, [f"ERROR: Failed to read file {p}: {e}"], 0
    return validate_geojson_content(raw_b, p.name, layer_type)


def onboard_state(
    state_name: str,
    input_path: str,
    layer_type: str = "auto",
    force: bool = False,
    base_gis_dir: Optional[Path] = None,
) -> int:
    """Validates and onboards state dataset into data/gis/<state_name>/."""
    state_key = state_name.strip().lower()
    if not state_key or not state_key.replace("_", "").isalnum():
        print(f"ERROR: Invalid state identifier '{state_name}'. Use alphanumeric characters and underscores.")
        return 1

    in_path = Path(input_path)
    if not in_path.exists():
        print(f"ERROR: Input path '{input_path}' does not exist.")
        return 1

    root_dir = base_gis_dir if base_gis_dir is not None else DATA_DIR
    target_dir = root_dir / state_key
    if target_dir.exists() and not force:
        # Check if already has files
        existing_files = list(target_dir.glob("*.json*")) + list(target_dir.glob("*.geojson*"))
        if existing_files:
            print(f"ERROR: State directory '{target_dir}' already exists and contains datasets: {[f.name for f in existing_files]}")
            print("Use --force to overwrite existing dataset.")
            return 1

    files_to_process: List[Tuple[Path, str]] = []

    if in_path.is_dir():
        for p in sorted(in_path.iterdir()):
            name_lower = p.name.lower()
            if "villages" in name_lower and (name_lower.endswith(".geojson") or name_lower.endswith(".geojson.gz") or name_lower.endswith(".json")):
                files_to_process.append((p, "villages"))
            elif "spatial_registry" in name_lower and name_lower.endswith(".json"):
                files_to_process.append((p, "registry"))
    else:
        # Single file
        inferred_layer = layer_type
        if inferred_layer == "auto":
            if "villages" in in_path.name.lower():
                inferred_layer = "villages"
            elif "spatial_registry" in in_path.name.lower() or "registry" in in_path.name.lower():
                inferred_layer = "registry"
            else:
                inferred_layer = "villages"
        files_to_process.append((in_path, inferred_layer))

    if not files_to_process:
        print(f"ERROR: No valid GeoJSON files found at '{input_path}'.")
        return 1

    # Validate all files before writing anything
    validated_files: List[Tuple[Path, str, bytes, int]] = []
    has_errors = False

    print(f"\nValidating dataset for state: '{state_key}'...")
    for fpath, ltype in files_to_process:
        print(f"  Checking {fpath.name} (layer: {ltype})...")
        raw_b = fpath.read_bytes()
        ok, errs, count = validate_geojson_content(raw_b, fpath.name, ltype)
        if not ok:
            has_errors = True
            for err in errs[:15]:
                print(f"    {err}")
            if len(errs) > 15:
                print(f"    ... and {len(errs) - 15} more errors.")
        else:
            print(f"    Validation PASS: {count} valid features.")
            validated_files.append((fpath, ltype, raw_b, count))

    if has_errors:
        print(f"\nSchema validation: FAILED for state '{state_key}'. No files were installed.")
        return 1

    # Copy files to destination
    target_dir.mkdir(parents=True, exist_ok=True)
    total_features = 0

    for src_path, ltype, content, count in validated_files:
        total_features += count
        if ltype == "villages":
            dest_name = f"{state_key}_villages{src_path.suffix if src_path.suffix in ('.gz', '.geojson') else '.geojson'}"
        else:
            dest_name = f"{state_key}_spatial_registry.json"

        dest_path = target_dir / dest_name
        dest_path.write_bytes(content)
        print(f"  Installed: {dest_path}")

    print(f"\nSchema validation: PASS")
    print(f"State: {state_key}")
    print(f"Features: {total_features}")
    print(f"Output: {target_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OneBhoomi GIS State Dataset Onboarding Tool")
    parser.add_argument("--state", required=True, help="State identifier (e.g. demo_state, maharashtra)")
    parser.add_argument("--input", required=True, help="Input GeoJSON file or dataset directory")
    parser.add_argument("--layer", choices=["auto", "villages", "registry"], default="auto", help="Layer type")
    parser.add_argument("--force", action="store_true", help="Overwrite destination without prompt")
    args = parser.parse_args()

    return onboard_state(
        state_name=args.state,
        input_path=args.input,
        layer_type=args.layer,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
