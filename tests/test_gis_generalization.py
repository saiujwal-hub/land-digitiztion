#!/usr/bin/env python3
"""
tests/test_gis_generalization.py - Test suite for GIS Architecture Generalization & Onboarding.

Verifies:
1. Existing Telangana resolution path through gis_service.verify_gis_location.
2. Existing Karnataka resolution path through gis_service.verify_gis_location.
3. Third-state (demo_state) successful resolution through the SAME code path.
4. Flexible name formatting normalization (e.g. 'Demo State', 'demo_state', 'DEMO_STATE').
5. Unknown state rejection without guessing (status: UNSUPPORTED_STATE).
6. Substring collision safety (avoiding accidental false-positive matching).
7. Dynamic future-state discovery (new state directory in data/gis/ recognized immediately).
8. Positive and negative onboarding CLI validations (onboard_state.py).
9. Dynamic read-only admin GIS page rendering.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import gis_service
import dashboard_view
from onboard_state import validate_dataset_file, onboard_state


class TestGISGeneralization(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TASK 3 & TASK 9: Resolution paths & Third-State Proof
    # -------------------------------------------------------------------------

    def test_1_telangana_existing_resolution_path(self):
        """Confirm Telangana resolves via verify_gis_location with TGRAC attribution."""
        payload = {
            "state": "Telangana",
            "property": {
                "district": "Sangareddy",
                "mandal": "Kandi",
                "village": "Kandi"
            }
        }
        res = gis_service.verify_gis_location(payload)
        self.assertEqual(res.get("status"), "resolved")
        self.assertEqual(res.get("resolution_level"), "village")
        self.assertIn("TGRAC", res.get("source_attribution", ""))

    def test_2_karnataka_existing_resolution_path(self):
        """Confirm Karnataka resolves via verify_gis_location with DataMeet attribution."""
        payload = {
            "state": "Karnataka",
            "property": {
                "district": "Mandya",
                "mandal": "Nagamangala",
                "village": "Devalapura"
            }
        }
        res = gis_service.verify_gis_location(payload)
        self.assertEqual(res.get("status"), "resolved")
        self.assertEqual(res.get("resolution_level"), "village")
        self.assertIn("DataMeet", res.get("source_attribution", ""))

    def test_3_third_state_resolution_success(self):
        """
        Confirm that calling verify_gis_location for demo_state succeeds through
        the exact SAME GIS resolution path, returns village level resolution,
        calculates centroid, and clearly attributes synthetic demo data.
        """
        payload = {
            "state": "demo_state",
            "property": {
                "district": "Central District",
                "mandal": "Sunrise Mandal",
                "village": "Green Valley"
            }
        }
        res = gis_service.verify_gis_location(payload)
        self.assertEqual(res.get("status"), "resolved")
        self.assertEqual(res.get("resolution_level"), "village")
        self.assertEqual(res.get("village_status"), "RESOLVED")
        self.assertEqual(res.get("district"), "Central District")
        self.assertEqual(res.get("mandal"), "Sunrise Mandal")
        self.assertEqual(res.get("village"), "Green Valley")
        self.assertIsNotNone(res.get("latitude"))
        self.assertIsNotNone(res.get("longitude"))
        self.assertIn("SYNTHETIC / DEMO DATA", res.get("source_attribution", ""))
        self.assertIn("SYNTHETIC / DEMO DATA", res.get("village_disclaimer", ""))

    def test_4_demo_state_variant_names(self):
        """Confirm that variant name forms of the state resolve to the same canonical dataset."""
        variants = ["Demo State", "demo state", "demo_state", "DEMO_STATE"]
        for var in variants:
            payload = {
                "state": var,
                "property": {
                    "district": "Central District",
                    "mandal": "Sunrise Mandal",
                    "village": "Green Valley"
                }
            }
            res = gis_service.verify_gis_location(payload)
            self.assertEqual(res.get("status"), "resolved", f"Failed for variant: {var}")
            self.assertEqual(res.get("resolution_level"), "village", f"Failed for variant: {var}")

    def test_5_unknown_state_rejected(self):
        """Confirm unknown or non-configured states return UNSUPPORTED_STATE."""
        unknown_states = ["NonExistentStateXYZ", "Atlantis", "Singapore", "New York"]
        for unk in unknown_states:
            payload = {
                "state": unk,
                "property": {
                    "district": "Some District",
                    "village": "Some Village"
                }
            }
            res = gis_service.verify_gis_location(payload)
            self.assertEqual(res.get("status"), "UNSUPPORTED_STATE", f"Failed for state: {unk}")
            self.assertIn("not supported in offline GIS registry", res.get("message", ""))

    def test_6_substring_collision_safety(self):
        """Confirm names containing substrings like 'ap' do not falsely map to AP/Telangana."""
        safe_states = ["AppleState", "Japan", "Singapore", "Maputo"]
        for s in safe_states:
            canon = gis_service.normalize_state(s)
            self.assertIsNone(canon, f"State '{s}' should not resolve to any dataset, got {canon}")

    def test_7_dynamic_future_state_discovery(self):
        """
        Confirm that adding a new state directory to data/gis/ makes it immediately
        discoverable and resolvable without editing any Python code.
        """
        future_state_key = "test_dyn_state"
        future_dir = Path("data/gis") / future_state_key
        try:
            future_dir.mkdir(parents=True, exist_ok=True)
            # Create a minimal valid spatial registry
            registry_content = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[75.0, 15.0], [75.1, 15.0], [75.1, 15.1], [75.0, 15.0]]]
                        },
                        "properties": {
                            "level": "district",
                            "district": "Future District"
                        }
                    }
                ]
            }
            (future_dir / f"{future_state_key}_spatial_registry.json").write_text(
                json.dumps(registry_content), encoding="utf-8"
            )

            # Test normalize_state dynamically discovers the new state
            resolved_key = gis_service.normalize_state("Test Dyn State")
            self.assertEqual(resolved_key, future_state_key.upper())

            # Test verify_gis_location resolves the new district
            payload = {
                "state": "Test Dyn State",
                "property": {
                    "district": "Future District"
                }
            }
            res = gis_service.verify_gis_location(payload)
            self.assertEqual(res.get("status"), "resolved")
            self.assertEqual(res.get("resolution_level"), "district")
            self.assertEqual(res.get("district"), "Future District")

        finally:
            # Clean up temporary test state directory
            if future_dir.exists():
                shutil.rmtree(future_dir, ignore_errors=True)
            # Clear caches
            gis_service._DATASET_CACHE.clear()

    # -------------------------------------------------------------------------
    # TASK 8 & TASK 10: onboard_state.py Validation & Negative Tests
    # -------------------------------------------------------------------------

    def test_8_onboard_valid_synthetic_dataset(self):
        """Confirm onboard_state successfully validates synthetic demo_state datasets."""
        village_file = Path("data/gis/demo_state/demo_state_villages.geojson")
        self.assertTrue(village_file.exists(), "demo_state_villages.geojson must exist")
        
        valid, errors, count = validate_dataset_file(village_file, layer="villages")
        self.assertTrue(valid, f"Validation failed with errors: {errors}")
        self.assertEqual(count, 4)

        registry_file = Path("data/gis/demo_state/demo_state_spatial_registry.json")
        self.assertTrue(registry_file.exists(), "demo_state_spatial_registry.json must exist")

        valid_reg, errors_reg, count_reg = validate_dataset_file(registry_file, layer="registry")
        self.assertTrue(valid_reg, f"Validation failed with errors: {errors_reg}")
        self.assertEqual(count_reg, 5)

    def test_9_negative_validation_missing_required_property(self):
        """Confirm validator rejects features missing required attributes."""
        invalid_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[78.0, 17.0], [78.1, 17.0], [78.1, 17.1], [78.0, 17.0]]]
                    },
                    "properties": {
                        # Missing 'village' or 'name' property
                        "district": "Test District",
                        "mandal": "Test Mandal"
                    }
                }
            ]
        }
        bad_file = Path(self.temp_dir) / "missing_prop.geojson"
        bad_file.write_text(json.dumps(invalid_data), encoding="utf-8")

        valid, errors, count = validate_dataset_file(bad_file, layer="villages")
        self.assertFalse(valid)
        self.assertTrue(any("missing required village name property" in e for e in errors))

    def test_10_negative_validation_invalid_geometry_type(self):
        """Confirm validator rejects unsupported geometry types for spatial boundaries."""
        invalid_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",  # LineString is not a valid boundary polygon
                        "coordinates": [[78.0, 17.0], [78.1, 17.1]]
                    },
                    "properties": {
                        "level": "district",
                        "district": "Test District"
                    }
                }
            ]
        }
        bad_file = Path(self.temp_dir) / "invalid_geom.geojson"
        bad_file.write_text(json.dumps(invalid_data), encoding="utf-8")

        valid, errors, count = validate_dataset_file(bad_file, layer="registry")
        self.assertFalse(valid)
        self.assertTrue(any("unsupported geometry type 'LineString'" in e for e in errors))

    def test_11_negative_validation_malformed_geojson(self):
        """Confirm validator rejects corrupt or non-JSON files."""
        bad_file = Path(self.temp_dir) / "corrupted.geojson"
        bad_file.write_text("{ this is not valid json! }", encoding="utf-8")

        valid, errors, count = validate_dataset_file(bad_file, layer="villages")
        self.assertFalse(valid)
        self.assertTrue(any("Failed to parse JSON" in e for e in errors))

    def test_12_negative_validation_not_a_feature_collection(self):
        """Confirm validator rejects files where root type is not FeatureCollection."""
        invalid_data = {
            "type": "Point",
            "coordinates": [78.0, 17.0]
        }
        bad_file = Path(self.temp_dir) / "not_fc.geojson"
        bad_file.write_text(json.dumps(invalid_data), encoding="utf-8")

        valid, errors, count = validate_dataset_file(bad_file, layer="villages")
        self.assertFalse(valid)
        self.assertTrue(any("FeatureCollection" in e for e in errors))

    def test_13_onboard_overwrite_protection(self):
        """Confirm onboard_state refuses to overwrite existing files unless force=True."""
        valid_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[78.0, 17.0], [78.1, 17.0], [78.1, 17.1], [78.0, 17.0]]]
                    },
                    "properties": {
                        "village": "Test Village",
                        "district": "Test District",
                        "mandal": "Test Mandal"
                    }
                }
            ]
        }
        src_file = Path(self.temp_dir) / "sample_villages.geojson"
        src_file.write_text(json.dumps(valid_data), encoding="utf-8")

        # Fake target state dir in temp_dir
        dest_state_dir = Path(self.temp_dir) / "test_state"
        dest_state_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_state_dir / "test_state_villages.geojson"
        dest_file.write_text("existing content", encoding="utf-8")

        # Calling onboard_state without force should return 1 (error) and not overwrite
        res_code = onboard_state("test_state", str(src_file), layer_type="villages", base_gis_dir=Path(self.temp_dir), force=False)
        self.assertEqual(res_code, 1)
        self.assertEqual(dest_file.read_text(encoding="utf-8"), "existing content")

        # Calling onboard_state with force=True should return 0 (success) and overwrite
        res_code_force = onboard_state("test_state", str(src_file), layer_type="villages", base_gis_dir=Path(self.temp_dir), force=True)
        self.assertEqual(res_code_force, 0)
        self.assertIn("Test Village", dest_file.read_text(encoding="utf-8"))

    # -------------------------------------------------------------------------
    # TASK 12: Read-only GIS Admin Page
    # -------------------------------------------------------------------------

    def test_14_admin_gis_page_renders_dynamically(self):
        """Confirm /admin/gis page dynamically renders configured states including demo_state."""
        page_bytes = dashboard_view.render_admin_gis_page(host_name="testserver:8001", user_name="Admin")
        page_str = page_bytes.decode("utf-8")

        self.assertIn("Configured GIS State Repositories", page_str)
        self.assertIn("Telangana", page_str)
        self.assertIn("Karnataka", page_str)
        self.assertIn("Demo State", page_str)
        self.assertIn("SYNTHETIC / DEMO DATA", page_str)
        self.assertIn("Read-Only Inspection Mode", page_str)


if __name__ == "__main__":
    unittest.main()
