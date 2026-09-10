"""
Comprehensive unit test suite for DataMeet & TGRAC Offline GIS Resolver (Final Integration Pass).
"""

import json
import socket
import unittest
from pathlib import Path
from gis_service import verify_gis_location, parse_dimensions_from_text, get_local_dataset


class TestDataMeetGISResolver(unittest.TestCase):

    def test_1_karnataka_district_resolution(self):
        sample_ocr = {"state": "Karnataka", "property": {"district": "Mandya"}}
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["district"], "Mandya")
        self.assertEqual(res["resolution_level"], "district")

    def test_2_karnataka_taluk_resolution(self):
        sample_ocr = {"state": "Karnataka", "property": {"district": "Mandya", "taluk": "Pandavapura"}}
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["mandal"], "Pandavapura")
        self.assertEqual(res["resolution_level"], "mandal")

    def test_3_karnataka_pandavapura_taluk(self):
        sample_ocr = {"state": "Karnataka", "property": {"district": "Mandya", "taluk": "Pandavapura"}}
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["mandal"], "Pandavapura")

    def test_4_karnataka_maddur_taluk(self):
        sample_ocr = {"state": "Karnataka", "property": {"district": "Mandya", "taluk": "Maddur"}}
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["mandal"], "Maddur")

    def test_5_karnataka_devalapura_nagamangala(self):
        sample_ocr = {
            "state": "Karnataka",
            "property": {
                "district": "Mandya",
                "mandal": "Nagamangala",
                "village": "Devalapura"
            }
        }
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["resolution_level"], "village")
        self.assertEqual(res["village_status"], "RESOLVED")
        self.assertEqual(res["gis_resolution_confidence"], 90)

    def test_6_karnataka_devalapura_tumkur(self):
        sample_ocr = {
            "state": "Karnataka",
            "property": {
                "district": "Tumkur",
                "mandal": "Tumkur",
                "village": "Devalapura"
            }
        }
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["resolution_level"], "village")
        self.assertEqual(res["district"], "Tumkur")

    def test_7_duplicate_village_disambiguation(self):
        # Disambiguates Tumkur vs Nagamangala Devalapura
        res_naga = verify_gis_location({
            "state": "Karnataka",
            "property": {"district": "Mandya", "mandal": "Nagamangala", "village": "Devalapura"}
        })
        res_tumk = verify_gis_location({
            "state": "Karnataka",
            "property": {"district": "Tumkur", "mandal": "Tumkur", "village": "Devalapura"}
        })
        self.assertNotEqual(res_naga["latitude"], res_tumk["latitude"])

    def test_8_actual_village_polygon_returned(self):
        sample_ocr = {
            "state": "Karnataka",
            "property": {"district": "Mandya", "mandal": "Nagamangala", "village": "Devalapura"}
        }
        res = verify_gis_location(sample_ocr)
        geom = res.get("administrative_geometry")
        self.assertIsNotNone(geom)
        self.assertIn(geom.get("type"), ("Polygon", "MultiPolygon", "Point"))

    def test_9_centroid_calculated_from_geometry(self):
        sample_ocr = {
            "state": "Karnataka",
            "property": {"district": "Mandya", "mandal": "Nagamangala", "village": "Devalapura"}
        }
        res = verify_gis_location(sample_ocr)
        self.assertIsNotNone(res.get("latitude"))
        self.assertIsNotNone(res.get("longitude"))

    def test_10_no_known_centroids_fallback(self):
        import gis_service
        self.assertFalse(hasattr(gis_service, "KNOWN_CENTROIDS"))

    def test_11_telangana_district_resolution(self):
        sample_ocr = {"state": "Telangana", "property": {"district": "Sangareddy"}}
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["district"], "Sangareddy")
        self.assertEqual(res["resolution_level"], "district")

    def test_12_telangana_kandi_mandal_resolution(self):
        sample_ocr = {"state": "Telangana", "property": {"district": "Sangareddy", "mandal": "Kandi"}}
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["mandal"], "Kandi")
        self.assertEqual(res["resolution_level"], "mandal")

    def test_13_telangana_village_resolution_tgrac(self):
        sample_ocr = {
            "state": "Telangana",
            "property": {"district": "Sangareddy", "mandal": "Kandi", "village": "Kandi"}
        }
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["resolution_level"], "village")
        self.assertEqual(res["village_status"], "RESOLVED")
        self.assertEqual(res["gis_resolution_confidence"], 90)
        self.assertIn("TGRAC", res.get("source_attribution", ""))

    def test_14_cadastral_status_not_available(self):
        sample_ocr = {"state": "Karnataka", "property": {"district": "Mandya", "survey_number": "45"}}
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["cadastral_status"], "NOT_AVAILABLE")
        self.assertIsNone(res["survey_geometry"])

    def test_15_dimension_rectangle_aspect_ratio(self):
        parsed = parse_dimensions_from_text("East-West = 60 ft, North-South = 40 ft")
        self.assertIsNotNone(parsed)
        ew_m, ns_m, unit = parsed
        self.assertAlmostEqual(ew_m, 18.288, places=2)
        self.assertAlmostEqual(ns_m, 12.192, places=2)

    def test_16_offline_resolution_zero_network(self):
        original_socket = socket.socket
        try:
            socket.socket = None
            sample_ocr = {
                "state": "Telangana",
                "property": {"district": "Warangal", "mandal": "Geesugonda", "village": "Kommala"}
            }
            res = verify_gis_location(sample_ocr)
            self.assertEqual(res["status"], "resolved")
            self.assertEqual(res["resolution_level"], "village")
        finally:
            socket.socket = original_socket

    def test_17_invalid_village_no_guessing(self):
        sample_ocr = {
            "state": "Karnataka",
            "property": {"district": "Mandya", "mandal": "Maddur", "village": "NonExistentVillageXYZ"}
        }
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["resolution_level"], "mandal")
        self.assertEqual(res["village_status"], "NOT_AVAILABLE")

    def test_18_ambiguous_village_no_guessing(self):
        sample_ocr = {
            "state": "Karnataka",
            "property": {"village": "Devalapura"} # No district/taluk context
        }
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "ambiguous")
        auth_val = res.get("authority_validation", {})
        self.assertEqual(auth_val.get("hierarchy_status"), "AMBIGUOUS")

    def test_19_compressed_dataset_loading(self):
        dataset = get_local_dataset("KARNATAKA", layer="villages")
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.get("state"), "KARNATAKA")
        self.assertGreater(len(dataset.get("features", [])), 25000)

    def test_20_sources_documentation_file(self):
        sources_path = Path("data/gis/SOURCES.md")
        self.assertTrue(sources_path.exists())
        content = sources_path.read_text(encoding="utf-8")
        self.assertIn("geoBoundaries", content)
        self.assertIn("DataMeet", content)
        self.assertIn("TGRAC", content)

    def test_21_telangana_compressed_dataset_loading(self):
        dataset = get_local_dataset("TELANGANA", layer="villages")
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.get("state"), "TELANGANA")
        self.assertEqual(len(dataset.get("features", [])), 10906)

    def test_22_contradictory_telangana_hierarchy(self):
        # Telangana -> Jangaon -> Geesugonda -> Mangapet is contradictory
        sample_ocr = {
            "state": "Telangana",
            "property": {
                "district": "Jangaon",
                "mandal": "Geesugonda",
                "village": "Mangapet"
            }
        }
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["resolution_level"], "district") # Safely falls back to district level
        self.assertEqual(res["district"], "Jangaon")
        auth_val = res.get("authority_validation", {})
        self.assertEqual(auth_val.get("national_registry_status"), "NOT_AVAILABLE")
        self.assertEqual(auth_val.get("state_registry_status"), "VALIDATED")
        self.assertEqual(auth_val.get("hierarchy_status"), "CONTRADICTORY")
        self.assertIn("Contradictory", res.get("village_disclaimer", ""))

    def test_23_valid_telangana_hierarchy_authority(self):
        # Telangana -> Warangal -> Geesugonda -> Kommala is valid
        sample_ocr = {
            "state": "Telangana",
            "property": {
                "district": "Warangal",
                "mandal": "Geesugonda",
                "village": "Kommala"
            }
        }
        res = verify_gis_location(sample_ocr)
        self.assertEqual(res["status"], "resolved")
        self.assertEqual(res["resolution_level"], "village")
        auth_val = res.get("authority_validation", {})
        self.assertEqual(auth_val.get("national_registry_status"), "NOT_AVAILABLE")
        self.assertEqual(auth_val.get("state_registry_status"), "VALIDATED")
        self.assertEqual(auth_val.get("hierarchy_status"), "CONSISTENT")
        self.assertIn("TGRAC", res.get("source_attribution", ""))


if __name__ == "__main__":
    unittest.main()
