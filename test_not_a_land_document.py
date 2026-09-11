import unittest
import json
import verification_service
import web_app


class TestNotALandDocument(unittest.TestCase):

    def test_empty_document_detected_as_not_land_document(self):
        empty_payload = {
            "document_type": None,
            "document_number": None,
            "parties": [],
            "property": {
                "survey_number": None,
                "sub_survey_number": None,
                "area": None,
                "village": None,
                "mandal": None,
                "district": None,
            },
            "stamp_information": {
                "stamp_number": None,
                "stamp_value": None,
                "sold_to": None,
            },
            "document_date": None,
            "execution_date": None,
        }

        is_land_doc, populated = verification_service.check_is_land_document(empty_payload)
        self.assertFalse(is_land_doc)
        self.assertEqual(len(populated), 0)

        checks = verification_service.run_verification_checks(empty_payload)
        classification_check = next((c for c in checks if c.get("check_id") == "land_document_classification"), None)
        self.assertIsNotNone(classification_check)
        self.assertEqual(classification_check["status"], "FAIL")
        self.assertIn("This is not a land document", classification_check["message"])

        status = verification_service.calculate_overall_status(checks)
        self.assertEqual(status, "FAIL")

    def test_valid_land_document_recognized(self):
        valid_payload = {
            "document_type": "Sale Deed",
            "document_number": "5121/2002",
            "parties": [{"name": "Rao", "role": "Vendor"}],
            "property": {
                "survey_number": "278, 281",
                "sub_survey_number": "1023/1",
                "area": "480",
                "village": "Aushapur",
                "mandal": "Ghatkesar",
                "district": "Ranga Reddy",
            },
            "stamp_information": {
                "stamp_number": "12104",
                "stamp_value": "100",
                "sold_to": "Smt. Suvarna",
            },
            "document_date": "09-10-2003",
            "execution_date": "09-10-2003",
        }

        is_land_doc, populated = verification_service.check_is_land_document(valid_payload)
        self.assertTrue(is_land_doc)
        self.assertGreaterEqual(len(populated), 5)

        checks = verification_service.run_verification_checks(valid_payload)
        classification_check = next((c for c in checks if c.get("check_id") == "land_document_classification"), None)
        self.assertIsNotNone(classification_check)
        self.assertEqual(classification_check["status"], "PASS")

    def test_ui_renders_error_banner_when_not_land_document(self):
        empty_payload = {
            "document_type": None,
            "document_number": None,
            "parties": [],
            "property": {},
            "stamp_information": {},
            "document_date": None,
            "execution_date": None,
        }
        rec = verification_service.create_verification_record(empty_payload)
        html_bytes = web_app.render_page(
            payload="{}",
            message="Error: This is not a land document. All land registry fields are empty.",
            active_record=rec,
        )
        html_str = html_bytes.decode("utf-8")
        self.assertIn("ERROR: THIS IS NOT A LAND DOCUMENT", html_str)
        self.assertIn("This is not a land document. All of the required land registry fields are empty.", html_str)
        self.assertIn("not-land-doc-alert-box", html_str)
        self.assertIn("not-land-doc-alert", html_str)


if __name__ == "__main__":
    unittest.main()
