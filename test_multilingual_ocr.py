"""
test_multilingual_ocr.py
Unit tests for multilingual OCR foundation, script detection, and model availability.

Tests:
1. English Unicode text is detected only as English.
2. Word "Telangana" inside English text does NOT trigger Telugu.
3. Telugu Unicode text is detected as Telugu.
4. Hindi/Devanagari Unicode text is detected as Devanagari / Hindi/Marathi unknown.
5. Mixed English/Telugu page reports both languages with per-line preservation.
6. Kannada, Tamil, and Urdu scripts are correctly classified.
7. Unsupported/unavailable language configuration produces needs_review=True and explicit warnings.
8. Value preservation without unverified transliteration (transliteration: null).
9. Page-level and line-level metadata conform to required schemas.
"""

import unittest
import numpy as np
from land_document_extractor import (
    OCRLine,
    detect_script_and_language,
    detect_page_language_metadata,
    format_multilingual_value,
    resolve_paddle_model_names,
    is_model_locally_available,
    get_paddle_ocr_model,
    run_paddle_ocr_page_image,
    ModelUnavailableError,
    SUPPORTED_LANGUAGE_CODES,
)


class TestMultilingualOCR(unittest.TestCase):

    def test_english_detected_only_as_english(self):
        """English text must be detected as Latin/English and not trigger Indic languages."""
        text = "GOVERNMENT OF INDIA REGISTRATION AND STAMPS DEPARTMENT SALE DEED"
        script, lang = detect_script_and_language(text)
        self.assertEqual(script, "Latin")
        self.assertEqual(lang, "English")

        lines = [
            OCRLine(text="SALE DEED DOCUMENT", score=0.98, x_min=100, y_min=100, x_max=500, y_max=130),
            OCRLine(text="Schedule of property situated at Hyderabad", score=0.95, x_min=100, y_min=200, x_max=600, y_max=230),
        ]
        meta = detect_page_language_metadata(lines, raw_text=" ".join(l.text for l in lines), ocr_model="PP-OCRv6_medium_rec")
        self.assertEqual(meta["detected_languages"], ["English"])
        self.assertEqual(meta["primary_language"], "English")
        self.assertEqual(meta["language_detection_method"], "unicode_script")
        self.assertEqual(meta["ocr_language_model"], "PP-OCRv6_medium_rec")

    def test_telangana_keyword_does_not_trigger_telugu(self):
        """The word 'Telangana' or 'Andhra Pradesh' inside English text must NOT trigger Telugu."""
        text = "Government of Telangana Registration and Stamps Department District Registrar Hyderabad Telangana State"
        script, lang = detect_script_and_language(text)
        self.assertEqual(script, "Latin")
        self.assertEqual(lang, "English")

        lines = [OCRLine(text=text, score=0.96, x_min=50, y_min=50, x_max=800, y_max=90)]
        meta = detect_page_language_metadata(lines, raw_text=text)
        self.assertIn("English", meta["detected_languages"])
        self.assertNotIn("Telugu", meta["detected_languages"])
        self.assertEqual(meta["primary_language"], "English")

    def test_telugu_script_detection(self):
        """Actual Telugu Unicode characters must be detected as Telugu."""
        telugu_text = "తెలంగాణ ప్రభుత్వం రిజిస్ట్రేషన్ శాఖ సేల్ డీడ్"
        script, lang = detect_script_and_language(telugu_text)
        self.assertEqual(script, "Telugu")
        self.assertEqual(lang, "Telugu")

        lines = [OCRLine(text=telugu_text, score=0.92, x_min=100, y_min=100, x_max=600, y_max=140)]
        meta = detect_page_language_metadata(lines, raw_text=telugu_text)
        self.assertIn("Telugu", meta["detected_languages"])
        self.assertEqual(meta["primary_language"], "Telugu")

    def test_devanagari_hindi_marathi_distinction(self):
        """Devanagari text must report 'Devanagari' and 'Hindi/Marathi unknown' unless context is provided."""
        devanagari_text = "विक्रय विलेख भारत सरकार भूमि अभिलेख"
        # Without context:
        script, lang = detect_script_and_language(devanagari_text)
        self.assertEqual(script, "Devanagari")
        self.assertEqual(lang, "Hindi/Marathi unknown")

        # With explicit Hindi context:
        script_hi, lang_hi = detect_script_and_language(devanagari_text, context_lang="hi")
        self.assertEqual(script_hi, "Devanagari")
        self.assertEqual(lang_hi, "Hindi")

        # With explicit Marathi context:
        script_mr, lang_mr = detect_script_and_language(devanagari_text, context_lang="mr")
        self.assertEqual(script_mr, "Devanagari")
        self.assertEqual(lang_mr, "Marathi")

    def test_kannada_tamil_urdu_detection(self):
        """Kannada, Tamil, and Urdu scripts must be accurately identified."""
        # Kannada
        kn_text = "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ"
        s_kn, l_kn = detect_script_and_language(kn_text)
        self.assertEqual(s_kn, "Kannada")
        self.assertEqual(l_kn, "Kannada")

        # Tamil
        ta_text = "தமிழ்நாடு அரசு நில ஆவணங்கள்"
        s_ta, l_ta = detect_script_and_language(ta_text)
        self.assertEqual(s_ta, "Tamil")
        self.assertEqual(l_ta, "Tamil")

        # Urdu / Arabic
        ur_text = "اراضی ریکارڈ دستاویز حکومت"
        s_ur, l_ur = detect_script_and_language(ur_text)
        self.assertEqual(s_ur, "Arabic")
        self.assertEqual(l_ur, "Urdu")

    def test_mixed_language_page_preservation(self):
        """Mixed English/Telugu page must detect both languages and tag each line with its own script."""
        l1 = OCRLine(text="SALE DEED", score=0.98, x_min=100, y_min=50, x_max=300, y_max=80)
        l2 = OCRLine(text="ఈ దస్తావేజు విక్రయదారుడు మరియు కొనుగోలుదారు మధ్య జరిగింది", score=0.91, x_min=100, y_min=120, x_max=800, y_max=160)
        l3 = OCRLine(text="Document Registered at S.R.O. Hyderabad", score=0.94, x_min=100, y_min=200, x_max=600, y_max=230)

        # Set line-level metadata
        for line in [l1, l2, l3]:
            s, l = detect_script_and_language(line.text)
            line.script = s
            line.language = l

        self.assertEqual(l1.language, "English")
        self.assertEqual(l1.script, "Latin")
        self.assertEqual(l2.language, "Telugu")
        self.assertEqual(l2.script, "Telugu")
        self.assertEqual(l3.language, "English")
        self.assertEqual(l3.script, "Latin")

        meta = detect_page_language_metadata([l1, l2, l3])
        self.assertIn("English", meta["detected_languages"])
        self.assertIn("Telugu", meta["detected_languages"])
        self.assertEqual(meta["language_detection_method"], "unicode_script")

    def test_model_resolution_through_paddle_api(self):
        """Model names must be resolved via the installed PaddleOCR API without hardcoded guesses."""
        langs = ["en", "te", "hi", "kn", "ta", "mr", "ur"]
        for l in langs:
            det, rec = resolve_paddle_model_names(l)
            self.assertIsNotNone(det, f"Failed to resolve detection model for '{l}'")
            self.assertIsNotNone(rec, f"Failed to resolve recognition model for '{l}'")
            self.assertTrue(len(det) > 0)
            self.assertTrue(len(rec) > 0)

        # Kannada maps to ka backend
        det_kn, rec_kn = resolve_paddle_model_names("kn")
        self.assertIn("ka", rec_kn)

    def test_local_availability_status(self):
        """English model is locally downloaded, while un-downloaded Indic models remain unavailable."""
        self.assertTrue(is_model_locally_available("en"))
        # Languages whose model weights are not downloaded locally
        self.assertFalse(is_model_locally_available("hi"))
        self.assertFalse(is_model_locally_available("ta"))
        self.assertFalse(is_model_locally_available("ur"))

    def test_unavailable_model_raises_or_flags_needs_review(self):
        """Requesting an unavailable model must not silently succeed with English; it must set needs_review=True."""
        # 1. Direct call raises ModelUnavailableError for un-downloaded model (e.g. 'hi')
        with self.assertRaises(ModelUnavailableError):
            get_paddle_ocr_model("hi")

        # 2. Page runner handles unavailable model with explicit warning and needs_review=True
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        lines, raw, timings = run_paddle_ocr_page_image(dummy_img, page_num=1, lang="hi")
        self.assertTrue(timings.get("needs_review"))
        self.assertEqual(timings.get("ocr_language_model_status"), "UNAVAILABLE")
        self.assertEqual(timings.get("unsupported_language"), "hi")
        self.assertIn("language_warning", timings)

    def test_format_multilingual_value_preservation(self):
        """Multilingual values must preserve original_value, script, language, and set transliteration: null."""
        val = format_multilingual_value("పి. శ్రీనివాస్ రెడ్డి")
        self.assertEqual(val["original_value"], "పి. శ్రీనివాస్ రెడ్డి")
        self.assertIsNone(val["transliteration"])
        self.assertEqual(val["script"], "Telugu")
        self.assertEqual(val["language"], "Telugu")

        val_hi = format_multilingual_value("राजेश कुमार", context_lang="hi")
        self.assertEqual(val_hi["original_value"], "राजेश कुमार")
        self.assertIsNone(val_hi["transliteration"])
        self.assertEqual(val_hi["script"], "Devanagari")
        self.assertEqual(val_hi["language"], "Hindi")


if __name__ == "__main__":
    unittest.main()
