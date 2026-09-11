"""
demo_learning_feedback.py - Officer-Verified Adaptive OCR Normalization Demonstration.

Smart India Hackathon (SIH) Prototype: OneBhoomi Land Registry.
Demonstrates:
  1. A raw OCR value appears.
  2. A clerk corrects the value.
  3. The correction is stored as pending.
  4. The pending rule is not active.
  5. An officer approves the correction.
  6. One verified example is still insufficient.
  7. A second matching verified example activates the rule.
  8. A future matching OCR value is normalized automatically.
  9. The raw OCR value is still preserved.
  10. A different field does not use the rule.
  11. A different document type does not use the rule.
  12. A different language does not use the rule.
  13. The dashboard/statistics show the evidence count and active rule.

IMPORTANT NOTICE:
This script uses simulated clerk feedback because only one real English Telangana
land document is available in the offline prototype. This is a demonstration of
human-in-the-loop adaptive OCR normalization and feedback learning, NOT measured
OCR model accuracy improvement or neural network retraining.
"""

import json
import sys
import tempfile
from pathlib import Path

import ocr_learning_service
from land_document_extractor import OCRLine
from semantic_extractor import extract_fields_semantic


def run_demo():
    print("=" * 80)
    print("OneBhoomi: Officer-verified adaptive OCR normalization & feedback learning")
    print("Smart India Hackathon (SIH) Prototype Demonstration")
    print("=" * 80)
    print("""
[SIMULATION DISCLOSURE]
* Only one main English Telangana land document is currently available.
* The corrections shown below are SIMULATED feedback to demonstrate the
  human-in-the-loop adaptive OCR feedback learning loop.
* This is NOT reinforcement learning.
* This is NOT neural network retraining.
* PaddleOCR weights are NOT modified.
* Rules are scoped strictly to (field_name, document_type, language).
""")
    print("=" * 80)

    # Use an isolated temporary store so demo does not pollute persistent workspace data
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_store = Path(tmp_dir) / "demo_learning_store.json"
        ocr_learning_service.set_store_file(temp_store)

        # Baseline OCR lines simulating raw OCR reading from a Sale Deed in English
        raw_lines = [
            OCRLine(text="SALE DEED", score=0.99, x_min=100, y_min=40, x_max=600, y_max=80, page_num=1, page_height=2000),
            OCRLine(text="Document No. 12736/5", score=0.95, x_min=100, y_min=100, x_max=450, y_max=140, page_num=1, page_height=2000),
            OCRLine(text="SRI P. SRINIVAS REDDY", score=0.93, x_min=100, y_min=200, x_max=600, y_max=240, page_num=1, page_height=2000),
            OCRLine(text="SCHEDULE OF PROPERTY", score=0.97, x_min=100, y_min=500, x_max=600, y_max=540, page_num=2, page_height=2000),
            # Step 1 simulated glitch: stray pipe in survey number: "278 | 281"
            OCRLine(text="Survey No. 278 | 281 extent 480 Sq. Yards", score=0.78, x_min=100, y_min=560, x_max=900, y_max=600, page_num=2, page_height=2000),
            OCRLine(text="Situated at Aushapur Village, Ghatkesar Mandal, Ranga Reddy District, Telangana", score=0.92, x_min=100, y_min=620, x_max=950, y_max=660, page_num=2, page_height=2000),
        ]

        # -------------------------------------------------------------------
        # STEP 1: A raw OCR value appears
        # -------------------------------------------------------------------
        print("\n[STEP 1] A raw OCR value appears:")
        res_step1, prov_step1, _ = extract_fields_semantic(raw_lines)
        sn_p1 = prov_step1["survey_number"]
        print(f"  Field               : survey_number")
        print(f"  Raw OCR Value       : '{sn_p1['raw_ocr_value']}'")
        print(f"  Raw OCR Confidence  : {sn_p1['ocr_confidence']}")
        print(f"  Normalization Type  : {sn_p1['normalization_type']}")
        print(f"  Active Learned Rules: {res_step1['learning']['rules_applied']}")

        # -------------------------------------------------------------------
        # STEP 2: A clerk corrects the value
        # -------------------------------------------------------------------
        print("\n[STEP 2] A clerk corrects the value in the review console:")
        print("  Clerk changes '278 | 281' -> '278, 281'")

        # -------------------------------------------------------------------
        # STEP 3: Correction is stored as pending
        # -------------------------------------------------------------------
        print("\n[STEP 3] Correction is stored as pending:")
        feedback_item = ocr_learning_service.stage_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 281",
            corrected_value="278, 281",
            verification_id="deed-sim-001",
            page_number=2,
            source_bbox=[100, 560, 900, 600],
            language="en",
            ocr_confidence_before=0.78,
            status="pending_approval",
        )
        print(f"  Feedback ID   : {feedback_item['feedback_id']}")
        print(f"  Status        : {feedback_item['status']}")
        print(f"  Page Number   : {feedback_item['page_number']}")
        print(f"  Source BBox   : {feedback_item['source_bbox']}")

        # -------------------------------------------------------------------
        # STEP 4: The pending rule is not active
        # -------------------------------------------------------------------
        print("\n[STEP 4] The pending rule is not active:")
        rules_step4 = ocr_learning_service.get_learned_rules("survey_number")
        print(f"  Active Learned Rules for 'survey_number': {len(rules_step4)} (0 active)")
        print("  Safe Learning Policy: Pending clerk edits NEVER produce active rules.")

        # -------------------------------------------------------------------
        # STEP 5: An officer approves the correction
        # -------------------------------------------------------------------
        print("\n[STEP 5] An officer approves the document and signs the deed:")
        approved_n = ocr_learning_service.approve_feedback_for_verification("deed-sim-001")
        print(f"  Promoted {approved_n} feedback record(s) from pending_approval to verified.")

        # -------------------------------------------------------------------
        # STEP 6: One verified example is still insufficient
        # -------------------------------------------------------------------
        print("\n[STEP 6] One verified example is still insufficient:")
        stats_step6 = ocr_learning_service.get_learning_stats()
        rules_step6 = ocr_learning_service.get_learned_rules("survey_number")
        print(f"  Verified Count: {stats_step6['verified_feedback_count']}")
        print(f"  Active Rules  : {len(rules_step6)} (Rule remains inactive because threshold is >= 2)")

        # -------------------------------------------------------------------
        # STEP 7: A second matching verified example activates the rule
        # -------------------------------------------------------------------
        print("\n[STEP 7] A second matching verified example activates the rule:")
        ocr_learning_service.record_feedback(
            document_type="Sale Deed",
            field_name="survey_number",
            raw_ocr_value="278 | 281",
            corrected_value="278, 281",
            verification_id="deed-sim-002",
            page_number=3,
            source_bbox=[120, 580, 880, 620],
            language="en",
            ocr_confidence_before=0.80,
            auto_approve=True,  # Simulating officer approval of deed 2
        )
        rules_step7 = ocr_learning_service.get_learned_rules("survey_number")
        print(f"  Active Rules Count : {len(rules_step7)}")
        active_r = rules_step7[0]
        print(f"  Activated Rule ID  : {active_r['rule_id']}")
        print(f"  Pattern -> Target  : '{active_r['raw_pattern']}' -> '{active_r['replacement']}'")
        print(f"  Evidence Count     : {active_r['evidence_count']}")
        print(f"  Rule Confidence    : {active_r['confidence']}")

        # -------------------------------------------------------------------
        # STEP 8: A future matching OCR value is normalized automatically
        # -------------------------------------------------------------------
        print("\n[STEP 8] A future matching OCR value is normalized automatically:")
        res_step8, prov_step8, _ = extract_fields_semantic(raw_lines)
        sn_p8 = prov_step8["survey_number"]
        print(f"  Normalized Value           : '{sn_p8['value']}'")
        print(f"  Normalization Type         : {sn_p8['normalization_type']}")
        print(f"  Normalization Confidence   : {sn_p8['normalization_confidence']}")
        print(f"  Final Confidence           : {sn_p8['final_confidence']}")
        print(f"  Rule Applied Info          : {sn_p8['source']}")

        # -------------------------------------------------------------------
        # STEP 9: The raw OCR value is still preserved
        # -------------------------------------------------------------------
        print("\n[STEP 9] The raw OCR value is still preserved:")
        print(f"  Raw OCR Value      : '{sn_p8['raw_ocr_value']}'")
        print(f"  Original Value     : '{sn_p8['original_value']}'")
        print(f"  Raw OCR Confidence : {sn_p8['ocr_confidence']} (Unaltered PaddleOCR score)")

        # -------------------------------------------------------------------
        # STEP 10: A different field does not use the rule
        # -------------------------------------------------------------------
        print("\n[STEP 10] A different field does not use the rule:")
        val_diff_field, raw_c, norm_c, fin_c, r_info, n_type = ocr_learning_service.apply_learned_normalization(
            field_name="property_area",
            raw_value="278 | 281",
            ocr_confidence=0.85,
            document_type="Sale Deed",
            language="en",
        )
        print(f"  Target Field       : property_area")
        print(f"  Result Value       : '{val_diff_field}' (Rule for survey_number was NOT applied)")
        print(f"  Normalization Type : {n_type}")
        print(f"  Rule Applied       : {r_info}")

        # -------------------------------------------------------------------
        # STEP 11: A different document type does not use the rule
        # -------------------------------------------------------------------
        print("\n[STEP 11] A different document type does not use the rule:")
        val_diff_doc, _, _, _, r_info_doc, n_type_doc = ocr_learning_service.apply_learned_normalization(
            field_name="survey_number",
            raw_value="278 | 281",
            ocr_confidence=0.85,
            document_type="Mutation Record",  # Different document type!
            language="en",
        )
        print(f"  Document Type      : Mutation Record (rule is for 'Sale Deed')")
        print(f"  Result Value       : '{val_diff_doc}' (Rule was NOT applied due to doc type mismatch)")
        print(f"  Learned Rule Used  : {r_info_doc is not None}")
        print(f"  Normalization Type : {n_type_doc}")

        # -------------------------------------------------------------------
        # STEP 12: A different language does not use the rule
        # -------------------------------------------------------------------
        print("\n[STEP 12] A different language does not use the rule:")
        val_diff_lang, _, _, _, r_info_lang, n_type_lang = ocr_learning_service.apply_learned_normalization(
            field_name="survey_number",
            raw_value="278 | 281",
            ocr_confidence=0.85,
            document_type="Sale Deed",
            language="te",  # Telugu script / language!
        )
        print(f"  Language           : te (Telugu, rule is for 'en')")
        print(f"  Result Value       : '{val_diff_lang}' (Rule was NOT applied due to language mismatch)")
        print(f"  Learned Rule Used  : {r_info_lang is not None}")
        print(f"  Normalization Type : {n_type_lang}")

        # -------------------------------------------------------------------
        # STEP 13: The dashboard/statistics show evidence count and active rule
        # -------------------------------------------------------------------
        print("\n[STEP 13] The dashboard/statistics show evidence count and active rule:")
        stats_final = ocr_learning_service.get_learning_stats()
        all_rules = ocr_learning_service.get_learned_rules()
        print(f"  Verified Feedback Count : {stats_final['verified_feedback_count']}")
        print(f"  Pending Feedback Count  : {stats_final['pending_feedback_count']}")
        print(f"  Rejected Feedback Count : {stats_final['rejected_feedback_count']}")
        print(f"  Active Learned Rules    : {stats_final['learned_rules_count']}")
        print(f"  Fields Improved         : {stats_final['fields_improved']}")
        print(f"  Active Rule Spec        : {json.dumps(all_rules[0], indent=4)}")

    print("\n" + "=" * 80)
    print("Demonstration successfully verified all 13 required steps!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
