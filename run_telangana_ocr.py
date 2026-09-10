"""
Run OCR and calibrated generic semantic extraction on C:\\Users\\meesa\\Downloads\\telangana_official_land_document.pdf
Saves per-page raw OCR, confidences, bounding boxes, and comprehensive structured extraction.
"""

import os
# Configure OpenMP thread pool for optimal CPU performance
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("HUB_DATASET_ENDPOINT", "https://modelscope.cn/api/v1/datasets")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

import sys
import json
import time
import re
import inspect
from pathlib import Path
import pypdfium2 as pdfium
import cv2
import numpy as np

# Ensure current workspace is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from land_document_extractor import (
    run_paddle_ocr_page_image,
    run_remote_ocr_page_image,
    OCRLine,
    detect_page_language_metadata,
    detect_script_and_language,
    SUPPORTED_LANGUAGE_CODES,
    is_model_locally_available,
)
from semantic_extractor import (
    extract_fields_semantic,
    clean_user_facing_schema,
    _norm,
)

PDF_PATH = r"C:\Users\meesa\Downloads\telangana_official_land_document.pdf"
OUT_JSON = BASE_DIR / "telangana_document_extraction.json"

def validate_ocr_results(total_pages: int, pages_ocr: list, failed_pages: list | None = None, strict: bool = False) -> tuple[bool, list[str]]:
    """
    Validate that:
    1. total_ocr_pages equals total_pages for a successful run.
    2. processed_page_numbers contains every page from 1 to total_pages exactly once.
    3. failed_pages is empty.

    Returns:
        (is_valid, error_messages)
    """
    errors = []
    failed = failed_pages or []
    if failed:
        errors.append(f"Processing failed for pages: {[f.get('page_number') for f in failed]}")

    total_ocr_pages = len(pages_ocr)
    if total_ocr_pages != total_pages:
        errors.append(f"total_ocr_pages ({total_ocr_pages}) != total_pages ({total_pages})")

    processed_page_numbers = [p.get("page_number") for p in pages_ocr]
    expected_page_numbers = list(range(1, total_pages + 1))

    if len(processed_page_numbers) != len(set(processed_page_numbers)):
        errors.append(f"Duplicate page numbers found in processed_page_numbers: {processed_page_numbers}")

    if sorted(processed_page_numbers) != expected_page_numbers:
        errors.append(f"processed_page_numbers {processed_page_numbers} does not match expected sequence {expected_page_numbers}")

    is_valid = len(errors) == 0
    if strict and not is_valid:
        raise ValueError(f"OCR validation failed: {'; '.join(errors)}")
    return is_valid, errors

def parse_args(argv=None):
    """
    Parse command-line arguments.
    - --from-cache loads existing JSON cache.
    - --reextract forces re-extraction from PDF.
    - With no flag, process the PDF again by default.
    - --reextract is never treated as a cache-loading flag.
    """
    if argv is None:
        argv = sys.argv[1:]

    from_cache = "--from-cache" in argv
    reextract = "--reextract" in argv
    use_cache = from_cache and not reextract

    pdf_path = PDF_PATH
    out_json = OUT_JSON
    lang = "auto"
    ocr_url = os.environ.get("OCR_SERVER_URL")

    for i, arg in enumerate(argv):
        if arg == "--pdf" and i + 1 < len(argv):
            pdf_path = argv[i + 1]
        elif arg == "--out" and i + 1 < len(argv):
            out_json = argv[i + 1]
        elif arg == "--lang" and i + 1 < len(argv):
            lang = argv[i + 1].lower()
        elif arg == "--ocr-url" and i + 1 < len(argv):
            ocr_url = argv[i + 1]
        elif arg == "--remote":
            if not ocr_url:
                txt_path = BASE_DIR / "colab_url.txt"
                if txt_path.exists():
                    ocr_url = txt_path.read_text(encoding="utf-8").strip()

    return {
        "use_cache": use_cache,
        "from_cache": from_cache,
        "reextract": reextract,
        "pdf_path": pdf_path,
        "out_json": out_json,
        "lang": lang,
        "ocr_url": ocr_url,
    }

def locate_field_provenance(val, prov_entry, field_name, all_lines=None, force_needs_review=None):
    if all_lines is None:
        all_lines = []
    if prov_entry is None:
        prov_entry = {}
    status = prov_entry.get("status", "EXTRACTED" if val is not None else "NOT_FOUND")
    needs_rev = prov_entry.get("needs_review", True if val is None else False)
    if force_needs_review is not None:
        needs_rev = force_needs_review

    if not val or status == "NOT_FOUND":
        return {
            "field_name": field_name,
            "value": None,
            "status": "NOT_FOUND",
            "confidence": 0.0,
            "needs_review": True,
            "candidates": [],
            "conflicting_candidates": [],
            "evidence": [],
            "page_number": None,
            "source_text": None,
            "bounding_box": None,
            "original_ocr_value": None,
            "correction_applied": False,
        }

    cand_str = str(val).lower()
    matched_line = None

    # Look in all_lines for direct match or substring
    for l in all_lines:
        if cand_str in l.text.lower():
            matched_line = l
            break

    # Fallback search by tokens
    if not matched_line and isinstance(val, str):
        tokens = [t.lower() for t in re.split(r"[^\w]+", val) if len(t) > 2]
        if tokens:
            best_match = None
            best_overlap = 0
            for l in all_lines:
                l_lower = l.text.lower()
                overlap = sum(1 for t in tokens if t in l_lower)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = l
            if best_overlap > 0:
                matched_line = best_match

    page = matched_line.page_num if matched_line else None
    bbox = [matched_line.x_min, matched_line.y_min, matched_line.x_max, matched_line.y_max] if matched_line else None
    source_text = matched_line.text if matched_line else prov_entry.get("source")

    # Dynamic honest confidence calculation
    base_ocr_score = float(matched_line.score) if matched_line else 0.85
    semantic_conf = float(prov_entry.get("confidence", base_ocr_score))
    corr_applied = bool(prov_entry.get("correction_applied", False))

    needs_rev = bool(prov_entry.get("needs_review", False))
    conf = (base_ocr_score * 0.6) + (semantic_conf * 0.4)
    if corr_applied or "repair" in str(source_text).lower():
        conf *= 0.82
        corr_applied = True
        needs_rev = True
    if needs_rev:
        conf = min(conf, 0.78)
    if status == "CONFLICT":
        conf = min(conf * 0.75, 0.65)
        needs_rev = True
    if not matched_line:
        conf = min(conf * 0.80, 0.70)
        needs_rev = True

    # Audit for visibly corrupted OCR or suspicious fragments
    val_str = str(val or "")
    if field_name in ("stamp_sold_to", "vendor_owner", "purchaser", "village", "mandal_tehsil", "locality_or_address", "layout_name"):
        if re.search(r"[A-Za-z0-9]/[A-Za-z0-9]|is/|s/o|w/o|d/o|r/o|c/o|[@\\_~]", val_str, re.IGNORECASE):
            needs_rev = True
            conf = min(conf, 0.65)
        if re.search(r"\b(?:ROGIGTRAR|OFTICIO|VENDOT|VENDOC|SRAMP|VONDOT|PAGISTTAR|SKINTVAS|REDDT)\b", val_str.upper()):
            needs_rev = True
            conf = min(conf, 0.65)

    conf = min(0.98, max(0.20, conf))

    evidence = prov_entry.get("evidence", [])
    if not evidence and source_text:
        evidence = [source_text]

    orig_val = prov_entry.get("original_value") or (matched_line.text if matched_line else str(val))
    cands = prov_entry.get("candidates") or ([val] if val is not None else [])
    conflicts = prov_entry.get("conflicting_candidates", [])

    return {
        "field_name": field_name,
        "value": val,
        "status": status,
        "confidence": round(float(conf), 4),
        "needs_review": bool(needs_rev),
        "candidates": cands,
        "conflicting_candidates": conflicts,
        "page_number": page,
        "source_text": source_text,
        "bounding_box": bbox,
        "original_ocr_value": orig_val,
        "correction_applied": corr_applied,
        "evidence": evidence,
    }


def locate_party_provenance(party_obj, role_name, field_name, all_lines=None):
    if all_lines is None:
        all_lines = []
    if not party_obj or not party_obj.get("name"):
        return {
            "field_name": field_name,
            "value": None,
            "status": "NOT_FOUND",
            "confidence": 0.0,
            "needs_review": True,
            "candidates": [],
            "conflicting_candidates": [],
            "evidence": [],
            "page_number": None,
            "source_text": None,
            "bounding_box": None,
            "original_ocr_value": None,
            "correction_applied": False,
            "role": role_name,
        }
    p_name = party_obj.get("name")
    rep = party_obj.get("represented_by")
    rel = party_obj.get("relation")

    extra_desc = []
    if rep:
        extra_desc.append(f"Represented by: {rep}")
    if rel:
        extra_desc.append(rel)
    full_desc = f"{p_name} ({', '.join(extra_desc)})" if extra_desc else p_name

    matched = None
    for l in all_lines:
        if p_name.lower() in l.text.lower():
            matched = l
            break
    if not matched and rep:
        for l in all_lines:
            if rep.lower() in l.text.lower():
                matched = l
                break
    if not matched:
        tokens = [t.lower() for t in re.split(r"[^\w]+", p_name) if len(t) > 3]
        if tokens:
            best_match = None
            best_ov = 0
            for l in all_lines:
                ov = sum(1 for t in tokens if t in l.text.lower())
                if ov > best_ov:
                    best_ov = ov
                    best_match = l
            if best_ov > 0:
                matched = best_match

    line_score = float(matched.score) if matched else 0.70
    party_conf = party_obj.get("confidence")
    if party_conf is not None:
        conf = party_conf
    elif not matched:
        conf = 0.60
    else:
        conf = min(0.98, max(0.50, line_score))

    corr_applied = bool(party_obj.get("correction_applied", False))
    needs_rev = bool(party_obj.get("needs_review", True if not matched or conf < 0.85 else False))
    if corr_applied:
        conf = min(conf * 0.85, 0.78)
        needs_rev = True
    elif needs_rev:
        conf = min(conf, 0.82)
    page = party_obj.get("page_number") or (matched.page_num if matched else None)
    bbox = [matched.x_min, matched.y_min, matched.x_max, matched.y_max] if matched else None
    source_text = party_obj.get("raw_text") or (matched.text if matched else p_name)
    orig_val = party_obj.get("original_ocr_value") or source_text
    corr_applied = bool(party_obj.get("correction_applied", False))
    cands = party_obj.get("candidates") or [full_desc]
    conflicts = party_obj.get("conflicting_candidates") or []
    status = party_obj.get("status", "CONFLICT" if conflicts else "EXTRACTED")
    evidence = party_obj.get("evidence") or ([source_text] if source_text else [])

    return {
        "field_name": field_name,
        "value": full_desc,
        "status": status,
        "confidence": round(float(conf), 4),
        "needs_review": bool(needs_rev),
        "candidates": cands,
        "conflicting_candidates": conflicts,
        "page_number": page,
        "source_text": source_text,
        "bounding_box": bbox,
        "original_ocr_value": orig_val,
        "correction_applied": corr_applied,
        "evidence": evidence,
        "role": role_name,
        "details": party_obj,
    }


def process_document(
    pdf_path: str | Path = PDF_PATH,
    out_json: str | Path = OUT_JSON,
    use_cache: bool = False,
    lang: str = "auto",
    ocr_url: str | None = None,
) -> dict:
    pdf_path = Path(pdf_path)
    out_json = Path(out_json)
    failed_pages = []
    total_time_s = 0.0
    pages_sent: list[int] = []
    pages_received: list[int] = []
    remote_failures: list[dict[str, Any]] = []
    fallback_used: bool = False

    if use_cache:
        print(f"Loading cached OCR results from {out_json}...")
        if not os.path.exists(out_json):
            raise FileNotFoundError(f"Cache file not found at {out_json}")
        with open(out_json, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        cached_meta = cached_data.get("metadata", {})
        total_time_s = cached_meta.get("total_processing_time_s", 0.0)
        total_pages = cached_meta.get("total_pages", len(cached_data.get("pages_ocr", [])))
        pages_ocr_results = cached_data.get("pages_ocr", [])
        failed_pages = cached_meta.get("failed_pages", [])
        pages_sent = cached_meta.get("pages_sent", [p["page_number"] for p in pages_ocr_results])
        pages_received = cached_meta.get("pages_received", [p["page_number"] for p in pages_ocr_results])
        remote_failures = cached_meta.get("remote_failures", [])
        fallback_used = cached_meta.get("fallback_used", False)
        all_lines = []
        for p in pages_ocr_results:
            pg_num = p["page_number"]
            for elem in p["ocr_elements"]:
                bb = elem.get("bounding_box") or elem.get("bbox")
                all_lines.append(OCRLine(
                    text=elem["text"],
                    score=elem["confidence"],
                    x_min=bb[0],
                    y_min=bb[1],
                    x_max=bb[2],
                    y_max=bb[3],
                    page_num=pg_num,
                    language=elem.get("language", "English"),
                    script=elem.get("script", "Latin"),
                ))
        print(f"Loaded {len(all_lines)} OCR lines from {len(pages_ocr_results)} pages.")
    else:
        print(f"Opening PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f"Error: PDF not found at {pdf_path}")
            sys.exit(1)

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            total_pages = len(pdf)
            print(f"Document has {total_pages} pages. Language strategy: '{lang}'")

            pages_ocr_results = []
            all_lines: list[OCRLine] = []

            total_start = time.perf_counter()

            for idx in range(total_pages):
                page_num = idx + 1
                print(f"\n--- Processing Page {page_num}/{total_pages} ---")
                p_start = time.perf_counter()

                try:
                    # Render at 1.5x scale (optimized for high speed and crisp character recognition)
                    t_render_0 = time.perf_counter()
                    pil_img = pdf[idx].render(scale=1.5).to_pil()
                    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]
                    render_ms = (time.perf_counter() - t_render_0) * 1000

                    ocr_start = time.perf_counter()
                    if ocr_url:
                        pages_sent.append(page_num)
                        lines, raw_text, timings = run_remote_ocr_page_image(
                            bgr, page_num=page_num, lang=lang, ocr_url=ocr_url
                        )
                        model_status = timings.get("ocr_language_model_status") or timings.get("model_status")
                        if model_status == "UNAVAILABLE" or (not lines and timings.get("warnings")):
                            err_msg = "; ".join(timings.get("warnings", [f"Remote OCR failed with status {model_status}"]))
                            remote_failures.append({"page_number": page_num, "error": err_msg})
                            # Strict: do not silently fall back to local OCR
                            raise RuntimeError(f"Remote OCR failed on page {page_num}: {err_msg}")
                        pages_received.append(page_num)
                    else:
                        target_fn = getattr(run_paddle_ocr_page_image, "side_effect", None) or run_paddle_ocr_page_image
                        try:
                            sig = inspect.signature(target_fn)
                            accepts_lang = "lang" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                        except Exception:
                            accepts_lang = True

                        if accepts_lang:
                            lines, raw_text, timings = run_paddle_ocr_page_image(bgr, page_num=page_num, lang=lang)
                        else:
                            lines, raw_text, timings = run_paddle_ocr_page_image(bgr, page_num=page_num)
                    page_ocr_time_ms = (time.perf_counter() - ocr_start) * 1000
                    page_total_time_ms = (time.perf_counter() - p_start) * 1000

                    # Extract per-line bounding boxes, script, and language
                    page_elements = []
                    scores = []
                    for line in lines:
                        scores.append(line.score)
                        page_elements.append({
                            "text": line.text,
                            "language": getattr(line, "language", "English"),
                            "script": getattr(line, "script", "Latin"),
                            "confidence": round(float(line.score), 4),
                            "bbox": [line.x_min, line.y_min, line.x_max, line.y_max],
                            "bounding_box": [line.x_min, line.y_min, line.x_max, line.y_max],
                            "page": page_num,
                        })

                    avg_conf = round(float(np.mean(scores)), 4) if scores else 0.0

                    page_lang_meta = detect_page_language_metadata(
                        lines,
                        raw_text,
                        ocr_model=timings.get("model_name")
                    )

                    model_status_val = timings.get("ocr_language_model_status") or timings.get("model_status", "AVAILABLE")
                    page_record = {
                        "page_number": page_num,
                        "image_dimensions": {"width": w, "height": h},
                        "processing_time_ms": round(page_total_time_ms, 2),
                        "pdf_render_ms": round(render_ms, 2),
                        "ocr_inference_ms": round(page_ocr_time_ms, 2),
                        "line_count": len(lines),
                        "average_confidence": avg_conf,
                        "raw_text": raw_text,
                        "model_status": model_status_val,
                        "detected_languages": page_lang_meta["detected_languages"],
                        "primary_language": page_lang_meta["primary_language"],
                        "language_detection_method": page_lang_meta["language_detection_method"],
                        "ocr_language_model": page_lang_meta["ocr_language_model"],
                        "ocr_elements": page_elements,
                    }
                    if timings.get("needs_review"):
                        page_record["needs_review"] = True
                    if timings.get("ocr_language_model_status"):
                        page_record["ocr_language_model_status"] = timings["ocr_language_model_status"]
                    if timings.get("unsupported_language"):
                        page_record["unsupported_language"] = timings["unsupported_language"]
                    if timings.get("language_warning"):
                        page_record["language_warning"] = timings["language_warning"]

                    pages_ocr_results.append(page_record)
                    all_lines.extend(lines)

                    print(f"Page {page_num} completed in {page_total_time_ms/1000:.2f}s (OCR: {page_ocr_time_ms/1000:.2f}s, Render: {render_ms:.1f}ms): {len(lines)} lines, avg conf {avg_conf:.3f}, lang: {page_lang_meta['primary_language']}")
                    first_few = [l.text for l in lines[:5]]
                    print(f"Page {page_num} sample lines: {first_few}")
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    failed_pages.append({
                        "page_number": page_num,
                        "error": str(e),
                    })

            total_time_s = round(time.perf_counter() - total_start, 2)
            print(f"\nAll {total_pages} pages processed in {total_time_s}s. Total lines: {len(all_lines)}")
        finally:
            pdf.close()

    # Validation before saving
    is_complete, val_errors = validate_ocr_results(total_pages, pages_ocr_results, failed_pages)
    if not is_complete:
        print(f"\n[VALIDATION WARNING] OCR run did not achieve complete processing:")
        for err in val_errors:
            print(f"  - {err}")
    else:
        print(f"\n[VALIDATION PASSED] All {total_pages} pages processed successfully.")

    # Run generic semantic extraction across all lines
    print("\nRunning generic semantic extraction...")
    semantic_result, provenance, debug_table = extract_fields_semantic(all_lines)
    user_schema = clean_user_facing_schema(semantic_result)

    # Map parties (Owner / Vendor / Purchaser)
    parties = semantic_result.get("parties_list") or []
    vendor_party = next((p for p in parties if p.get("role") == "Vendor"), None)
    purchaser_party = next((p for p in parties if p.get("role") == "Purchaser"), None)

    vendor_field = locate_party_provenance(vendor_party, "Vendor / Owner", "vendor_owner", all_lines=all_lines)
    purchaser_field = locate_party_provenance(purchaser_party, "Purchaser", "purchaser", all_lines=all_lines)

    def locate_field(val, prov_entry, field_name, force_needs_review=None):
        return locate_field_provenance(val, prov_entry, field_name, all_lines=all_lines, force_needs_review=force_needs_review)

    cs_prov = provenance.get("city_survey_number", {})
    city_survey_field = locate_field(
        user_schema.get("city_survey_number"),
        cs_prov,
        "city_survey_number",
        force_needs_review=True if user_schema.get("city_survey_number") else None
    )

    area_val = user_schema.get("property_area") or user_schema.get("property_area_text")
    area_prov = provenance.get("property_area", {})
    area_field = locate_field(
        area_val,
        area_prov,
        "area",
        force_needs_review=True if (area_prov.get("confidence", 1.0) <= 0.80 or "repair" in str(area_prov.get("source", "")).lower()) else None
    )

    # Stamp serials & stamp sheet dates
    stamp_serial_field = locate_field(user_schema.get("stamp_serial_number"), provenance.get("stamp_serial_number", {}), "stamp_serial_number")
    stamp_serial_field["stamp_serial_numbers"] = semantic_result.get("stamp_serial_numbers") or []

    stamp_pur_date_field = locate_field(user_schema.get("stamp_purchase_date"), provenance.get("stamp_purchase_date", {}), "stamp_purchase_date")
    stamp_pur_date_field["stamp_sheet_dates"] = semantic_result.get("stamp_sheet_dates") or []

    locality_field = locate_field(user_schema.get("locality_or_address"), provenance.get("locality_or_address", {}), "locality_or_address")

    # Format all requested fields with complete provenance
    structured_fields = {
        "document_type": locate_field(user_schema.get("document_type"), provenance.get("document_type", {}), "document_type"),
        "document_number": locate_field(user_schema.get("document_number"), provenance.get("document_number", {}), "document_number"),
        "vendor_owner": vendor_field,
        "purchaser": purchaser_field,
        "survey_number": locate_field(user_schema.get("survey_number"), provenance.get("survey_number", {}), "survey_number"),
        "city_survey_number": city_survey_field,
        "khasra_number": locate_field(user_schema.get("khasra_number"), provenance.get("khasra_number", {}), "khasra_number"),
        "khata_number": locate_field(user_schema.get("khata_number"), provenance.get("khata_number", {}), "khata_number"),
        "patta_number": locate_field(user_schema.get("patta_number"), provenance.get("patta_number", {}), "patta_number"),
        "plot_number": locate_field(user_schema.get("plot_number"), provenance.get("plot_number", {}), "plot_number"),
        "layout_name": locate_field(user_schema.get("layout_name"), provenance.get("layout_name", {}), "layout_name"),
        "locality_or_address": locality_field,
        "area": area_field,
        "village": locate_field(user_schema.get("village"), provenance.get("village", {}), "village"),
        "mandal_tehsil": locate_field(user_schema.get("mandal"), provenance.get("mandal", {}), "mandal"),
        "district": locate_field(user_schema.get("district"), provenance.get("district", {}), "district"),
        "state": locate_field(user_schema.get("state"), provenance.get("state", {}), "state"),
        "stamp_serial_number": stamp_serial_field,
        "stamp_value": locate_field(user_schema.get("stamp_value"), provenance.get("stamp_value", {}), "stamp_value"),
        "stamp_sold_to": locate_field(user_schema.get("stamp_sold_to"), provenance.get("stamp_sold_to", {}), "stamp_sold_to"),
        "stamp_purchase_date": stamp_pur_date_field,
        "document_date": locate_field(user_schema.get("document_date"), provenance.get("document_date", {}), "document_date"),
        "execution_date": locate_field(user_schema.get("execution_date"), provenance.get("execution_date", {}), "execution_date", force_needs_review=True if user_schema.get("execution_date") is None else None),
        "registration_date": locate_field(user_schema.get("registration_date"), provenance.get("registration_date", {}), "registration_date"),
    }

    # Summary of missing and conflicting fields
    missing_fields = [k for k, v in structured_fields.items() if v["status"] == "NOT_FOUND" or v["value"] is None]
    conflicting_fields = [k for k, v in structured_fields.items() if v["status"] == "CONFLICT"]
    needs_review_fields = [k for k, v in structured_fields.items() if v.get("needs_review") is True]

    # Check if any page was marked as needing review due to multilingual/model flags
    if any(p.get("needs_review") for p in pages_ocr_results):
        if "multilingual_model_warning" not in needs_review_fields:
            needs_review_fields.append("multilingual_model_warning")

    # Handwriting detection and pluggable recognition (kept strictly separate)
    from land_document_extractor import (
        detect_handwriting_regions,
        get_handwriting_recognizer,
        build_handwriting_metadata,
    )
    all_hw_regions = []
    hw_recognizer = get_handwriting_recognizer(ocr_url=ocr_url if ocr_url else None)
    for page_res in pages_ocr_results:
        p_num = page_res.get("page_number", 1)
        p_lines = [
            OCRLine(
                text=line.get("text", ""),
                score=float(line.get("confidence", 0.0)),
                x_min=int(line.get("bbox", [0, 0, 0, 0])[0]),
                y_min=int(line.get("bbox", [0, 0, 0, 0])[1]),
                x_max=int(line.get("bbox", [0, 0, 0, 0])[2]),
                y_max=int(line.get("bbox", [0, 0, 0, 0])[3]),
                page_num=p_num,
            )
            for line in page_res.get("ocr_lines", [])
        ]
        # Approximate image height/width from bounding boxes or standard 1200x1700 at 1.5 scale
        max_y = max([l.y_max for l in p_lines] + [1600])
        max_x = max([l.x_max for l in p_lines] + [1200])
        p_regions = detect_handwriting_regions(p_lines, image_height=int(max_y), image_width=int(max_x), page_num=p_num)
        for r in p_regions:
            processed_r = hw_recognizer.recognize_region(r, image=None)
            all_hw_regions.append(processed_r)

    handwriting_info = build_handwriting_metadata(all_hw_regions)

    processed_page_numbers = [p.get("page_number") for p in pages_ocr_results]
    doc_languages = sorted(set(
        lng for p in pages_ocr_results for lng in p.get("detected_languages", [])
    ))

    remote_run_summary = {
        "execution_mode": "remote_gpu" if ocr_url else ("cached" if use_cache else "local_cpu"),
        "requested_language": lang,
        "pages_sent": pages_sent if (ocr_url or use_cache) else list(range(1, total_pages + 1)),
        "pages_received": pages_received if (ocr_url or use_cache) else [p.get("page_number") for p in pages_ocr_results],
        "remote_failures": remote_failures,
        "fallback_used": fallback_used,
    }

    final_output = {
        "metadata": {
            "source_document": str(pdf_path),
            "total_pages": total_pages,
            "total_ocr_pages": len(pages_ocr_results),
            "processed_page_numbers": processed_page_numbers,
            "cache_used": use_cache,
            "complete_processing": is_complete,
            "failed_pages": failed_pages,
            "validation_errors": val_errors,
            "total_processing_time_s": total_time_s,
            "total_lines_extracted": len(all_lines),
            "render_scale": 1.5,
            "omp_num_threads": 8,
            "execution_mode": remote_run_summary["execution_mode"],
            "requested_language": remote_run_summary["requested_language"],
            "pages_sent": remote_run_summary["pages_sent"],
            "pages_received": remote_run_summary["pages_received"],
            "remote_failures": remote_run_summary["remote_failures"],
            "fallback_used": remote_run_summary["fallback_used"],
            "remote_run_summary": remote_run_summary,
        },
        "multilingual_metadata": {
            "language_strategy": lang,
            "detected_document_languages": doc_languages,
            "primary_document_language": pages_ocr_results[0].get("primary_language") if pages_ocr_results else None,
            "language_detection_method": "unicode_script",
            "models_available_locally": {
                l: is_model_locally_available(l) for l in ["en", "te", "hi", "kn", "ta", "mr", "ur"]
            },
        },
        "handwriting": handwriting_info,
        "pages_ocr": pages_ocr_results,
        "structured_extraction": structured_fields,
        "missing_fields": missing_fields,
        "conflicting_fields": conflicting_fields,
        "needs_review_fields": needs_review_fields,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved extraction results to {out_json}")
    print(f"Missing fields: {missing_fields}")
    print(f"Conflicting fields: {conflicting_fields}")
    print(f"Needs review fields: {needs_review_fields}")
    return final_output

def main(argv=None):
    opts = parse_args(argv)
    return process_document(
        pdf_path=opts["pdf_path"],
        out_json=opts["out_json"],
        use_cache=opts["use_cache"],
        lang=opts.get("lang", "auto"),
        ocr_url=opts.get("ocr_url"),
    )

if __name__ == "__main__":
    main()
