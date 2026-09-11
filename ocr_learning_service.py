"""
ocr_learning_service.py - OneBhoomi Adaptive OCR Feedback-Learning Service.

Smart India Hackathon (SIH) prototype component.
Provides an offline-first, officer-verified feedback-learning loop that improves
future OCR normalization and confidence estimation without retraining neural models
or modifying cryptographic verification records.

Expected Learning Flow:
OCR output -> Clerk corrects a field -> Correction is stored as pending ->
Officer approves the document -> Correction becomes verified ->
After at least two verified matching corrections, a rule becomes active ->
Future matching OCR values are normalized.

Key Safeguards & Principles:
1. Officer-verified adaptive OCR normalization & human-in-the-loop feedback learning.
2. Neural OCR weights (PaddleOCR) are NOT retrained or modified.
3. No reinforcement learning; strictly deterministic rule induction from verified examples.
4. Raw OCR confidence is NEVER altered.
5. Tripartite metrics maintained separately:
   - ocr_confidence (raw PaddleOCR confidence)
   - normalization_confidence (evidence-based rule confidence)
   - final_confidence (bounded overall confidence, capped at 0.95)
6. Scoped strictly by (field_name, document_type, language). Rules never bleed across fields,
   document types, or languages.
7. Normalization timing: post-selection field normalization.
8. Raw OCR values and original field values are permanently preserved.
"""

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default local persistence path (ignored runtime scratch directory)
DEFAULT_STORE_DIR = Path(__file__).resolve().parent / "scratch" / "learning"
DEFAULT_STORE_FILE = DEFAULT_STORE_DIR / "learning_store.json"

_STORE_LOCK = threading.RLock()
_CURRENT_STORE_FILE = DEFAULT_STORE_FILE


def set_store_file(file_path: Path | str) -> None:
    """Override store file path for isolated unit testing."""
    global _CURRENT_STORE_FILE
    with _STORE_LOCK:
        _CURRENT_STORE_FILE = Path(file_path)


def get_store_file() -> Path:
    """Return the active store file path."""
    with _STORE_LOCK:
        return _CURRENT_STORE_FILE


# ---------------------------------------------------------------------------
# Storage & Persistence Layer
# ---------------------------------------------------------------------------

def _load_store() -> Dict[str, Any]:
    """Read the learning store JSON from disk with fallback to default structure."""
    store_file = get_store_file()
    if not store_file.exists():
        return {
            "version": 2,
            "feedback": [],
            "rules": {},
            "stats": {
                "verified_feedback_count": 0,
                "pending_feedback_count": 0,
                "rejected_feedback_count": 0,
                "learned_rules_count": 0,
                "rules_applied_total": 0,
                "fields_improved": [],
            },
        }

    try:
        with open(store_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Store root is not a dictionary")
            data.setdefault("feedback", [])
            data.setdefault("rules", {})
            data.setdefault("stats", {
                "verified_feedback_count": 0,
                "pending_feedback_count": 0,
                "rejected_feedback_count": 0,
                "learned_rules_count": 0,
                "rules_applied_total": 0,
                "fields_improved": [],
            })
            return data
    except Exception:
        return {
            "version": 2,
            "feedback": [],
            "rules": {},
            "stats": {
                "verified_feedback_count": 0,
                "pending_feedback_count": 0,
                "rejected_feedback_count": 0,
                "learned_rules_count": 0,
                "rules_applied_total": 0,
                "fields_improved": [],
            },
        }


def _save_store(data: Dict[str, Any]) -> None:
    """Write store atomically to disk using a temporary file."""
    store_file = get_store_file()
    store_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = store_file.parent / f"{store_file.name}.{uuid.uuid4().hex[:8]}.tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomic replace
        os.replace(temp_file, store_file)
    except Exception as e:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Failed to persist learning store: {e}")


def reset_learning_store() -> None:
    """Reset the learning store (primarily used for test teardown)."""
    with _STORE_LOCK:
        store_file = get_store_file()
        if store_file.exists():
            try:
                store_file.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Feedback Capture & Workflow Management
# ---------------------------------------------------------------------------

def stage_feedback(
    document_type: str,
    field_name: str,
    raw_ocr_value: Any,
    corrected_value: Any,
    verification_id: str,
    page_number: Optional[int] = None,
    source_bbox: Optional[List[int]] = None,
    language: str = "en",
    ocr_confidence_before: float = 0.85,
    source_crop_path: Optional[str] = None,
    status: str = "pending_approval",
    original_value: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage a human clerk correction for subsequent officer approval.
    Only called when raw_ocr_value != corrected_value.
    Preserves page_number and source_bbox evidence when available (or null if unavailable).
    """
    with _STORE_LOCK:
        store = _load_store()

        raw_str = str(raw_ocr_value).strip() if raw_ocr_value is not None else ""
        corr_str = str(corrected_value).strip() if corrected_value is not None else ""
        orig_str = str(original_value).strip() if original_value is not None else raw_str

        # Skip if effectively identical or empty correction
        if raw_str == corr_str or not corr_str:
            return {}

        clean_page = int(page_number) if isinstance(page_number, (int, float)) and not isinstance(page_number, bool) else None
        clean_bbox = list(source_bbox) if isinstance(source_bbox, list) else None

        feedback_id = f"fb_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        item = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "document_type": document_type or "Sale Deed",
            "field_name": field_name,
            "raw_ocr_value": raw_str,
            "original_value": orig_str,
            "corrected_value": corr_str,
            "page_number": clean_page,
            "source_bbox": clean_bbox,
            "language": language or "en",
            "ocr_confidence_before": round(float(ocr_confidence_before), 4) if ocr_confidence_before is not None else 0.85,
            "status": status,
            "verification_id": verification_id,
            "source_crop_path": source_crop_path,
        }

        # Check for existing staged feedback for this verification_id + field_name
        replaced = False
        for i, existing in enumerate(store["feedback"]):
            if (
                existing.get("verification_id") == verification_id
                and existing.get("field_name") == field_name
                and existing.get("status") == "pending_approval"
            ):
                store["feedback"][i] = item
                replaced = True
                break

        if not replaced:
            store["feedback"].append(item)

        _recalculate_rules(store)
        _save_store(store)
        return item


def record_feedback(
    document_type: str,
    field_name: str,
    raw_ocr_value: Any,
    corrected_value: Any,
    verification_id: str,
    page_number: Optional[int] = None,
    source_bbox: Optional[List[int]] = None,
    language: str = "en",
    ocr_confidence_before: float = 0.85,
    source_crop_path: Optional[str] = None,
    auto_approve: bool = False,
    original_value: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience public API to record feedback, with optional auto-approval for tests/demo."""
    status = "verified" if auto_approve else "pending_approval"
    return stage_feedback(
        document_type=document_type,
        field_name=field_name,
        raw_ocr_value=raw_ocr_value,
        corrected_value=corrected_value,
        verification_id=verification_id,
        page_number=page_number,
        source_bbox=source_bbox,
        language=language,
        ocr_confidence_before=ocr_confidence_before,
        source_crop_path=source_crop_path,
        status=status,
        original_value=original_value,
    )


def approve_feedback_for_verification(verification_id: str) -> int:
    """
    When an officer approves a deed, promote all pending feedback items
    associated with this verification_id to 'verified'.
    Compiles newly verified rules into active learned normalization models.
    """
    if not verification_id:
        return 0

    with _STORE_LOCK:
        store = _load_store()
        approved_count = 0

        for item in store["feedback"]:
            if (
                item.get("verification_id") == verification_id
                and item.get("status") == "pending_approval"
            ):
                item["status"] = "verified"
                item["approved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                approved_count += 1

        if approved_count > 0:
            _recalculate_rules(store)
            _save_store(store)

        return approved_count


def reject_feedback_for_verification(verification_id: str) -> int:
    """
    When an officer rejects a deed or clerk abandons edits, mark pending feedback
    items as 'rejected' so they are NEVER used to generate learned normalization rules.
    """
    if not verification_id:
        return 0

    with _STORE_LOCK:
        store = _load_store()
        rejected_count = 0

        for item in store["feedback"]:
            if (
                item.get("verification_id") == verification_id
                and item.get("status") == "pending_approval"
            ):
                item["status"] = "rejected"
                item["rejected_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                rejected_count += 1

        if rejected_count > 0:
            _recalculate_rules(store)
            _save_store(store)

        return rejected_count


def capture_changed_payload_fields(
    old_payload: Dict[str, Any],
    new_form_fields: Dict[str, Any],
    verification_id: str,
    document_type: str = "Sale Deed",
    auto_approve: bool = False,
    field_provenance: Optional[Dict[str, Any]] = None,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """
    Compares the original extracted document payload with the human-submitted form fields.
    Records ONLY fields that actually changed, ignoring identical or empty entries.
    Passes original field provenance (page_number, source_bbox, ocr_confidence_before)
    into the learning service.
    """
    if not isinstance(old_payload, dict):
        old_payload = {}

    old_prop = old_payload.get("property") or {}
    old_stamp = old_payload.get("stamp_information") or {}

    field_mappings = [
        ("document_type", old_payload.get("document_type"), new_form_fields.get("document_type")),
        ("document_number", old_payload.get("document_number"), new_form_fields.get("document_number")),
        ("survey_number", old_prop.get("survey_number"), new_form_fields.get("survey_number")),
        ("sub_survey_number", old_prop.get("sub_survey_number"), new_form_fields.get("sub_survey_number")),
        ("property_area", old_prop.get("area") or old_prop.get("property_area"), new_form_fields.get("area")),
        ("village", old_prop.get("village"), new_form_fields.get("village")),
        ("mandal", old_prop.get("mandal"), new_form_fields.get("mandal")),
        ("district", old_prop.get("district"), new_form_fields.get("district")),
        ("stamp_number", old_stamp.get("stamp_number") or old_payload.get("stamp_number"), new_form_fields.get("stamp_number")),
        ("stamp_value", old_stamp.get("stamp_value") or old_payload.get("stamp_value"), new_form_fields.get("stamp_value")),
        ("sold_to", old_stamp.get("sold_to"), new_form_fields.get("sold_to")),
        ("document_date", old_payload.get("document_date"), new_form_fields.get("document_date")),
        ("execution_date", old_payload.get("execution_date"), new_form_fields.get("execution_date")),
    ]

    recorded = []
    status = "verified" if auto_approve else "pending_approval"
    prov_dict = field_provenance if isinstance(field_provenance, dict) else {}

    for field_name, old_val, new_val in field_mappings:
        if new_val is None:
            continue
        old_s = str(old_val).strip() if old_val is not None else ""
        new_s = str(new_val).strip()

        # Capture only actual changes
        if new_s and old_s != new_s:
            f_prov = prov_dict.get(field_name) or {}

            # Page number as int or None
            raw_page = f_prov.get("page_number") if f_prov.get("page_number") is not None else f_prov.get("page")
            try:
                page_num = int(raw_page) if raw_page is not None else None
            except (ValueError, TypeError):
                page_num = None

            # Bounding box as list or None
            src_box = f_prov.get("source_bbox") or f_prov.get("bbox")
            if not isinstance(src_box, list):
                src_box = None

            # OCR confidence before
            raw_conf = f_prov.get("ocr_confidence") if f_prov.get("ocr_confidence") is not None else f_prov.get("confidence", 0.85)
            try:
                conf_before = float(raw_conf)
            except (ValueError, TypeError):
                conf_before = 0.85

            raw_ocr_val = f_prov.get("raw_ocr_value") or f_prov.get("original_value") or old_s
            orig_val = f_prov.get("original_value") or old_s
            field_lang = f_prov.get("language") or language or "en"
            crop_path = f_prov.get("source_crop_path")

            item = stage_feedback(
                document_type=document_type,
                field_name=field_name,
                raw_ocr_value=raw_ocr_val,
                corrected_value=new_s,
                verification_id=verification_id,
                page_number=page_num,
                source_bbox=src_box,
                language=field_lang,
                ocr_confidence_before=conf_before,
                source_crop_path=crop_path,
                status=status,
                original_value=orig_val,
            )
            if item:
                recorded.append(item)

    return recorded


# ---------------------------------------------------------------------------
# Rule Compilation & Safe Learning Policy
# ---------------------------------------------------------------------------

def _calculate_confidence(evidence_count: int) -> float:
    """
    Evidence counts formula:
    0 verified examples -> inactive (0.0)
    1 verified example -> inactive (0.0)
    2 verified examples -> 0.80
    3 verified examples -> 0.85
    4 verified examples -> 0.90
    5+ verified examples -> 0.95 maximum (capped)
    """
    if evidence_count < 2:
        return 0.0
    if evidence_count == 2:
        return 0.80
    if evidence_count == 3:
        return 0.85
    if evidence_count == 4:
        return 0.90
    return 0.95


def _recalculate_rules(store: Dict[str, Any]) -> None:
    """
    Compile verified feedback items into active field-scoped learned rules.
    Rules are scoped strictly by (field_name, document_type, language).
    Rules are formed ONLY when evidence_count >= 2 for approved corrections.
    """
    verified_items = [
        item for item in store.get("feedback", [])
        if item.get("status") == "verified"
    ]
    pending_items = [
        item for item in store.get("feedback", [])
        if item.get("status") == "pending_approval"
    ]
    rejected_items = [
        item for item in store.get("feedback", [])
        if item.get("status") == "rejected"
    ]

    # Update stats
    store["stats"]["verified_feedback_count"] = len(verified_items)
    store["stats"]["pending_feedback_count"] = len(pending_items)
    store["stats"]["rejected_feedback_count"] = len(rejected_items)

    # Group verified items by:
    # (field_name, document_type, language, raw_pattern, replacement)
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for item in verified_items:
        f_name = (item.get("field_name") or "").strip()
        doc_type = (item.get("document_type") or "Sale Deed").strip()
        lang = (item.get("language") or "en").strip().lower()
        raw_v = str(item.get("raw_ocr_value", "")).strip()
        corr_v = str(item.get("corrected_value", "")).strip()

        if not f_name or not raw_v or not corr_v:
            continue

        key = (f_name, doc_type, lang, raw_v, corr_v)
        grouped.setdefault(key, []).append(item)

    rules: Dict[str, List[Dict[str, Any]]] = {}
    total_rule_count = 0

    for (f_name, doc_type, lang, raw_v, corr_v), items in grouped.items():
        evidence_count = len(items)

        # Enforce safety threshold: at least 2 verified occurrences required
        if evidence_count >= 2:
            conf = _calculate_confidence(evidence_count)
            rule_id = f"rule_{f_name}_{abs(hash(f'{f_name}|{doc_type}|{lang}|{raw_v}|{corr_v}')) % 1000000}"
            rule_entry = {
                "rule_id": rule_id,
                "field_name": f_name,
                "document_type": doc_type,
                "language": lang,
                "raw_pattern": raw_v,
                "replacement": corr_v,
                "evidence_count": evidence_count,
                "confidence": conf,
                "created_at": items[-1].get("timestamp") or items[-1].get("approved_at"),
            }
            rules.setdefault(f_name, []).append(rule_entry)
            total_rule_count += 1

    store["rules"] = rules
    store["stats"]["learned_rules_count"] = total_rule_count
    store["stats"]["fields_improved"] = sorted(list(rules.keys()))


# ---------------------------------------------------------------------------
# Learned Normalization Engine (Field-Scoped, Post-Selection Normalization)
# ---------------------------------------------------------------------------

# Built-in deterministic baseline normalizations (per field)
DETERMINISTIC_FIELD_RULES: Dict[str, List[Dict[str, Any]]] = {
    "survey_number": [
        # Normalize repeated OCR notation variations to canonical "Survey No"
        {"pattern": r"\b(?:Sy\.?\s*No\.?|S\.No\.?|SY\.?\s*NOS?)\b", "replace": "Survey No", "name": "sy_no_expand"},
        # Normalize stray OCR pipe in survey number lists e.g. "278 | 281" -> "278, 281"
        {"pattern": r"(?<=\d)\s*\|\s*(?=\d)", "replace": ", ", "name": "pipe_to_comma_survey_list"},
        # Clean double spaces
        {"pattern": r"\s+", "replace": " ", "name": "survey_space_norm"},
    ],
    "property_area": [
        # Normalize common OCR letter-O confusion in numeric area prefix or digits
        {"pattern": r"\b(\d+)O(\d*)\b", "replace": r"\g<1>0\g<2>", "name": "digit_O_confusion"},
        {"pattern": r"\bO(\d+)\b", "replace": r"0\g<1>", "name": "leading_O_confusion"},
    ],
    "document_number": [
        # Normalize pipe in registration slash-format e.g. "12736|5" -> "12736/5"
        {"pattern": r"(\d{3,6})\s*\|\s*(\d{1,4})", "replace": r"\1/\2", "name": "pipe_to_slash_docno"},
    ],
    "mandal": [
        # Standard spelling normalizations for common Telangana mandals
        {"pattern": r"^GHATKOSAR$", "replace": "Ghatkesar", "name": "ghatkesar_repair"},
        {"pattern": r"^QUTBULLAPUR$", "replace": "Quthbullapur", "name": "quthbullapur_repair"},
    ],
}


def apply_learned_normalization(
    field_name: str,
    raw_value: Any,
    ocr_confidence: Optional[float] = None,
    document_type: Optional[str] = None,
    language: Optional[str] = None,
) -> Tuple[Any, float, Optional[float], float, Optional[Dict[str, Any]], str]:
    """
    Applies learned rules to raw_value for the specified field_name.
    Timing: Post-selection field normalization.

    Requires:
        same field_name
        AND compatible document_type
        AND compatible language
        AND exact raw-value match

    Returns:
        (
            normalized_value,
            ocr_confidence,           # Raw OCR score (NEVER modified)
            normalization_confidence, # Evidence-based rule confidence or None
            final_confidence,         # Combined/bounded overall confidence (capped <= 0.95)
            rule_applied,             # Dict of rule metadata or None
            normalization_type,       # "raw_ocr" | "deterministic_normalization" | "learned_correction"
        )
    """
    raw_ocr_score = round(float(ocr_confidence if ocr_confidence is not None else 0.85), 4)

    if raw_value is None:
        return None, raw_ocr_score, None, 0.0, None, "raw_ocr"

    raw_str = str(raw_value).strip()
    if not raw_str:
        return raw_value, raw_ocr_score, None, raw_ocr_score, None, "raw_ocr"

    req_doc = (document_type or "").strip().lower()
    req_lang = (language or "").strip().lower()

    with _STORE_LOCK:
        store = _load_store()
        field_rules = store.get("rules", {}).get(field_name, [])

    # 1. Check verified learned rules (scoped strictly to field_name, document_type, language)
    # If doc_type or language is unknown/missing, do not apply narrowly scoped rules
    if req_doc and req_doc not in ("unknown", "null", "none") and req_lang and req_lang not in ("unknown", "null", "none"):
        for rule in field_rules:
            rule_field = rule.get("field_name")
            if rule_field != field_name:
                continue

            rule_doc = (rule.get("document_type") or "").strip().lower()
            rule_lang = (rule.get("language") or "").strip().lower()

            # Scoping compatibility check
            if rule_doc != req_doc:
                continue
            if rule_lang != req_lang:
                continue

            # Exact raw-value match required
            raw_pattern = rule.get("raw_pattern", "")
            if raw_str != raw_pattern:
                continue

            # Compatible rule matched!
            replacement = rule.get("replacement", "")
            evidence_count = rule.get("evidence_count", 0)
            norm_conf = rule.get("confidence", _calculate_confidence(evidence_count))
            # Final confidence is capped at 0.95 and incorporates evidence strength safely
            final_conf = round(min(0.95, max(raw_ocr_score, norm_conf)), 4)
            rule_info = {
                "rule_id": rule.get("rule_id"),
                "field_name": field_name,
                "document_type": rule.get("document_type"),
                "language": rule.get("language"),
                "raw_pattern": raw_pattern,
                "replacement": replacement,
                "evidence_count": evidence_count,
                "confidence": norm_conf,
                "created_at": rule.get("created_at"),
            }
            return replacement, raw_ocr_score, norm_conf, final_conf, rule_info, "learned_correction"

    # 2. Check deterministic baseline normalization (field-scoped)
    det_rules = DETERMINISTIC_FIELD_RULES.get(field_name, [])
    current_val = raw_str
    applied_det_rule = None

    for r in det_rules:
        pat = r["pattern"]
        rep = r["replace"]
        new_val = re.sub(pat, rep, current_val, flags=re.IGNORECASE)
        if new_val != current_val:
            current_val = new_val.strip()
            applied_det_rule = r["name"]

    if current_val != raw_str:
        det_conf = 0.90
        final_conf = round(min(0.95, max(raw_ocr_score, det_conf)), 4)
        return current_val, raw_ocr_score, det_conf, final_conf, None, "deterministic_normalization"

    # 3. No rule matched: return raw OCR value intact
    return raw_value, raw_ocr_score, None, raw_ocr_score, None, "raw_ocr"


# ---------------------------------------------------------------------------
# Statistics & Inspection APIs
# ---------------------------------------------------------------------------

def get_learning_stats() -> Dict[str, Any]:
    """Retrieve summary operational statistics of the adaptive learning store."""
    with _STORE_LOCK:
        store = _load_store()
        feedback = store.get("feedback", [])
        rules = store.get("rules", {})

        verified_count = sum(1 for item in feedback if item.get("status") == "verified")
        pending_count = sum(1 for item in feedback if item.get("status") == "pending_approval")
        rejected_count = sum(1 for item in feedback if item.get("status") == "rejected")

        total_rules = sum(len(rlist) for rlist in rules.values())
        fields_improved = sorted(list(rules.keys()))

        return {
            "verified_feedback_count": verified_count,
            "pending_feedback_count": pending_count,
            "rejected_feedback_count": rejected_count,
            "learned_rules_count": total_rules,
            "fields_improved": fields_improved,
            "learning_mode": "officer_verified_adaptive_feedback",
            "store_path": str(get_store_file()),
            "safety_policy": {
                "min_verified_examples": 2,
                "max_rule_confidence": 0.95,
                "raw_ocr_confidence_preserved": True,
                "field_scoped": True,
                "document_type_scoped": True,
                "language_scoped": True,
            },
        }


def get_learned_rules(field_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all active learned rules, optionally filtered by field_name."""
    with _STORE_LOCK:
        store = _load_store()
        rules_dict = store.get("rules", {})
        if field_name:
            return rules_dict.get(field_name, [])
        all_rules = []
        for r_list in rules_dict.values():
            all_rules.extend(r_list)
        return all_rules
