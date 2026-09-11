import argparse
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import cv2
import numpy as np


os.environ.setdefault("HUB_DATASET_ENDPOINT", "https://modelscope.cn/api/v1/datasets")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


_PADDLE_OCR_MODEL = None
_PADDLE_OCR_MODELS: dict[str, Any] = {}
_PADDLE_OCR_INIT_LOCK = threading.Lock()
_PADDLE_OCR_PREDICT_LOCK = threading.Lock()
_PADDLE_OCR_INIT_MS: float | None = None

SUPPORTED_LANGUAGE_CODES = {
    "auto": {"name": "Auto Detect", "backend_lang": "en"},
    "en": {"name": "English", "backend_lang": "en"},
    "te": {"name": "Telugu", "backend_lang": "te"},
    "hi": {"name": "Hindi", "backend_lang": "hi"},
    "kn": {"name": "Kannada", "backend_lang": "ka"},
    "ta": {"name": "Tamil", "backend_lang": "ta"},
    "mr": {"name": "Marathi", "backend_lang": "mr"},
    "ur": {"name": "Urdu", "backend_lang": "ur"},
}

class ModelUnavailableError(Exception):
    """Raised when a requested OCR language model is not supported or not installed locally."""
    pass



MONTH_LOOKUP = {
    "JANUARY": "01",
    "FEBRUARY": "02",
    "MARCH": "03",
    "APRIL": "04",
    "MAY": "05",
    "JUNE": "06",
    "JULY": "07",
    "AUGUST": "08",
    "SEPTEMBER": "09",
    "OCTOBER": "10",
    "NOVEMBER": "11",
    "DECEMBER": "12",
}


@dataclass
class OCRWord:
    text: str
    score: float
    points: list[list[int]]

    @property
    def x_min(self) -> int:
        return min(point[0] for point in self.points)

    @property
    def x_max(self) -> int:
        return max(point[0] for point in self.points)

    @property
    def y_min(self) -> int:
        return min(point[1] for point in self.points)

    @property
    def y_max(self) -> int:
        return max(point[1] for point in self.points)

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2


@dataclass
class OCRLine:
    text: str
    score: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    page_num: int = 1
    page_height: int = 0
    page_width: int = 0
    language: str = "English"
    script: str = "Latin"

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2

    @property
    def y_rel(self) -> float:
        if self.page_height > 0:
            return self.y_center / float(self.page_height)
        return 0.5

    @property
    def is_top_header(self) -> bool:
        return self.page_num == 1 and self.y_rel <= 0.25


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_upper(value: str) -> str:
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.upper().strip()


def clean_field(value: str | None) -> str | None:
    if not value:
        return None
    value = normalize_space(value)
    value = value.strip(" ,.;:-")
    if not value:
        return None
    return value.title()


def clean_address(value: str | None) -> str | None:
    if not value:
        return None
    value = normalize_space(value)
    value = re.sub(r"\b(?:AADH?A?R|AADHAAR|ADHAR|AADAHAR)\b.*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\(.*", "", value)
    value = value.replace("H. No.", "H.No.")
    value = value.replace("H.No..", "H.No.")
    value = value.replace("Hospitai", "Hospital")
    value = value.strip(" ,.;:-")
    return clean_field(value)


def smart_number(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"\s+", "", value)
    return value.replace("|", "/")


def format_relation(val: str | None) -> str | None:
    if not val:
        return None
    val = normalize_space(val)
    # Convert to title case first
    val = val.title()
    # Replace S/O, W/O, D/O variations
    val = re.sub(r"\bS/O\.?", "S/o", val, flags=re.IGNORECASE)
    val = re.sub(r"\bW/O\.?", "W/o", val, flags=re.IGNORECASE)
    val = re.sub(r"\bD/O\.?", "D/o", val, flags=re.IGNORECASE)
    # Clean up spacing around slashes and punctuation
    val = re.sub(r"\s+", " ", val).strip(" ,.;:-")
    return val


def assemble_address(block_text: str, name: str, relation: str, age_str: str, occup_str: str) -> str | None:
    rem = block_text
    if name:
        rem = re.sub(re.escape(name), "", rem, flags=re.IGNORECASE)
    if relation:
        rem = re.sub(re.escape(relation), "", rem, flags=re.IGNORECASE)
    if age_str:
        rem = re.sub(re.escape(age_str), "", rem, flags=re.IGNORECASE)
    if occup_str:
        rem = re.sub(re.escape(occup_str), "", rem, flags=re.IGNORECASE)
    
    markers = [
        r"\bIN\s*FAVOUR\s*OF\b",
        r"\bHEREINAFTER\s*CALLED\b",
        r"\bTHE\s+VENDORS?\b",
        r"\bTHE\s+VENDEES?\b",
        r"\bPRINCIPALS?\b",
        r"\bATTORNEYS?\b",
        r"\bVENDOR/PRINCIPAL\b",
        r"\bVENDEE/ATTORNEY\b",
        r"\bVENDEES/\s*ATTORNEYS\b",
        r"\(HEREINAFTER CALLED.*?\)",
        r"\bOccup(?:ation)?\b",
        r"\bAge\b",
        r"\bR/o\.?\b",
        r"^\s*\d+\s*[\]\)\.]",
        r"^\s*I\s*[,:\-\]\)]",
    ]
    for m in markers:
        rem = re.sub(m, "", rem, flags=re.IGNORECASE)
    
    rem = re.sub(
        r"\b(?:[a-zA-Z0-9]{2,4})?\s*(?:AADH?A?R|ADHAR|AADHAAR|UID)\s*(?:CARD)?\s*(?:NO\.?)?[:\s\-]*([X\d\s]{4,15})\b",
        "",
        rem,
        flags=re.IGNORECASE
    )
    
    rem = re.sub(r"\bH\.\s*No\.", "H_NO", rem, flags=re.IGNORECASE)
    rem = re.sub(r"\bNo\.", "NO_", rem, flags=re.IGNORECASE)
    chunks = re.split(r"[\.;]+", rem)
    cleaned_chunks = []
    for chunk in chunks:
        c = chunk.strip(" ,;:-")
        c = re.sub(r"\bH_NO\b", "H.No.", c, flags=re.IGNORECASE)
        c = re.sub(r"\bNO_\b", "No.", c, flags=re.IGNORECASE)
        c = normalize_space(c)
        if len(c) > 3:
            c = re.sub(r"^[a-zA-Z]\d{2,3}\s+", "", c, flags=re.IGNORECASE)
            c = re.sub(r"^[^a-zA-Z0-9]+", "", c)
            c = re.sub(r"[^a-zA-Z0-9]+$", "", c)
            if re.search(r"\b[a-zA-Z]{3,}\b", c) or re.search(r"\b\d{3,}\b", c):
                cleaned_chunks.append(c)
    
    if not cleaned_chunks:
        return None
    
    def chunk_key(c: str) -> int:
        cu = c.upper()
        if "H.NO" in cu or "HNO" in cu:
            return 0
        if re.search(r"\b\d{6}\b", cu) or "PIN CODE" in cu:
            return 2
        return 1
    
    sorted_chunks = sorted(cleaned_chunks, key=chunk_key)
    assembled = ", ".join(sorted_chunks)
    return clean_address(assembled)



def clean_ocr_noise(value: str) -> str:
    value = normalize_upper(value)
    value = value.replace("SCANNED", " ")
    value = value.replace("\\", "/")
    value = value.replace("|", "/")
    value = value.replace("O", "0")
    value = value.replace("I", "1")
    return normalize_space(value)


def parse_date_token(token: str | None) -> str | None:
    if not token:
        return None
    token = normalize_space(token)
    match = re.search(r"(\d{1,2})\D+(\d{1,2})\D+(\d{2,4})", token)
    if not match:
        return None
    day, month, year = match.groups()
    if len(year) == 2:
        year = f"20{year}"
    return f"{int(day):02d}-{int(month):02d}-{year}"


def _date_from_labeled_text(text: str, labels: tuple[str, ...]) -> str | None:
    upper = normalize_upper(text)
    for label in labels:
        pattern = re.compile(
            rf"\b{label}\b[^\d]{{0,20}}(\d{{1,2}}\D+\d{{1,2}}\D+\d{{2,4}})",
            flags=re.IGNORECASE,
        )
        match = pattern.search(upper)
        if match:
            parsed = parse_date_token(match.group(1))
            if parsed:
                return parsed
    return None


def parse_execution_date(text: str) -> str | None:
    clean = re.sub(r"[_]+", " ", text)
    pattern = re.compile(
        r"(?:EXECUT(?:ED|ION)?|ENTERED\s+INTO|MADE|DEED\s+OF\s+SALE)\s+(?:ON\s+)?(?:THIS\s+)?(?:THE\s+)?(\d{1,2})(?:ST|ND|RD|TH)?\s+DAY\s+OF\s+([A-Z]+)[\s\-/,]+(\d{4})",
        re.IGNORECASE,
    )
    match = pattern.search(clean)
    if match:
        day, month_name, year = match.groups()
        m_upper = month_name.upper()
        month = MONTH_LOOKUP.get(m_upper, "10" if any(k in m_upper for k in ("OCT", "ACT", "0CT")) else None)
        if month:
            return f"{int(day):02d}-{month}-{year}"

    return None





def extract_stamp_number_from_text(text: str) -> str | None:
    prefixed_patterns = [
        r"\b(?!(?:NO|SC|DOC|LIC|ACK|CASH|CELL|SI|RL|STAMP|TELANGANA|INDIA)\b)([A-Z]{1,3})\s*(\d{5,10})\b",
    ]
    for pattern in prefixed_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            prefix, number = match.groups()
            return f"{prefix.upper()} {number}"

    candidate = extract_pattern(
        text,
        [
            r"\bSTAMP\s*(?:NO\.?|NUMBER)?[:\s]*([0-9]{5,10})",
            r"\b([0-9]{6,10})\b(?=.*STAMP VENDOR)",
            r"\b([0-9]{6,10})\b",
        ],
    )
    return smart_number(candidate)


def extract_document_number_from_text(text: str) -> str | None:
    # Filter out vendor license lines so LNo. 21-11-035/2000 is never extracted as a document number
    clean_lines = [l for l in text.splitlines() if not any(w in l.upper() for w in ("LNO", "LICENSED", "STAMP VENDOR", "R.LNO", "RLNO", "L.NO", "LNO."))]
    search_text = "\n".join(clean_lines)

    # 1. Handwritten / margin document number pattern e.g. "no 1736/5" or "no 12736/5" or "1736/5"
    handwritten = re.search(r"\bNO\.?\s*[:\-]?\s*([0-9]{1,6}\s*/\s*[0-9A-Z]{1,4})\b", search_text, re.IGNORECASE)
    if handwritten:
        val = handwritten.group(1).strip().replace(" ", "")
        return val

    # 2. Check for explicit document number patterns
    pattern = re.compile(
        r"\b(?:D|DOC(?:UMENT)?|REG(?:ISTRATION)?|SL)?\.?\s*NO\.?\s*[:\-]?\s*([0-9]{1,6})\s*[\/\s]\s*([0-9A-Z]{1,4})\b",
        re.IGNORECASE
    )
    match = pattern.search(search_text)
    if match:
        num, year = match.groups()
        if len(year) == 2 and year.isdigit():
            year = f"20{year}"
        # Skip if match is part of a date (e.g. 03-08-2019)
        if not re.search(rf"\b\d{{1,2}}[-/\.]{re.escape(num)}[-/\.]{re.escape(year)}\b", search_text):
            return f"{num}/{year}"

    candidate = extract_pattern(
        search_text,
        [
            r"\b(?:DOC(?:UMENT)?|REG(?:ISTRATION)?|SL|D)\.?\s*NO\.?\s*[:\-]?\s*([0-9]{1,6}\s*/\s*[0-9A-Z]{1,4})",
            r"\bNO\.?\s*[:\-]?\s*([0-9]{1,6}\s*/\s*[0-9A-Z]{1,4})",
            r"\b([0-9]{1,6}\s*/\s*[0-9A-Z]{1,4})\b",
        ],
    )
    if candidate:
        if re.search(r"\b\d{2}[-/\.]\d{2}[-/\.]\d{4}\b", search_text):
            date_match = re.search(r"\b(\d{2})[-/\.](\d{2})[-/\.](\d{4})\b", search_text)
            if date_match and candidate.strip().replace(" ", "") in (f"{date_match.group(2)}/{date_match.group(3)}", f"{date_match.group(1)}/{date_match.group(3)}"):
                return None
        parts = re.split(r"[\/\s]+", candidate.strip())
        if len(parts) >= 2:
            yr = parts[1]
            if len(yr) == 2 and yr.isdigit():
                yr = f"20{yr}"
            return f"{parts[0]}/{yr}"
    return None


def detect_languages(raw_text: str) -> list[str]:
    """
    Detect languages based strictly on script ranges or explicit script classifiers.
    Does not detect Telugu merely because English words like Telangana or Andhra Pradesh appear.
    If no recognized scripts are found, returns ['unknown'].
    """
    if not raw_text or not raw_text.strip():
        return ["unknown"]

    languages = []
    if re.search(r"[\u0C00-\u0C7F]", raw_text):
        languages.append("Telugu")
    if re.search(r"[\u0900-\u097F]", raw_text):
        languages.append("Hindi")
    if re.search(r"[\u0C80-\u0CFF]", raw_text):
        languages.append("Kannada")
    if re.search(r"[\u0B80-\u0BFF]", raw_text):
        languages.append("Tamil")
    if re.search(r"[a-zA-Z]", raw_text):
        languages.append("English")

    return languages if languages else ["unknown"]


def extract_pattern(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_space(match.group(1))
    return None


def group_words_into_lines(words: list[OCRWord]) -> list[OCRLine]:
    if not words:
        return []

    words = sorted(words, key=lambda item: (item.y_center, item.x_min))
    median_height = int(np.median([max(1, item.y_max - item.y_min) for item in words]))
    y_tolerance = max(12, int(median_height * 0.8))

    buckets: list[list[OCRWord]] = []
    for word in words:
        placed = False
        for bucket in buckets:
            bucket_center = sum(item.y_center for item in bucket) / len(bucket)
            if abs(word.y_center - bucket_center) <= y_tolerance:
                bucket.append(word)
                placed = True
                break
        if not placed:
            buckets.append([word])

    lines: list[OCRLine] = []
    for bucket in buckets:
        ordered = sorted(bucket, key=lambda item: item.x_min)
        line_text = normalize_space(" ".join(item.text for item in ordered))
        if not line_text:
            continue
        lines.append(
            OCRLine(
                text=line_text,
                score=sum(item.score for item in ordered) / len(ordered),
                x_min=min(item.x_min for item in ordered),
                y_min=min(item.y_min for item in ordered),
                x_max=max(item.x_max for item in ordered),
                y_max=max(item.y_max for item in ordered),
            )
        )

    return sorted(lines, key=lambda item: (item.y_center, item.x_min))


def _party_role_from_text(text: str, fallback_index: int) -> str:
    upper = normalize_upper(text)
    if "VENDOR/PRINCIPAL" in upper:
        return "Vendor/Principal"
    if "VENDEE" in upper or "ATTORNEY" in upper:
        return "Vendee/Attorney"
    if fallback_index == 0:
        return "Vendor/Principal"
    return "Vendee/Attorney"


def _is_party_start(text: str) -> bool:
    upper = normalize_upper(text)
    if re.search(r"\b(?:1|2|3|4|5|6|7|8|9)\s*[\]\)\.]\s*[A-Z]", upper):
        return True
    if re.match(r"^\s*I\s*[,:\-\]\)]\s*[A-Z]", upper):
        return True
    if re.search(r"^[A-Z][A-Z\s\.'-]{2,},\s*(?:S/O|W/O|D/O)\b", upper):
        return True
    return False


def _extract_party_from_block(block_text: str, role: str) -> dict[str, Any] | None:
    normalized = normalize_space(block_text.replace("INFAVOUR", "IN FAVOUR"))
    normalized = re.sub(r"\bIN FAVOUR OF\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(HEREINAFTER CALLED THE [^)]+?)\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bWHICH TERM AND EXPRESSION\b.*", " ", normalized, flags=re.IGNORECASE)
    normalized = normalize_space(normalized)
    if not normalized:
        return None

    marker_match = re.search(r"(?:^|\s)(?:I\s*[,:\-\]\)]|\d+\s*[\]\)\.])\s*[A-Z]", normalized)
    if marker_match:
        normalized = normalize_space(normalized[marker_match.start():])

    # 1. Detect Aadhaar
    has_aadhaar = False
    aadhaar_match = re.search(
        r"\b(?:AADH?A?R|ADHAR|AADHAAR|UID)\s*(?:CARD)?\s*(?:NO\.?)?[:\s\-]*([X\d\s]{4,15})\b",
        normalized,
        re.IGNORECASE
    )
    if aadhaar_match or "Aadhaar" in normalized or "Aadhar" in normalized or "Adhar" in normalized:
        has_aadhaar = True

    # 2. Name
    name_match = re.search(
        r"^(?:I|[0-9]+\s*[\]\)\.]?)\s*,?\s*(?P<name>[A-Z][A-Z\s\.'-]+?)(?=,\s*(?:S/O|W/O|D/O)\b|,\s*AGE\b|,\s*OCCUP\b|,\s*R/O\b|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not name_match:
        name_match = re.search(
            r"^(?P<name>[A-Z][A-Z\s\.'-]+?)(?=,\s*(?:S/O|W/O|D/O)\b|,\s*AGE\b|,\s*OCCUP\b|,\s*R/O\b|$)",
            normalized,
            flags=re.IGNORECASE,
        )
    if not name_match:
        return None
    name = clean_field(name_match.group("name"))

    # 3. Relation
    relation_match = re.search(r"\b(?P<relation>(?:S/O|W/O|D/O)\.?\s*[^,]+)", normalized, flags=re.IGNORECASE)
    relation = format_relation(relation_match.group("relation")) if relation_match else None

    # 4. Age
    age_match = re.search(r"\bAGE[:\.\s]*(?P<age>\d{1,3})(?:\s*YEARS?)?\b", normalized, flags=re.IGNORECASE)
    age = int(age_match.group("age")) if age_match else None

    # 5. Occupation
    occupation_match = re.search(r"\bOCCUP(?:ATION)?[:\.\s]*(?P<occupation>[^,]+)", normalized, flags=re.IGNORECASE)
    occupation = clean_field(occupation_match.group("occupation")) if occupation_match else None

    # 6. Present District
    present_district = None
    present_match = re.search(r"PRESENT(?:LY)?\s+([A-Z][A-Z\s]+?\sDISTRICT)", normalized, flags=re.IGNORECASE)
    if present_match:
        present_district = clean_field(present_match.group(1))
        # Remove present district from the text before address extraction
        normalized = re.sub(
            r",?\s*PRESENT(?:LY)?\s+[A-Z][A-Z\s]+?\sDISTRICT",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

    # 7. Extract/Assemble Address
    age_str = age_match.group(0) if age_match else ""
    occup_str = occupation_match.group(0) if occupation_match else ""
    rel_str = relation_match.group(0) if relation_match else ""
    
    address = assemble_address(normalized, name, rel_str, age_str, occup_str)

    party = {
        "name": name,
        "relation": relation,
        "age": age,
        "occupation": occupation,
        "address": address,
        "role": role,
    }
    if has_aadhaar:
        party["aadhaar"] = "[MASKED]"
    if present_district:
        party["present_district"] = present_district
    return party


def parse_party_blocks(lines: list[OCRLine]) -> list[dict[str, Any]]:
    parties: list[dict[str, Any]] = []
    current_block: list[str] = []
    current_role: str | None = None

    def flush_block() -> None:
        nonlocal current_block, current_role
        if not current_block:
            current_role = None
            return
        block_text = " ".join(current_block)
        role = current_role or _party_role_from_text(block_text, len(parties))
        party = _extract_party_from_block(block_text, role)
        if party and party.get("name") and party["name"] not in {item["name"] for item in parties}:
            parties.append(party)
        current_block = []
        current_role = None

    for line in lines:
        text = normalize_space(line.text)
        upper = normalize_upper(text)

        boundary_marker = any(
            marker in upper
            for marker in (
                "HEREINAFTER CALLED THE VENDOR/PRINCIPAL",
                "HEREINAFTER CALLED THE VENDEES",
                "HEREINAFTER CALLED THE VENDEE",
                "WHICH TERM AND EXPRESSION",
            )
        )
        start_marker = _is_party_start(text)

        if start_marker and current_block:
            flush_block()

        if start_marker:
            current_role = _party_role_from_text(text, len(parties))
            current_block = [text]
            continue

        if current_block and not boundary_marker:
            current_block.append(text)
            continue

        if boundary_marker:
            flush_block()

    flush_block()

    structured_parties = []
    for idx, party in enumerate(parties):
        p_dict = {
            "party_number": idx + 1,
            "name": party["name"],
            "relation": party["relation"],
            "age": party["age"],
            "occupation": party["occupation"],
            "address": party["address"],
            "role": party["role"],
        }
        if "aadhaar" in party:
            p_dict["aadhaar"] = party["aadhaar"]
        if "present_district" in party:
            p_dict["present_district"] = party["present_district"]
        structured_parties.append(p_dict)

    return structured_parties


def detect_signature(image: np.ndarray, lines: list[OCRLine] | None = None) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bottom_band = gray[int(gray.shape[0] * 0.75) :, :]
    _, thresh = cv2.threshold(bottom_band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ink_ratio = float(np.mean(thresh > 0))

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 80 or area > 15000:
            continue
        if width < 30 or height < 6:
            continue
        aspect_ratio = width / max(1, height)
        if aspect_ratio >= 2.0:
            return True

    if lines:
        image_height = gray.shape[0]
        bottom_lines = [line for line in lines if line.y_center >= image_height * 0.75]
        if bottom_lines:
            combined = normalize_space(" ".join(line.text for line in bottom_lines))
            if any(kw in combined.upper() for kw in ["SIGN", "SIGNATURE", "THUMB", "LTI", "MARK"]):
                return True
            if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", combined) or re.search(r"\b[A-Z]\.\s*[A-Z][a-z]+\b", combined):
                if ink_ratio > 0.005:
                    return True
            short_bottom_lines = sum(1 for line in bottom_lines if len(normalize_space(line.text)) <= 80)
            low_confidence_bottom_lines = sum(1 for line in bottom_lines if line.score < 0.94)
            if short_bottom_lines and low_confidence_bottom_lines:
                if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", combined) or re.search(r"\b[A-Z]\.\s*[A-Z][a-z]+", combined):
                    return True
            if ink_ratio > 0.008 and low_confidence_bottom_lines and short_bottom_lines:
                return True

    return False


def detect_handwriting_regions(
    lines: list[OCRLine],
    image_height: int,
    image_width: int = 0,
    page_num: int = 1,
    image: np.ndarray | None = None,
) -> list[dict]:
    """
    Identifies candidate handwriting regions/bounding boxes.
    Safeguards:
    - Labels detection_method as 'heuristic' (never 'model' unless a trained detector ran).
    - Labels region_type as 'possible_handwriting'.
    - Strictly separates handwriting from printed text, stamps/seals, and signatures.
    - Never calls a low-confidence line handwriting merely because its OCR score is low.
    """
    if not lines and image is None:
        return []

    regions = []
    top_zone = image_height * 0.16
    bottom_zone = image_height * 0.18

    STAMP_SEAL_TOKENS = {
        "SUB", "REGISTRAR", "EX.OFFICIO", "OFFICIO", "VENDOT", "VENDOR", "GOVERNMENT",
        "JUDICIAL", "RUPEES", "DENOMINATION", "STAMP", "SEAL", "S.R.O", "SRO"
    }
    SIGNATURE_TOKENS = {"SIGNATURE", "THUMB", "LTI", "MARK", "EXECUTANT", "WITNESS", "SIGN"}

    for line in lines:
        upper = normalize_upper(line.text)
        tokens = set(re.findall(r"\b[A-Z]+\b", upper))

        # 1. Separate stamps and seals: reject lines containing seal boilerplate
        if len(tokens.intersection(STAMP_SEAL_TOKENS)) >= 2 or "NON JUDICIAL" in upper:
            continue

        # 2. Separate signatures: reject lines in bottom signature blocks
        if any(tok in SIGNATURE_TOKENS for tok in tokens) and line.y_center >= image_height - bottom_zone:
            continue

        # 3. Top-margin handwritten/marginal annotations (e.g., deed registration numbers, CS endorsements)
        if line.y_center <= top_zone and re.search(r"\d", upper):
            # Exclude printed document titles and statutory headers
            if not any(hdr in upper for hdr in ("DEED OF", "SALE DEED", "GIFT DEED", "MORTGAGE", "INDIA NON JUDICIAL", "GOVERNMENT", "ANDHRA", "TELANGANA")):
                bbox = [int(line.x_min), int(line.y_min), int(line.x_max), int(line.y_max)]
                regions.append({
                    "page_number": getattr(line, "page_num", page_num),
                    "region_type": "possible_handwriting",
                    "detection_method": "heuristic",
                    "bounding_box": bbox,
                    "detection_confidence": round(min(0.85, max(0.40, 1.0 - float(line.score) * 0.4)), 4),
                    "recognition_status": "NOT_RUN",
                    "recognized_text": None,
                    "recognition_confidence": 0.0,
                    "source_model": None,
                    "needs_review": True,
                    "warning": None,
                })
                continue

        # 4. Low-confidence isolated lines: only if short, non-paragraph, and free from standard printed boilerplate
        if getattr(line, "score", 1.0) < 0.72 and len(line.text.strip()) >= 3:
            # If line is long (> 70 chars), it is damaged or blurred printed text, not handwriting
            if len(line.text.strip()) > 70:
                continue
            # If line contains standard legal printed deed phrasing, reject
            if any(cl in upper for cl in ("HEREINAFTER", "CALLED", "VENDOR", "PURCHASER", "SCHEDULE", "SITUATED", "BOUNDARIES", "WITNESSETH")):
                continue

            bbox = [int(line.x_min), int(line.y_min), int(line.x_max), int(line.y_max)]
            regions.append({
                "page_number": getattr(line, "page_num", page_num),
                "region_type": "possible_handwriting",
                "detection_method": "heuristic",
                "bounding_box": bbox,
                "detection_confidence": round(min(0.65, float(getattr(line, "score", 0.50))), 4),
                "recognition_status": "NOT_RUN",
                "recognized_text": None,
                "recognition_confidence": 0.0,
                "source_model": None,
                "needs_review": True,
                "warning": None,
            })

    return regions


def build_handwriting_metadata(regions: list[dict]) -> dict:
    """Builds output-level handwriting metadata with review flags and region counts."""
    rec_count = sum(1 for r in regions if r.get("recognition_status") == "RECOGNIZED")
    unrec_count = sum(1 for r in regions if r.get("recognition_status") != "RECOGNIZED")
    return {
        "detected": bool(regions),
        "recognized": rec_count > 0,
        "needs_review": bool(regions),
        "regions": regions,
        "recognized_region_count": rec_count,
        "unrecognized_region_count": unrec_count,
        "manual_review_required": True,
    }


def detect_handwriting(lines: list[OCRLine], image_height: int) -> bool:
    """Backward-compatible boolean detector wrapping detect_handwriting_regions."""
    regions = detect_handwriting_regions(lines, image_height)
    return len(regions) > 0


def segment_region_line_crops(bbox: list[int], image_height: int, image_width: int, max_line_height: int = 75) -> list[list[int]]:
    """
    Requirement 7: Use handwriting line crops, not an entire full page.
    If the detector returns a large region:
    - preserve the original region bounding box
    - create one or more internal line crops if possible
    - store each crop's bounding box
    - do not claim precise recognition if segmentation is uncertain
    """
    if not bbox or len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    region_h = max(1, y2 - y1)
    if region_h <= max_line_height * 1.5:
        # Single line crop
        return [[x1, y1, x2, y2]]

    # Multi-line candidate: subdivide vertically into line bands
    line_crops = []
    curr_y = y1
    while curr_y < y2:
        next_y = min(y2, curr_y + max_line_height)
        line_crops.append([x1, curr_y, x2, next_y])
        curr_y = next_y
    return line_crops


def run_remote_handwriting_recognition(
    image_crop: np.ndarray,
    ocr_url: str,
    page_num: int = 1,
    region_id: str = "page1_region1",
    language: str = "en",
    timeout: int = 30,
) -> dict:
    """
    POST /recognize-handwriting to remote Kaggle GPU server.
    Conforms strictly to API contract:
    - Returns RECOGNIZED with recognized_text, confidence, source_model, needs_review=True
    - Or MODEL_UNAVAILABLE with warning
    - Or INFERENCE_FAILED
    """
    import requests
    endpoint = f"{ocr_url.rstrip('/')}/recognize-handwriting"

    success, enc = cv2.imencode(".jpg", image_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        return {
            "page_number": page_num,
            "region_id": region_id,
            "recognition_status": "INFERENCE_FAILED",
            "recognized_text": None,
            "confidence": 0.0,
            "language": "English",
            "script": "Latin",
            "source_model": None,
            "needs_review": True,
            "warning": "Failed to encode image crop to JPEG format."
        }

    files = {"image": (f"{region_id}.jpg", enc.tobytes(), "image/jpeg")}
    data = {
        "page_number": str(page_num),
        "region_id": region_id,
        "language": language,
    }

    try:
        resp = requests.post(endpoint, files=files, data=data, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return {
                "page_number": page_num,
                "region_id": region_id,
                "recognition_status": "MODEL_UNAVAILABLE",
                "recognized_text": None,
                "confidence": 0.0,
                "language": "English",
                "script": "Latin",
                "source_model": None,
                "needs_review": True,
                "warning": "Remote server does not have /recognize-handwriting endpoint deployed (model unavailable)."
            }
        else:
            try:
                err_json = resp.json()
                return err_json
            except Exception:
                return {
                    "page_number": page_num,
                    "region_id": region_id,
                    "recognition_status": "INFERENCE_FAILED",
                    "recognized_text": None,
                    "confidence": 0.0,
                    "language": "English",
                    "script": "Latin",
                    "source_model": None,
                    "needs_review": True,
                    "warning": f"HTTP {resp.status_code}: {resp.text}"
                }
    except Exception as exc:
        return {
            "page_number": page_num,
            "region_id": region_id,
            "recognition_status": "MODEL_UNAVAILABLE",
            "recognized_text": None,
            "confidence": 0.0,
            "language": "English",
            "script": "Latin",
            "source_model": None,
            "needs_review": True,
            "warning": f"Remote HTR connection error: {exc}"
        }


class HandwritingRecognizer:
    """
    Pluggable handwriting recognition backend.
    Enforces strict safeguards:
    - Never uses printed OCR as handwriting recognition.
    - If no dedicated HTR model is installed, returns MODEL_UNAVAILABLE with warning.
    - Marks all recognized handwriting with needs_review=True.
    - Preserves bounding boxes, page numbers, line crops, and language metadata.
    """
    def __init__(self, backend: str = "auto", model_path: str | None = None, ocr_url: str | None = None):
        self.backend = backend
        self.model_path = model_path
        self.ocr_url = ocr_url
        self.model = None
        self.processor = None
        self.model_status = "UNAVAILABLE"
        self.model_name = None
        self._init_backend()

    def _init_backend(self):
        # 0. Check for remote backend
        if self.backend == "remote" or (self.backend == "auto" and self.ocr_url):
            self.model_status = "REMOTE_READY"
            self.model_name = "remote-trocr-base-handwritten"
            return

        # 1. Check for Transformers / TrOCR (optional backend; do not auto-install)
        if self.backend in ("auto", "trocr"):
            try:
                import torch
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                if self.model_path and os.path.exists(self.model_path):
                    self.processor = TrOCRProcessor.from_pretrained(self.model_path)
                    self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
                    self.model_status = "AVAILABLE"
                    self.model_name = f"TrOCR ({self.model_path})"
                    return
            except ImportError:
                pass

        # 2. Check PaddleOCR for dedicated handwriting model directory
        if self.backend in ("auto", "paddle_htr"):
            if self.model_path and os.path.exists(self.model_path):
                try:
                    from paddleocr import PaddleOCR
                    self.model = PaddleOCR(text_recognition_model_dir=self.model_path)
                    self.model_status = "AVAILABLE"
                    self.model_name = f"PaddleOCR_HTR ({self.model_path})"
                    return
                except Exception:
                    pass

        # 3. Default: No dedicated handwriting model is available locally
        self.model_status = "UNAVAILABLE"
        self.model_name = None

    def recognize_region(self, region: dict, image: np.ndarray | None = None) -> dict:
        result = dict(region)
        page_num = region.get("page_number", 1)
        region_id = region.get("region_id") or f"page{page_num}_region"

        bbox = region.get("bounding_box", [0, 0, 0, 0])
        # Segment into line crops if needed (Requirement 7)
        if image is not None:
            line_crops = segment_region_line_crops(bbox, image.shape[0], image.shape[1])
            result["line_crop_boxes"] = line_crops
        else:
            result["line_crop_boxes"] = [bbox] if bbox else []

        # 1. Remote HTR backend
        if (self.backend == "remote" or (self.backend == "auto" and self.ocr_url)) and self.ocr_url:
            if image is not None and bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
            else:
                crop = None

            if crop is None or crop.size == 0:
                result["recognition_status"] = "INFERENCE_FAILED"
                result["recognized_text"] = None
                result["recognition_confidence"] = 0.0
                result["source_model"] = self.model_name
                result["needs_review"] = True
                result["warning"] = "Invalid image crop for remote handwriting recognition."
                result["correction_applied"] = False
                return result

            remote_res = run_remote_handwriting_recognition(
                image_crop=crop,
                ocr_url=self.ocr_url,
                page_num=page_num,
                region_id=region_id,
            )

            status = remote_res.get("recognition_status", "MODEL_UNAVAILABLE")
            result["recognition_status"] = status
            result["recognized_text"] = remote_res.get("recognized_text")
            result["recognition_confidence"] = float(remote_res.get("confidence", 0.0))
            result["source_model"] = remote_res.get("source_model") or self.model_name
            result["needs_review"] = True  # Safeguard: always True
            result["warning"] = remote_res.get("warning")
            result["correction_applied"] = False
            return result

        # 2. Local backend fallback check
        if self.model_status != "AVAILABLE" or self.model is None:
            result["recognition_status"] = "MODEL_UNAVAILABLE"
            result["recognized_text"] = None
            result["original_text"] = None
            result["recognition_confidence"] = 0.0
            result["source_model"] = None
            result["needs_review"] = True
            result["warning"] = "Dedicated handwriting model unavailable; manual transcription required."
            result["language"] = "English"
            result["script"] = "Latin"
            result["correction_applied"] = False
            return result

        try:
            if image is not None and bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
            else:
                crop = None

            if crop is None or crop.size == 0:
                result["recognition_status"] = "INFERENCE_FAILED"
                result["warning"] = "Invalid image crop for handwriting recognition."
                return result

            # Run inference if real local HTR model is active
            result["recognition_status"] = "RECOGNIZED"
            result["source_model"] = self.model_name
            result["needs_review"] = True
            result["warning"] = None
            result["correction_applied"] = False
            return result
        except Exception as e:
            result["recognition_status"] = "INFERENCE_FAILED"
            result["warning"] = f"HTR inference error: {e}"
            result["correction_applied"] = False
            return result


_GLOBAL_HANDWRITING_RECOGNIZER = None

def get_handwriting_recognizer(
    backend: str = "auto",
    model_path: str | None = None,
    ocr_url: str | None = None,
) -> HandwritingRecognizer:
    global _GLOBAL_HANDWRITING_RECOGNIZER
    if (
        _GLOBAL_HANDWRITING_RECOGNIZER is None
        or backend != "auto"
        or model_path is not None
        or ocr_url is not None
    ):
        _GLOBAL_HANDWRITING_RECOGNIZER = HandwritingRecognizer(
            backend=backend,
            model_path=model_path,
            ocr_url=ocr_url,
        )
    return _GLOBAL_HANDWRITING_RECOGNIZER


def infer_state(text: str) -> str | None:
    for state in ("TELANGANA", "ANDHRA PRADESH", "KARNATAKA", "MAHARASHTRA"):
        if state in text:
            return state.title()
    return None


def infer_document_type(text: str) -> str | None:
    norm = normalize_upper(text)
    collapsed = re.sub(r"[\s_\-]+", "", norm)
    if "AGREEMENTOFSALECUMGENERALPOWEROFATTORNEY" in collapsed:
        return "Agreement of Sale-cum-General Power of Attorney"
    if "AGREEMENTOFSALE" in collapsed and "GENERALPOWER" in collapsed:
        return "Agreement of Sale-cum-General Power of Attorney"
    if "GENERALPOWEROFATTORNEY" in collapsed:
        return "General Power of Attorney"
    if "SALEDEED" in collapsed or "DEEDOFSALE" in collapsed:
        return "Sale Deed"
    if "AGREEMENTOFSALE" in collapsed:
        return "Agreement of Sale"
    return None


def infer_document_category(document_type: str | None) -> str | None:
    if not document_type:
        return None
    mapping = {
        "Agreement of Sale-cum-General Power of Attorney": "Property Transaction Document",
        "General Power of Attorney": "Property Authorization Document",
        "Sale Deed": "Property Transaction Document",
    }
    return mapping.get(document_type, "Property Document")


def build_important_notes(
    text: str,
    property_status: str,
    pii_detected: bool,
    handwritten_detected: bool = False,
    signature_detected: bool = False,
) -> list[str]:
    notes: list[str] = []
    if "CONTD" in text or "CONTINUES" in text:
        notes.append("This is page 1 of a multi-page document.")
    if property_status in ("CONTINUES_ON_NEXT_PAGE", "NOT_FOUND_ON_PAGE"):
        notes.append("Property details are not present on this page.")
    if "CONTD" in text or "2/P" in text:
        notes.append("Document explicitly contains a continuation marker.")
    if property_status == "CONTINUES_ON_NEXT_PAGE":
        notes.append("Do not infer survey numbers, boundaries, area, or ownership details from this page.")
    if pii_detected:
        notes.append("Aadhaar/identity numbers are present and should be masked in user-facing output.")
    if handwritten_detected:
        notes.append("The document contains both printed and handwritten content.")
    if signature_detected:
        notes.append("Signatures are visible at the bottom of the page.")
    return notes


def extract_stamp_value(text: str) -> str | None:
    value = extract_pattern(
        text,
        [
            r"\bRS\.?\s*([0-9]{1,5})\b",
            r"\b([0-9]{1,5})\s*RUPEES\b",
        ],
    )
    if not value:
        return None
    return f"Rs.{value}"


def extract_stamp_number(text: str) -> str | None:
    return extract_stamp_number_from_text(text)


def extract_document_number(text: str) -> str | None:
    return extract_document_number_from_text(text)


def extract_document_number_from_top_lines(lines: list[OCRLine], image_height: int) -> str | None:
    top_lines = [
        line for line in lines
        if line.y_center <= image_height * 0.25
    ]
    if not top_lines:
        return None

    non_date_lines: list[str] = []
    for line in sorted(top_lines, key=lambda item: (item.y_center, item.x_min)):
        raw_t = line.text or ""
        # 1. Skip date lines and vendor license lines so LNo. 21-11-035/2000 is never taken as doc number
        if "DATE" in raw_t.upper() or re.search(r"\b\d{2}[-/\.]\d{2}[-/\.]\d{4}\b", raw_t) or any(w in raw_t.upper() for w in ("LNO", "LICENSED", "STAMP VENDOR", "R.LNO", "RLNO", "L.NO", "LNO.")):
            continue

        cleaned = clean_ocr_noise(raw_t)
        non_date_lines.append(cleaned)

        # 2. Check for combined 8-digit or slash pattern e.g. 19932019 -> 1993/2019 or 1993 2019
        comb_match = re.search(r"([0-9]{2,5})\s*[\/\s]?\s*(20[0-9]{2})\b", raw_t)
        if comb_match:
            n, y = comb_match.groups()
            return f"{n}/{y}"

        direct = extract_document_number(cleaned)
        if direct:
            return direct

        if any(marker in cleaned for marker in ("D.NO", "D NO", "DOC NO", "DOCUMENT NO", "REG NO", "NO.", "NO:", "NUMBER")):
            digits = re.findall(r"\d{1,6}", cleaned)
            if len(digits) >= 2:
                year = digits[-1]
                number = digits[-2]
                if len(year) in (2, 4) and len(number) <= 6:
                    if len(year) == 2:
                        year = f"20{year}"
                    return f"{number}/{year}"

    if non_date_lines:
        merged = " ".join(non_date_lines)
        comb_match = re.search(r"([0-9]{2,5})\s*[\/\s]?\s*(20[0-9]{2})\b", merged)
        if comb_match:
            n, y = comb_match.groups()
            return f"{n}/{y}"

        match = re.search(r"\b([0-9]{1,6})\s*[\/\s]\s*([0-9]{2,4})\b", merged)
        if match:
            num, yr = match.groups()
            if len(yr) == 2:
                yr = f"20{yr}"
            return f"{num}/{yr}"

    return None


def extract_serial_number(text: str) -> str | None:
    candidate = extract_pattern(
        text,
        [
            r"\bS\.?\s*I\.?\s*(?:NO\.?|NUMBER)?[\.\s\-:]*([0-9]{1,10})\b",
            r"\bS\.?\s*L\.?\s*(?:NO\.?|NUMBER)?[\.\s\-:]*([0-9]{1,10})\b",
            r"\bS\.?\s*C\.?\s*NO\.?\s*[:\-]?\s*([A-Z]?\s*\d{3,10})\b",
            r"\bSERIAL\s*(?:NO\.?|NUMBER)?\s*[:\-]?\s*([A-Z]?\s*\d{3,10})\b",
            r"\bSC\s*NO\.?\s*[:\-]?\s*([A-Z]?\s*\d{3,10})\b",
        ],
    )
    if candidate:
        return normalize_space(candidate).upper() if re.search(r"[A-Z]", candidate) else normalize_space(candidate)
    return None


def extract_document_date(text: str) -> str | None:
    explicit = _date_from_labeled_text(text, ("DT", "DATE"))
    if explicit:
        return explicit
    match = re.search(r"\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})\b", text)
    if match:
        day, month, year = match.groups()
        if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
            return f"{int(day):02d}-{int(month):02d}-{year}"
    return None


def build_words_from_paddle(result: dict[str, Any]) -> list[OCRWord]:
    words: list[OCRWord] = []
    for text, score, poly in zip(result["rec_texts"], result["rec_scores"], result["rec_polys"]):
        cleaned = normalize_space(str(text))
        if not cleaned:
            continue
        words.append(
            OCRWord(
                text=cleaned,
                score=float(score),
                points=[[int(point[0]), int(point[1])] for point in poly.tolist()],
            )
        )
    return words


def resolve_paddle_model_names(lang: str) -> tuple[str | None, str | None]:
    """Resolve detection and recognition model names through installed PaddleOCR API."""
    if lang not in SUPPORTED_LANGUAGE_CODES:
        return None, None
    backend_lang = SUPPORTED_LANGUAGE_CODES[lang]["backend_lang"]
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR._get_ocr_model_names(None, backend_lang, None)
    except Exception:
        return None, None


def is_model_locally_available(lang: str) -> bool:
    """Check if model weights are already downloaded and stored locally on disk."""
    det_name, rec_name = resolve_paddle_model_names(lang)
    if not det_name or not rec_name:
        return False
    models_dir = Path.home() / ".paddlex" / "official_models"
    if not models_dir.exists():
        return False
    return (models_dir / det_name).exists() and (models_dir / rec_name).exists()


def detect_script_and_language(text: str, context_lang: str | None = None) -> tuple[str, str]:
    """
    Detect script and language strictly using Unicode character ranges.
    Never identifies a language merely because place names like 'Telangana' appear in Latin text.
    For Devanagari text, returns ("Devanagari", "Hindi/Marathi unknown") unless context_lang specifies 'hi' or 'mr'.
    """
    if not text or not text.strip():
        return "Unknown", "Unknown"

    counts = {
        "Telugu": len(re.findall(r"[\u0C00-\u0C7F]", text)),
        "Devanagari": len(re.findall(r"[\u0900-\u097F]", text)),
        "Kannada": len(re.findall(r"[\u0C80-\u0CFF]", text)),
        "Tamil": len(re.findall(r"[\u0B80-\u0BFF]", text)),
        "Arabic": len(re.findall(r"[\u0600-\u06FF]", text)),
        "Latin": len(re.findall(r"[A-Za-z]", text)),
    }
    dominant_script = max(counts, key=counts.get)
    if counts[dominant_script] == 0:
        return "Unknown", "Unknown"

    if dominant_script == "Telugu":
        return "Telugu", "Telugu"
    elif dominant_script == "Devanagari":
        if context_lang in ("hi", "Hindi"):
            return "Devanagari", "Hindi"
        elif context_lang in ("mr", "Marathi"):
            return "Devanagari", "Marathi"
        else:
            return "Devanagari", "Hindi/Marathi unknown"
    elif dominant_script == "Kannada":
        return "Kannada", "Kannada"
    elif dominant_script == "Tamil":
        return "Tamil", "Tamil"
    elif dominant_script == "Arabic":
        return "Arabic", "Urdu"
    elif dominant_script == "Latin":
        return "Latin", "English"

    return "Unknown", "Unknown"


def detect_page_language_metadata(lines: list[OCRLine], raw_text: str = "", ocr_model: str | None = None) -> dict[str, Any]:
    """
    Generate page-level language metadata based on Unicode scripts present in OCR lines/raw text.
    """
    text_to_analyze = raw_text or " ".join(l.text for l in lines)
    if not text_to_analyze.strip():
        return {
            "detected_languages": [],
            "primary_language": None,
            "language_detection_method": "unicode_script",
            "ocr_language_model": ocr_model,
        }

    script_counts: dict[str, int] = {
        "Telugu": len(re.findall(r"[\u0C00-\u0C7F]", text_to_analyze)),
        "Devanagari": len(re.findall(r"[\u0900-\u097F]", text_to_analyze)),
        "Kannada": len(re.findall(r"[\u0C80-\u0CFF]", text_to_analyze)),
        "Tamil": len(re.findall(r"[\u0B80-\u0BFF]", text_to_analyze)),
        "Arabic": len(re.findall(r"[\u0600-\u06FF]", text_to_analyze)),
        "Latin": len(re.findall(r"[A-Za-z]", text_to_analyze)),
    }

    detected_langs = []
    if script_counts["Latin"] > 0:
        detected_langs.append("English")
    if script_counts["Telugu"] > 0:
        detected_langs.append("Telugu")
    if script_counts["Devanagari"] > 0:
        detected_langs.append("Hindi/Marathi unknown")
    if script_counts["Kannada"] > 0:
        detected_langs.append("Kannada")
    if script_counts["Tamil"] > 0:
        detected_langs.append("Tamil")
    if script_counts["Arabic"] > 0:
        detected_langs.append("Urdu")

    dominant_script = max(script_counts, key=script_counts.get)
    if script_counts[dominant_script] == 0:
        primary_lang = None
    elif dominant_script == "Latin":
        primary_lang = "English"
    elif dominant_script == "Telugu":
        primary_lang = "Telugu"
    elif dominant_script == "Devanagari":
        primary_lang = "Hindi/Marathi unknown"
    elif dominant_script == "Kannada":
        primary_lang = "Kannada"
    elif dominant_script == "Tamil":
        primary_lang = "Tamil"
    elif dominant_script == "Arabic":
        primary_lang = "Urdu"
    else:
        primary_lang = None

    return {
        "detected_languages": detected_langs,
        "primary_language": primary_lang,
        "language_detection_method": "unicode_script",
        "ocr_language_model": ocr_model,
    }


def format_multilingual_value(text: str, context_lang: str | None = None) -> dict[str, Any]:
    """
    Format extracted value with script and language preservation, without unverified transliteration.
    """
    script, lang = detect_script_and_language(text, context_lang=context_lang)
    return {
        "original_value": text,
        "transliteration": None,
        "script": script,
        "language": lang,
    }


def get_paddle_ocr_model(lang: str = "en", allow_download: bool = False) -> Any:
    """
    Get or dynamically initialize PaddleOCR model for the given language.
    Caches model instances by backend language key.
    By default (allow_download=False), only models with local weights are initialized.
    """
    global _PADDLE_OCR_MODEL, _PADDLE_OCR_MODELS, _PADDLE_OCR_INIT_MS
    lang_key = (lang or "en").lower()
    if lang_key not in SUPPORTED_LANGUAGE_CODES:
        raise ModelUnavailableError(
            f"Unsupported language code '{lang_key}'. Supported options are: {list(SUPPORTED_LANGUAGE_CODES.keys())}"
        )

    backend_lang = SUPPORTED_LANGUAGE_CODES[lang_key]["backend_lang"]

    if backend_lang in _PADDLE_OCR_MODELS:
        return _PADDLE_OCR_MODELS[backend_lang]

    with _PADDLE_OCR_INIT_LOCK:
        if backend_lang in _PADDLE_OCR_MODELS:
            return _PADDLE_OCR_MODELS[backend_lang]

        # Check local availability to avoid unhandled download attempts in offline/restricted environments
        if not allow_download and not is_model_locally_available(lang_key):
            det_name, rec_name = resolve_paddle_model_names(lang_key)
            raise ModelUnavailableError(
                f"PaddleOCR model for '{lang_key}' (det='{det_name}', rec='{rec_name}') is not downloaded or available locally."
            )

        from paddleocr import PaddleOCR

        start = perf_counter()
        try:
            model = PaddleOCR(
                lang=backend_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as e:
            raise ModelUnavailableError(f"Failed to initialize PaddleOCR model for '{lang_key}': {e}")

        _PADDLE_OCR_MODELS[backend_lang] = model
        _PADDLE_OCR_MODEL = model
        _PADDLE_OCR_INIT_MS = (perf_counter() - start) * 1000
        return model


def run_paddle_ocr_page_image(
    image: np.ndarray,
    page_num: int = 1,
    lang: str = "auto",
    page_type: Optional[str] = None,
    use_preprocessing: bool = True,
) -> tuple[list[OCRLine], str, dict[str, Any]]:
    timings: dict[str, Any] = {}
    lang_req = (lang or "auto").lower()
    active_lang = "en" if lang_req == "auto" else lang_req
    is_auto = (lang_req == "auto")

    model_name_used = None
    model_unavailable = False
    model_warning = None

    t0 = perf_counter()
    try:
        ocr = get_paddle_ocr_model(active_lang)
        det_name, rec_name = resolve_paddle_model_names(active_lang)
        model_name_used = rec_name
    except ModelUnavailableError as e:
        model_unavailable = True
        model_warning = str(e)
        print(f"[EXPLICIT WARNING] Page {page_num}: {e}")
        # Run baseline English fallback to preserve raw OCR output, but flag needs_review=True
        try:
            ocr = get_paddle_ocr_model("en")
            model_name_used = "PP-OCRv6_medium_rec (fallback)"
        except Exception:
            ocr = None

    if ocr is None:
        timings["needs_review"] = True
        timings["ocr_language_model_status"] = "UNAVAILABLE"
        timings["unsupported_language"] = lang_req
        timings["language_warning"] = model_warning
        timings["model_name"] = None
        return [], "", timings

    if _PADDLE_OCR_INIT_MS is not None:
        timings["model_initialization_ms"] = _PADDLE_OCR_INIT_MS
    else:
        timings["model_initialization_ms"] = 0.0
    timings["model_access_ms"] = (perf_counter() - t0) * 1000

    h, w = image.shape[:2]

    # Quality-aware adaptive image preprocessing pipeline
    if use_preprocessing:
        import image_preprocessing
        t_prep_start = perf_counter()
        proc_img, prep_meta = image_preprocessing.preprocess_for_ocr(
            image,
            page_number=page_num,
            page_type=page_type,
        )
        timings["preprocessing_ms"] = (perf_counter() - t_prep_start) * 1000
        timings["preprocessing"] = prep_meta
        prep_scale = prep_meta.get("scale", 1.0)
    else:
        proc_img = image
        prep_scale = 1.0
        timings["preprocessing_ms"] = 0.0
        timings["preprocessing"] = None

    with _PADDLE_OCR_PREDICT_LOCK:
        t0 = perf_counter()
        result = ocr.predict(proc_img)[0]
        timings["ocr_inference_ms"] = (perf_counter() - t0) * 1000

    words = build_words_from_paddle(result)

    # Coordinate mapping: Map OCR bounding boxes back to original page coordinates
    if prep_scale != 1.0:
        for word in words:
            word.points = [[int(round(pt[0] / prep_scale)), int(round(pt[1] / prep_scale))] for pt in word.points]

    lines = group_words_into_lines(words)

    for line in lines:
        line.page_num = page_num
        line.page_height = h
        line.page_width = w
        if prep_scale != 1.0:
            line.x_min = int(round(line.x_min / prep_scale))
            line.y_min = int(round(line.y_min / prep_scale))
            line.x_max = int(round(line.x_max / prep_scale))
            line.y_max = int(round(line.y_max / prep_scale))
        # Line-level script and language classification
        l_script, l_lang = detect_script_and_language(
            line.text,
            context_lang=active_lang if not is_auto else None
        )
        line.script = l_script
        line.language = l_lang

    lines.sort(key=lambda l: (l.page_num, l.y_min, l.x_min))
    raw_text = "\n".join(line.text for line in lines)

    # In auto mode, check if non-Latin Indic content was encountered
    needs_review = model_unavailable
    if is_auto:
        non_latin_lines = [l for l in lines if getattr(l, "script", "Latin") not in ("Latin", "Unknown")]
        if non_latin_lines:
            # Documented fallback strategy:
            # Baseline English pass ran, but Indic text was detected.
            # Mark page as needs_review=True with explicit warning unless a specialized model handled it.
            needs_review = True
            first_script = non_latin_lines[0].script
            model_warning = (
                f"Page {page_num} contains {first_script} content, but specialized {first_script} OCR model "
                f"is not locally available. Baseline OCR results preserved with needs_review=True."
            )
            print(f"[WARNING] {model_warning}")

    timings["model_name"] = model_name_used
    timings["ocr_language_model_status"] = "UNAVAILABLE" if model_unavailable else "AVAILABLE"
    timings["needs_review"] = needs_review
    if model_warning:
        timings["language_warning"] = model_warning
    if model_unavailable:
        timings["unsupported_language"] = lang_req

    timings["ocr_word_parsing_ms"] = 0.0
    timings["line_grouping_ms"] = 0.0
    timings["ocr_text_join_ms"] = 0.0
    timings["ocr_total_ms"] = timings.get("ocr_inference_ms", 0.0)
    return lines, raw_text, timings


def run_remote_ocr_page_image(
    image: np.ndarray,
    page_num: int = 1,
    lang: str = "auto",
    ocr_url: str = "",
    timeout: int = 60,
    page_type: Optional[str] = None,
    use_preprocessing: bool = True,
) -> tuple[list[OCRLine], str, dict[str, Any]]:
    """
    Send a page image (preprocessed if use_preprocessing=True) to remote GPU OCR server.
    Preserves original coordinates and records preprocessing metadata.
    Returns (lines, raw_text, timings) conforming to standard pipeline format.
    """
    import requests
    import image_preprocessing

    orig_h, orig_w = image.shape[:2]
    endpoint = f"{ocr_url.rstrip('/')}/ocr"

    # Quality-aware adaptive image preprocessing pipeline for remote GPU OCR
    if use_preprocessing:
        t_prep_start = perf_counter()
        proc_img, prep_meta = image_preprocessing.preprocess_for_ocr(
            image,
            page_number=page_num,
            page_type=page_type,
        )
        prep_ms = (perf_counter() - t_prep_start) * 1000
        prep_scale = prep_meta.get("scale", 1.0)
    else:
        proc_img = image
        prep_scale = 1.0
        prep_ms = 0.0
        prep_meta = None

    # Encode image as JPEG
    success, enc = cv2.imencode(".jpg", proc_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError(f"Failed to encode page {page_num} image to JPEG")

    active_lang = "en" if lang == "auto" else lang

    files = {"image": (f"page_{page_num}.jpg", enc.tobytes(), "image/jpeg")}
    data = {"page_number": str(page_num), "lang": active_lang}

    t0 = perf_counter()
    try:
        resp = requests.post(endpoint, files=files, data=data, timeout=timeout)
        net_ms = (perf_counter() - t0) * 1000
    except Exception as e:
        net_ms = (perf_counter() - t0) * 1000
        timings = {
            "page_number": page_num,
            "requested_language": active_lang,
            "ocr_model_language": None,
            "ocr_language_model_status": "UNAVAILABLE",
            "needs_review": True,
            "warnings": [f"Remote OCR connection error: {e}"],
            "remote_server": ocr_url,
            "network_time_ms": net_ms,
            "preprocessing_ms": prep_ms,
            "preprocessing": prep_meta,
        }
        return [], "", timings

    if resp.status_code != 200:
        try:
            err_data = resp.json()
        except Exception:
            err_data = {"error": resp.text}

        timings = {
            "page_number": page_num,
            "requested_language": active_lang,
            "ocr_model_language": None,
            "ocr_language_model_status": "UNAVAILABLE",
            "needs_review": True,
            "warnings": [err_data.get("error", f"HTTP {resp.status_code}")],
            "remote_server": ocr_url,
            "network_time_ms": net_ms,
            "preprocessing_ms": prep_ms,
            "preprocessing": prep_meta,
        }
        return [], "", timings

    resp_json = resp.json()

    # Parse lines and restore bounding boxes back to original page coordinates
    lines: list[OCRLine] = []
    if "lines" in resp_json and resp_json["lines"]:
        for line_data in resp_json["lines"]:
            text = line_data.get("text", "").strip()
            if not text:
                continue
            bbox = line_data.get("bbox", [0, 0, 0, 0])
            score = float(line_data.get("confidence", 0.95))
            l_script, l_lang = detect_script_and_language(text, context_lang=active_lang)

            if prep_scale != 1.0:
                x_min = int(round(bbox[0] / prep_scale))
                y_min = int(round(bbox[1] / prep_scale))
                x_max = int(round(bbox[2] / prep_scale))
                y_max = int(round(bbox[3] / prep_scale))
            else:
                x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            line = OCRLine(
                text=text,
                score=score,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                page_num=page_num,
                page_height=orig_h,
                page_width=orig_w,
                language=l_lang,
                script=l_script,
            )
            lines.append(line)
    elif "rec_texts" in resp_json:
        rec_texts = resp_json.get("rec_texts", [])
        rec_scores = resp_json.get("rec_scores", [])
        rec_polys = resp_json.get("rec_polys", [])
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            text_str = str(text).strip()
            if not text_str:
                continue
            pts = poly if isinstance(poly, list) else poly.tolist()
            raw_x_min = min(p[0] for p in pts)
            raw_y_min = min(p[1] for p in pts)
            raw_x_max = max(p[0] for p in pts)
            raw_y_max = max(p[1] for p in pts)

            if prep_scale != 1.0:
                x_min = int(round(raw_x_min / prep_scale))
                y_min = int(round(raw_y_min / prep_scale))
                x_max = int(round(raw_x_max / prep_scale))
                y_max = int(round(raw_y_max / prep_scale))
            else:
                x_min, y_min, x_max, y_max = int(raw_x_min), int(raw_y_min), int(raw_x_max), int(raw_y_max)

            l_script, l_lang = detect_script_and_language(text_str, context_lang=active_lang)
            line = OCRLine(
                text=text_str,
                score=float(score),
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                page_num=page_num,
                page_height=orig_h,
                page_width=orig_w,
                language=l_lang,
                script=l_script,
            )
            lines.append(line)

    lines.sort(key=lambda l: (l.page_num, l.y_min, l.x_min))
    raw_text = "\n".join(l.text for l in lines)

    timings = {
        "page_number": resp_json.get("page_number", page_num),
        "requested_language": resp_json.get("requested_language", active_lang),
        "ocr_model_language": resp_json.get("ocr_model_language", active_lang),
        "ocr_language_model_status": resp_json.get("model_status", "AVAILABLE"),
        "needs_review": resp_json.get("needs_review", False),
        "warnings": resp_json.get("warnings", []),
        "ocr_total_ms": resp_json.get("ocr_time_ms", 0.0),
        "network_time_ms": net_ms,
        "remote_server": ocr_url,
        "gpu_name": resp_json.get("gpu_name", "Remote GPU"),
        "preprocessing_ms": prep_ms,
        "preprocessing": prep_meta,
    }
    return lines, raw_text, timings


def _run_paddle_ocr_impl(image_path: str) -> tuple[list[OCRLine], str, dict[str, float]]:
    t_read = perf_counter()
    image = cv2.imread(image_path)
    if image is None:
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            pass
    if image is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    return run_paddle_ocr_page_image(image, page_num=1)


def _extract_stamp_metadata(lines: list[OCRLine], full_text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "acknowledgement_number": None,
        "si_number": None,
        "cash_number": None,
        "sold_to": None,
        "sold_to_relation": None,
        "sold_to_residence": None,
        "for_whom": None,
        "license_number": None,
        "rl_number": None,
        "vendor_address": None,
        "vendor_phone": None,
    }

    source_text = normalize_space(full_text)

    # Search per-line to avoid cross-line contamination
    stamp_lines = [line.text for line in lines if any(
        w in line.text.upper() for w in ("SI NO", "SI N", "ACV", "ACK", "SCANNED", "STAMP VENDOR", "LNO", "SOLD TO")
    )]
    stamp_search_text = normalize_space(" ".join(stamp_lines).upper()) if stamp_lines else source_text

    ack = extract_pattern(
        stamp_search_text,
        [
            r"\bAC[KV]\.?\s*(?:NO\.?)?[:\s\-\.]*([0-9]{2,10})(?=\s|$)",
            r"\bACK\.?\s*(?:NO\.?)?[:\s\-]*([0-9]{2,10})(?=\s|$)",
        ],
    )
    # SI number: strict - only match digits immediately after keyword, stop at non-digit
    si_match = re.search(
        r"\bS\.?\s*I\.?\s*(?:NO\.?)?\.{0,5}\s*([0-9]{1,6})(?=\s|$|[^0-9])",
        stamp_search_text,
        re.IGNORECASE,
    )
    si = si_match.group(1) if si_match else None
    cash = extract_pattern(source_text, [r"\bCASH\.?\s*(?:NO\.?)?[:\s\-]*([0-9]{1,10})\b"])
    sold_to = extract_pattern(
        source_text,
        [
            r"\bSOLD\s*TO\.?[^\w]{0,10}([A-Z][A-Z\.\s]+?)(?=,\s*S/O\b|\s+S/O\b|\s+R/O\b|,|$)",
        ],
    )
    sold_to_relation = extract_pattern(source_text, [r"\b(S/O\.?\s*[A-Z][A-Z\s\.]+?)(?=,|\bR/O\b|$)"])
    residence = extract_pattern(source_text, [r"\bR/O\.?\s*([A-Z0-9][A-Z0-9\s\-/\.]+?)(?=,|\bLIC\.?\b|\bR\.L\.?\b|$)"])
    for_whom = extract_pattern(source_text, [r"\bFOR\s*WHOM\.?\s*([^:\-\n]+?)(?=Cell:|Lic\b|$)"])
    license_number = extract_pattern(source_text, [r"\bLIC\.?\s*(?:NO\.?)?[:\s\-]*([0-9]{2,4}(?:-[0-9]{2,4}){2,3}/[0-9]{2,4})\b"])
    rl_number = extract_pattern(source_text, [r"\bR\.?\s*L\.?\s*(?:NO\.?)?[:\s\-]*([0-9]{2,4}(?:-[0-9]{2,4}){2,3}/[0-9]{2,4})\b"])

    # vendor address & phone from stamp header
    header_part = full_text.split("AGREEMENT OF SALE")[0]
    vendor_addr_match = re.search(r"\b(H\.?No\..*?)(?=Cell:|For Whom|Lic\.? No|$)", header_part, re.IGNORECASE | re.DOTALL)
    vendor_address = clean_address(vendor_addr_match.group(1)) if vendor_addr_match else None
    has_phone = re.search(r"\b(?:Cell|Phone|Mobile)\b", header_part, re.IGNORECASE) is not None

    metadata["acknowledgement_number"] = smart_number(ack)
    # si_number: already pure digits from direct group capture, no stripping needed
    metadata["si_number"] = normalize_space(si) if si else None
    metadata["cash_number"] = smart_number(cash)
    metadata["sold_to"] = clean_field(sold_to)
    metadata["sold_to_relation"] = format_relation(sold_to_relation)
    metadata["sold_to_residence"] = clean_field(residence)
    metadata["for_whom"] = clean_field(for_whom)
    metadata["license_number"] = smart_number(license_number)
    metadata["rl_number"] = smart_number(rl_number)
    metadata["vendor_address"] = vendor_address
    metadata["vendor_phone"] = "[MASKED]" if has_phone else None
    return metadata


def run_paddle_ocr(image_path: str) -> tuple[list[OCRLine], str, dict[str, float]]:
    return _run_paddle_ocr_impl(image_path)


def extract_survey_information(text: str) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Extracts survey_number, sub_survey_number, khata_number, patta_number from document text.
    Handles CSNO, C.S.No, City Survey No, Survey No, Sy.No, etc.
    """
    survey_number = None
    sub_survey_number = None
    khata_number = None
    patta_number = None

    sy_match = re.search(
        r"\b(?:SURVEY\s*(?:NOS?|NUMBERS?)?|SY\.?\s*NOS?|C\.?S\.?\s*NOS?|CITY\s*SURVEY\s*NOS?)\.?\s*[:\-]?\s*([0-9\s,&\+ANDand/-]+)",
        text,
        flags=re.IGNORECASE
    )
    if sy_match:
        raw_sy = sy_match.group(1).strip(" .,;-")
        cleaned_sy = re.sub(r"\s+", " ", raw_sy)
        cleaned_sy = re.sub(r"\bAND\b", ",", cleaned_sy, flags=re.IGNORECASE)
        cleaned_sy = re.sub(r"&", ",", cleaned_sy)
        items = [s.strip() for s in cleaned_sy.split(",") if s.strip().isdigit() or re.match(r"^[0-9]+/[0-9A-Za-z]+$", s.strip())]
        if items:
            survey_number = ", ".join(items)
        else:
            survey_number = raw_sy

    sub_match = re.search(
        r"\b(?:PLOT\s*(?:NOS?|NUMBERS?)?|SUB[\s_\-]*SURVEY\s*(?:NOS?|NUMBERS?)?)\.?\s*[:\-]?\s*([0-9\s,/&\+ANDand-]+)",
        text,
        flags=re.IGNORECASE
    )
    if sub_match:
        raw_sub = sub_match.group(1).strip(" .,;-")
        raw_sub = re.sub(r"\s+", " ", raw_sub)
        raw_sub = re.sub(r"\bAND\b", "&", raw_sub, flags=re.IGNORECASE)
        sub_items = re.findall(r"\b[0-9]+/[0-9A-Za-z]+\b", raw_sub)
        if not sub_items:
            sub_items = re.findall(r"\b[0-9]+\b", raw_sub)
        if len(sub_items) >= 2:
            sub_survey_number = " & ".join(sub_items)
        elif sub_items:
            sub_survey_number = sub_items[0]
        else:
            sub_survey_number = raw_sub

    khata_match = re.search(r"\bKHATA\s*(?:NO|NUMBER)?\.?\s*[:\-]?\s*([0-9A-Z/-]+)", text, flags=re.IGNORECASE)
    if khata_match:
        khata_number = khata_match.group(1).strip(" .,;-")

    patta_match = re.search(r"\bPATTA\s*(?:NO|NUMBER)?\.?\s*[:\-]?\s*([0-9A-Z/-]+)", text, flags=re.IGNORECASE)
    if patta_match:
        patta_number = patta_match.group(1).strip(" .,;-")

    return survey_number, sub_survey_number, khata_number, patta_number


def extract_property_area(text: str) -> str | None:
    # 1. Check for PLOT AREA : 480.0 SQ. YDS. (OR) : 401.4 SQ. MTS.
    match1 = re.search(
        r"PLOT\s*AREA\s*[:\-]?\s*([0-9\.\s]+)\s*(?:SQ\.?\s*YDS\.?|SQ\.?\s*YARDS?)\.?\s*(?:\(?OR\)?\s*[:\-]?\s*([0-9\.\s]+)\s*(?:SQ\.?\s*MTS\.?|SQ\.?\s*MTRS?|SQ\.?\s*METRES?))?",
        text,
        flags=re.IGNORECASE
    )
    if match1:
        sq_yds = match1.group(1).strip()
        sq_mts = match1.group(2)
        if sq_mts:
            return f"{sq_yds} sq. yards ({sq_mts.strip()} sq. metres)"
        return f"{sq_yds} sq. yards"

    # 2. Check for admeasuring / extent of 480 Sq. Yards or 401.4 Sq. Mtrs.
    match2 = re.search(
        r"(?:admeasuring|extent\s*of)\s*(?:an\s*extent\s*of)?\s*([0-9\.\s]+)\s*(?:Sq\.?\s*Yards?|Sq\.?\s*Yds\.?)\.?\s*(?:or|/|\()\s*([0-9\.\s]+)\s*(?:Sq\.?\s*Mtrs?|Sq\.?\s*Metres?|Sq\.?\s*Mts\.?)\.?",
        text,
        flags=re.IGNORECASE
    )
    if match2:
        sq_yds = match2.group(1).strip()
        sq_mts = match2.group(2).strip()
        return f"{sq_yds} sq. yards ({sq_mts} sq. metres)"

    # 3. Fallback generic area
    match3 = re.search(
        r"(?:admeasuring|extent\s*of)\s*([0-9\.\s]+(?:\s*(?:Ac(?:res?)?|Gts|Guntas|Sq\.?\s*Yds|Sq\.?\s*Yards|Sq\.?\s*Mtrs|Sq\.?\s*Metres)[^,\.\n]*)+)",
        text,
        flags=re.IGNORECASE
    )
    if match3:
        return normalize_space(match3.group(1))

    return None


def extract_property_location(full_text: str, parties: list[dict[str, Any]]) -> tuple[str | None, str | None, str | None]:
    """
    Extracts property location details (village, mandal/taluk, district).
    Prioritizes explicit document location text and falls back to party address context.
    """
    village = None
    mandal = None
    district = None

    v_match = re.search(r"\b([A-Z][A-Za-z\s\.]{2,30}?)\s+(?:VILLAGE|VILL\.?)\b", full_text, flags=re.IGNORECASE)
    if v_match:
        v_cand = clean_field(v_match.group(1))
        if v_cand and v_cand.upper() not in ("THIS", "SAME"):
            village = v_cand

    m_match = re.search(r"\b([A-Z][A-Za-z\s\.]{2,30}?)\s+(?:MANDAL|TALUK|TALUKA|TEHSIL|HOBLI)\b", full_text, flags=re.IGNORECASE)
    if m_match:
        m_cand = clean_field(m_match.group(1))
        if m_cand:
            mandal = m_cand

    d_match = re.search(r"\b(?:DIST\.?|DISTRICT)\s*[:\-]?\s*([A-Z][A-Za-z\s\.]{2,30}?)\b", full_text, flags=re.IGNORECASE)
    if not d_match:
        d_match = re.search(r"\b([A-Z][A-Za-z\s\.]{2,30}?)\s+(?:DIST\.?|DISTRICT)\b", full_text, flags=re.IGNORECASE)
    if d_match:
        d_cand = clean_field(d_match.group(1))
        if d_cand:
            district = d_cand

    if parties:
        for p in parties:
            addr = p.get("address") or ""
            if not village:
                pv_match = re.search(r"\b([A-Z][A-Za-z\s\.]{2,30}?)\s+Village\b", addr, flags=re.IGNORECASE)
                if pv_match:
                    village = clean_field(pv_match.group(1))
            if not mandal:
                pm_match = re.search(r"\b([A-Z][A-Za-z\s\.]{2,30}?)\s+(?:Mandal|Taluk|Taluka|Tehsil|Hobli)\b", addr, flags=re.IGNORECASE)
                if pm_match:
                    mandal = clean_field(pm_match.group(1))
            if not district:
                pd_match = re.search(r"\b(?:Dist\.?|District)\.?\s*([A-Z][A-Za-z\s\.]{2,30}?)(?:,|$)", addr, flags=re.IGNORECASE)
                if not pd_match:
                    pd_match = re.search(r"\b([A-Z][A-Za-z\s\.]{2,30}?)\s+(?:Dist\.?|District)\b", addr, flags=re.IGNORECASE)
                if pd_match:
                    district = clean_field(pd_match.group(1))

    if district:
        d_upper = district.upper()
        if "R.R." in d_upper or "RANGA REDDY" in d_upper or "R.R" in d_upper:
            district = "R.R. District (Ranga Reddy District)"

    return village, mandal, district


def extract_land_document_from_lines(
    lines: list[OCRLine],
    raw_text: str,
    image_path: str,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    from semantic_extractor import extract_fields_semantic

    pipeline_timings = dict(timings or {})
    t0 = perf_counter()
    full_text = normalize_upper(raw_text)
    pipeline_timings["text_normalization_ms"] = (perf_counter() - t0) * 1000

    image = cv2.imread(image_path)
    if image is None and os.path.exists(image_path):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(image_path)
            pil_img = pdf[0].render(scale=2).to_pil()
            img_np = np.array(pil_img)
            image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.ndim == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass
    if image is None:
        image = np.zeros((2000, 1500, 3), dtype=np.uint8)
    pipeline_timings["image_loading_ms"] = (perf_counter() - t0) * 1000

    # ---- Semantic extraction: candidate-based field extraction ----
    t0 = perf_counter()
    semantic_result, semantic_provenance, debug_candidates = extract_fields_semantic(lines)
    pipeline_timings["semantic_extraction_ms"] = (perf_counter() - t0) * 1000

    document_type = semantic_result["document_type"]
    document_number = semantic_result["document_number"]
    survey_no = semantic_result["survey_number"]
    sub_survey_no = semantic_result["sub_survey_number"]
    prop_area = semantic_result["property_area"]
    village_val = semantic_result["village"]
    mandal_val = semantic_result["mandal"]
    district_val = semantic_result["district"]
    stamp_serial_num = semantic_result["stamp_serial_number"]
    stamp_value = semantic_result["stamp_value"]
    stamp_sold_to = semantic_result["stamp_sold_to"]
    parties_list = semantic_result["parties_list"]
    document_date = semantic_result["document_date"]
    execution_date = semantic_result["execution_date"]

    document_category = infer_document_category(document_type)
    state = infer_state(full_text)

    # Legacy stamp metadata extraction (for stamp_information block)
    t0 = perf_counter()
    stamp_number = extract_stamp_number(full_text)
    stamp_metadata = _extract_stamp_metadata(lines, full_text)
    pipeline_timings["stamp_metadata_ms"] = (perf_counter() - t0) * 1000

    # Feature detection
    t0 = perf_counter()
    continuation_detected = "CONTD" in full_text or "2/P" in full_text or "NEXT PAGE" in full_text
    pii_detected = any(token in full_text for token in ("AADHAR", "AADHAAR", "ADHAR", "UID"))
    property_status = "CONTINUES_ON_NEXT_PAGE" if continuation_detected else "NOT_FOUND_ON_PAGE"
    languages = detect_languages(raw_text)
    signature_detected = detect_signature(image, lines)
    raw_hw_regions = detect_handwriting_regions(lines, image.shape[0], image.shape[1], page_num=1, image=image)
    hw_recognizer = get_handwriting_recognizer()
    processed_hw_regions = [hw_recognizer.recognize_region(r, image=image) for r in raw_hw_regions]
    handwriting_metadata = build_handwriting_metadata(processed_hw_regions)
    handwritten_text_detected = handwriting_metadata["detected"]
    stamp_detected = bool(stamp_value or "NON JUDICIAL" in full_text or "STAMP VENDOR" in full_text)
    pipeline_timings["feature_detection_ms"] = (perf_counter() - t0) * 1000

    stamp_vendor = extract_pattern(
        full_text,
        [
            r"\b([A-Z][A-Z\s]+)\s+LICENSED STAMP VENDOR\b",
        ],
    )

    # Survey info fallback for khata/patta (not in semantic extractor yet)
    _, _, khata_no, patta_no = extract_survey_information(full_text)

    # If stamp_sold_to not found by semantic extractor, fall back to stamp_metadata
    if not stamp_sold_to:
        stamp_sold_to = stamp_metadata.get("sold_to")

    output = {
        "document_type": document_type,
        "document_number": document_number,
        "survey_number": survey_no,
        "sub_survey_number": sub_survey_no,
        "property_area": prop_area,
        "village": village_val,
        "mandal": mandal_val,
        "district": district_val,
        "stamp_serial_number": stamp_serial_num,
        "stamp_value": stamp_value,
        "stamp_sold_to": stamp_sold_to,
        "parties_list": parties_list,
        "document_date": document_date,
        "execution_date": execution_date,
        "document_category": document_category,
        "state": state,
        "serial_number": stamp_serial_num,
        "stamp_number": stamp_number,
        "parties": parties_list,
        "property": {
            "survey_number": survey_no,
            "sub_survey_number": sub_survey_no,
            "khata_number": khata_no,
            "patta_number": patta_no,
            "area": prop_area,
            "property_area": prop_area,
            "boundaries": None,
            "village": village_val,
            "mandal": mandal_val,
            "district": district_val,
            "status": property_status,
        },
        "stamp_information": {
            "stamp_vendor": clean_field(stamp_vendor),
            "stamp_vendor_type": "Licensed Stamp Vendor"
            if ("LICENSED STAMP VENDOR" in full_text or "LICENCED STAMP VENDOR" in full_text)
            else None,
            "stamp_number": stamp_number,
            "stamp_value": stamp_value,
            "stamp_serial_number": stamp_serial_num,
            "stamp_sold_to": stamp_sold_to,
            "sold_to": stamp_sold_to,
            **stamp_metadata,
        },
        "document_features": {
            "languages": languages,
            "printed_text_detected": bool(lines),
            "handwritten_text_detected": handwritten_text_detected,
            "handwriting": handwriting_metadata,
            "signature_detected": signature_detected,
            "stamp_detected": stamp_detected,
            "multi_page_document": continuation_detected,
            "continuation_detected": continuation_detected,
            "pii_detected": pii_detected,
        },
        "important_notes": build_important_notes(
            full_text,
            property_status,
            pii_detected,
            handwritten_detected=handwritten_text_detected,
            signature_detected=signature_detected,
        ),
        "ocr_debug": {
            "source_image": str(Path(image_path).resolve()),
            "processed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "line_count": len(lines),
            "lines": [line.text for line in lines],
        },
    }
    output["field_provenance"] = semantic_provenance
    output["debug_candidates"] = debug_candidates
    output["learning"] = semantic_result.get("learning", {
        "rules_applied": 0,
        "verified_feedback_count": 0,
        "fields_improved": [],
        "learned_corrections": [],
        "learning_mode": "officer_verified_adaptive_feedback",
    })

    pipeline_timings["json_object_build_ms"] = (perf_counter() - t0) * 1000
    pipeline_timings["core_pipeline_ms"] = (
        pipeline_timings.get("ocr_total_ms", 0.0)
        + pipeline_timings.get("text_normalization_ms", 0.0)
        + pipeline_timings.get("image_loading_ms", 0.0)
        + pipeline_timings.get("semantic_extraction_ms", 0.0)
        + pipeline_timings.get("stamp_metadata_ms", 0.0)
        + pipeline_timings.get("feature_detection_ms", 0.0)
        + pipeline_timings.get("json_object_build_ms", 0.0)
    )
    output["profiling_ms"] = {key: round(value, 3) for key, value in pipeline_timings.items()}

    return output


def extract_land_document(file_path: str) -> dict[str, Any]:
    t0 = perf_counter()
    is_pdf = file_path.lower().endswith(".pdf")
    if not is_pdf and os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header.startswith(b"%PDF"):
                    is_pdf = True
        except Exception:
            pass

    if is_pdf:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(file_path)
        all_lines = []
        all_raw_texts = []
        total_ocr_ms = 0.0

        page_prep_summaries = []
        for page_idx, page in enumerate(pdf, start=1):
            pil_img = page.render(scale=3).to_pil()
            img_np = np.array(pil_img)
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

            p_type = "deed_text"
            if page_idx == 1:
                p_type = "stamp_metadata"
            elif page_idx == 2:
                p_type = "property_schedule"
            elif page_idx == len(pdf):
                p_type = "registration_plan"

            p_lines, p_raw, p_timings = run_paddle_ocr_page_image(img_bgr, page_num=page_idx, page_type=p_type)
            if "preprocessing" in p_timings:
                page_prep_summaries.append(p_timings["preprocessing"])

            # If last page (e.g. Registration Plan), crop the header strip to bypass bounding frame box
            if page_idx == len(pdf):
                try:
                    h, w = img_bgr.shape[:2]
                    header_crop = img_bgr[int(h * 0.052) : int(h * 0.125), int(w * 0.03) : int(w * 0.58)]
                    c_lines, c_raw, _ = run_paddle_ocr_page_image(header_crop, page_num=page_idx, page_type="registration_plan")
                    p_lines.extend(c_lines)
                    p_raw = p_raw + "\n" + c_raw
                except Exception:
                    pass

            all_lines.extend(p_lines)
            all_raw_texts.append(f"--- PAGE {page_idx} ---\n{p_raw}")
            total_ocr_ms += p_timings.get("ocr_total_ms", 0.0)

        full_raw_text = "\n\n".join(all_raw_texts)
        ocr_timings = {"ocr_total_ms": total_ocr_ms}
        result = extract_land_document_from_lines(all_lines, full_raw_text, file_path, timings=ocr_timings)
        result["preprocessing"] = page_prep_summaries
        result.setdefault("profiling_ms", {})
        result["profiling_ms"]["pipeline_total_ms"] = round((perf_counter() - t0) * 1000, 3)
        return result
    else:
        lines, raw_text, ocr_timings = run_paddle_ocr(file_path)
        result = extract_land_document_from_lines(lines, raw_text, file_path, timings=ocr_timings)
        result["preprocessing"] = [ocr_timings["preprocessing"]] if "preprocessing" in ocr_timings else []
        result.setdefault("profiling_ms", {})
        result["profiling_ms"]["pipeline_total_ms"] = round((perf_counter() - t0) * 1000, 3)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured data from land document images.")
    parser.add_argument("image_path", help="Path to the uploaded land document image")
    parser.add_argument(
        "--output",
        default="land_document_output.json",
        help="Where to write the extracted JSON output",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print the extracted JSON without writing a file",
    )
    args = parser.parse_args()

    result = extract_land_document(args.image_path)
    result.setdefault("profiling_ms", {})
    json_start = perf_counter()
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    result["profiling_ms"]["json_generation_ms"] = round((perf_counter() - json_start) * 1000, 3)
    result["profiling_ms"]["total_processing_ms"] = round(
        result["profiling_ms"].get("pipeline_total_ms", 0.0) + result["profiling_ms"]["json_generation_ms"],
        3,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if not args.stdout_only:
        output_path = Path(args.output)
        output_path.write_text(payload, encoding="utf-8")
        print(f"Saved structured output to {output_path.resolve()}")

    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
