import sys
import os
import json
import socket
import time

sys.path.insert(0, ".")

from land_document_extractor import OCRLine, extract_land_document_from_lines
from gis_service import verify_gis_location, parse_dimensions_from_text, get_local_dataset
from web_app import render_gis_section

print("==========================================================")
print("FINAL USER-FACING RUNTIME VALIDATION & REGRESSION SUITE")
print("==========================================================")

# 1. Test Real Extraction Pipeline with Location Lines
karnataka_lines = [
    OCRLine(text="SALE DEED", score=0.98, x_min=0, y_min=10, x_max=100, y_max=30),
    OCRLine(text="GOVERNMENT OF KARNATAKA", score=0.99, x_min=0, y_min=35, x_max=100, y_max=55),
    OCRLine(text="Document No: 5412/2021", score=0.96, x_min=0, y_min=60, x_max=100, y_max=80),
    OCRLine(text="Devalapura Village, Nagamangala Taluk", score=0.95, x_min=0, y_min=85, x_max=100, y_max=115),
    OCRLine(text="Dist: Mandya", score=0.95, x_min=0, y_min=120, x_max=100, y_max=135),
    OCRLine(text="Survey Number: 142/3A", score=0.97, x_min=0, y_min=140, x_max=100, y_max=155),
    OCRLine(text="Measurement East-West 60 ft, North-South 40 ft", score=0.96, x_min=0, y_min=160, x_max=100, y_max=180),
]

raw_text_ka = "\n".join(line.text for line in karnataka_lines)
extracted_doc = extract_land_document_from_lines(karnataka_lines, raw_text_ka, "sample_document.png")
# Ensure dimensions are present on property to test dual-layer Leaflet rendering
extracted_doc["property"]["dimensions"] = "East to West 60 ft, North to South 40 ft"
extracted_doc["property"]["boundaries"] = "East: Road, West: Plot 5, North: Plot 12, South: Lane"

print("\n--- 1. Extracted OCR Fields ---")
print("Document Type:", extracted_doc.get("document_type"))
print("State:", extracted_doc.get("state"))
print("District:", extracted_doc.get("property", {}).get("district"))
print("Taluk/Mandal:", extracted_doc.get("property", {}).get("taluk") or extracted_doc.get("property", {}).get("mandal"))
print("Village:", extracted_doc.get("property", {}).get("village"))
print("Survey Number:", extracted_doc.get("property", {}).get("survey_number"))
print("Dimensions:", extracted_doc.get("property", {}).get("dimensions"))

# 2. Pass OCR Output into GIS Resolver
t0 = time.time()
gis_res = verify_gis_location(extracted_doc)
t_resolve = time.time() - t0

print("\n--- 2. GIS Resolution Result ---")
print(f"Resolution Time: {t_resolve*1000:.2f} ms")
print("Status:", gis_res.get("status"))
print("Resolution Level:", gis_res.get("resolution_level"))
print("Village Status:", gis_res.get("village_status"))
print("Coordinates (Dynamic Centroid):", gis_res.get("coordinates"))
print("Confidence:", gis_res.get("gis_resolution_confidence"))
print("Source Attribution:", gis_res.get("source_attribution"))
print("Dimensions Area:", (gis_res.get("dimensions") or {}).get("area_sqm"), "m²")
print("Cadastral Status:", gis_res.get("cadastral_status"))
print("Survey Geometry:", gis_res.get("survey_geometry"))

# 3. Render Leaflet Map HTML Section
t1 = time.time()
html_output = render_gis_section(extracted_doc)
t_html = time.time() - t1

print("\n--- 3. Web App Leaflet HTML Card ---")
print(f"HTML Render Time: {t_html*1000:.2f} ms")
print("HTML Output Length:", len(html_output), "bytes")
print("Contains Leaflet Script?:", "leaflet.js" in html_output)
print("Contains Map Container (#gis-map)?:", "id=\"gis-map\"" in html_output)
print("Contains Blue Source Layer?:", "adminGeojson" in html_output and "#2563eb" in html_output)
print("Contains Green Parcel Layer?:", "parcelGeojson" in html_output and "#059669" in html_output)
print("Contains Cadastral Disclaimer?:", "authoritative cadastral boundary" in html_output.lower())
assert "authoritative cadastral boundary" in html_output.lower()

# 4. Ambiguity Protection Test
print("\n--- 4. Ambiguity Protection Test ---")
ambig_doc = {"state": "Karnataka", "property": {"village": "Devalapura"}}
res_ambig = verify_gis_location(ambig_doc)
print("Ambiguous Village Status:", res_ambig.get("status"))
print("Ambiguous Village Message:", res_ambig.get("message"))
assert res_ambig.get("status") == "ambiguous"

# 5. Telangana Safety Test
print("\n--- 5. Telangana Safety Test ---")
ts_doc = {"state": "Telangana", "property": {"district": "Sangareddy", "mandal": "Kandi", "village": "Kandi"}}
res_ts = verify_gis_location(ts_doc)
print("Telangana Status:", res_ts.get("status"))
print("Telangana Resolution Level:", res_ts.get("resolution_level"))
print("Telangana Village Status:", res_ts.get("village_status"))
print("Telangana Disclaimer:", res_ts.get("village_disclaimer"))
assert res_ts.get("village_status") == "NOT_AVAILABLE"

# 6. Offline Network Independence Test
print("\n--- 6. Offline Network Independence Test ---")
orig_socket = socket.socket
try:
    socket.socket = None
    offline_res = verify_gis_location(extracted_doc)
    print("Offline Resolution Status:", offline_res.get("status"))
    print("Offline Resolution Level:", offline_res.get("resolution_level"))
    assert offline_res.get("status") == "resolved"
finally:
    socket.socket = orig_socket

print("\nAll Runtime Validations Successfully Passed!")
