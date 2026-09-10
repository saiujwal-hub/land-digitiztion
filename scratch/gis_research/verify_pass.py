import sys
import os
import json
import time
import psutil

sys.path.insert(0, ".")
from gis_service import verify_gis_location

print("=== SECTION 1, 2, 3 & 8: E2E VERIFICATION ===")

process = psutil.Process(os.getpid())
ram_before = process.memory_info().rss / (1024 * 1024)

t0 = time.time()
naga_ocr = {"state": "Karnataka", "property": {"district": "Mandya", "mandal": "Nagamangala", "village": "Devalapura"}}
res_naga = verify_gis_location(naga_ocr)
t_first = time.time() - t0

ram_after = process.memory_info().rss / (1024 * 1024)

print(f"First GIS Lookup Time (incl. dataset load & indexing): {t_first:.4f}s")
print(f"RAM Before: {ram_before:.2f} MB, RAM After: {ram_after:.2f} MB (Delta: {ram_after - ram_before:.2f} MB)")

t1 = time.time()
tumk_ocr = {"state": "Karnataka", "property": {"district": "Tumkur", "mandal": "Tumkur", "village": "Devalapura"}}
res_tumk = verify_gis_location(tumk_ocr)
t_repeated = time.time() - t1

print(f"Repeated GIS Lookup Time: {t_repeated*1000:.4f} ms")

print("\n--- Devalapura Hierarchy Tests ---")
print("Nagamangala Devalapura:", res_naga["status"], res_naga["resolution_level"], res_naga["coordinates"])
print("Tumkur Devalapura:", res_tumk["status"], res_tumk["resolution_level"], res_tumk["coordinates"])

ambig_ocr = {"state": "Karnataka", "property": {"village": "Devalapura"}}
res_ambig = verify_gis_location(ambig_ocr)
print("Village-only Devalapura Status:", res_ambig["status"])

print("Geometries Distinct?:", res_naga["coordinates"] != res_tumk["coordinates"])
