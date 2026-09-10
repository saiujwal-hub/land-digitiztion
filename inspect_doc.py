import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import cv2
import pypdfium2 as pdfium
import numpy as np

data = json.load(open('telangana_document_extraction.json', encoding='utf-8'))

print("=== PAGE 2 DETAILED OCR ELEMENTS ===")
for el in data['pages_ocr'][1]['ocr_elements']:
    bbox = el['bounding_box']
    print(f"y={bbox[1]}-{bbox[3]}, conf={el['confidence']:.3f}: {el['text']}")

# Let's check the gap between y of line 35 and 36
pdf = pdfium.PdfDocument(r'C:\Users\meesa\Downloads\telangana_official_land_document.pdf')
img2 = cv2.cvtColor(np.array(pdf[1].render(scale=2.0).to_pil()), cv2.COLOR_RGB2BGR)

# Crop the recital area
# Let's crop from y=950 to 1250 across full width
crop = img2[950:1250, 200:1300]
cv2.imwrite('page2_recital_crop.png', crop)
print("Saved page2_recital_crop.png")
