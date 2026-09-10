import time
import cv2
import numpy as np
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument(r"C:\Users\meesa\Downloads\telangana_official_land_document.pdf")

# Test rendering at scale 1.5 vs 2.0
t0 = time.perf_counter()
img_1_5 = np.array(pdf[0].render(scale=1.5).to_pil())
t_render_1_5 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
img_2_0 = np.array(pdf[0].render(scale=2.0).to_pil())
t_render_2_0 = (time.perf_counter() - t0) * 1000

print(f"Scale 1.5 shape: {img_1_5.shape}, render time: {t_render_1_5:.1f}ms")
print(f"Scale 2.0 shape: {img_2_0.shape}, render time: {t_render_2_0:.1f}ms")
