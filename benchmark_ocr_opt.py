import time
import cv2
import numpy as np
import pypdfium2 as pdfium
from paddleocr import PaddleOCR

pdf = pdfium.PdfDocument(r"C:\Users\meesa\Downloads\telangana_official_land_document.pdf")
img = np.array(pdf[0].render(scale=1.5).to_pil())
bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

print("Testing PaddleOCR with text_det_limit_side_len=960...")
t0 = time.perf_counter()
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=960,
    text_recognition_batch_size=8,
)
init_time = time.perf_counter() - t0
print(f"Init time: {init_time:.2f}s")

t0 = time.perf_counter()
res = ocr.predict(bgr)[0]
infer_time = time.perf_counter() - t0
print(f"Inference time (det_limit=960, scale=1.5): {infer_time:.2f}s, found {len(res['rec_texts'])} texts")
for text, score in zip(res['rec_texts'][:8], res['rec_scores'][:8]):
    print(f"  [{score:.2f}] {text}")
