import os
os.environ.setdefault('HUB_DATASET_ENDPOINT', 'https://modelscope.cn/api/v1/datasets')
os.environ.setdefault('FLAGS_use_mkldnn', '0')
os.environ.setdefault('PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT', '0')

import time
import cv2
import numpy as np
import pypdfium2 as pdfium
from paddleocr import PaddleOCR

pdf = pdfium.PdfDocument(r"C:\Users\meesa\Downloads\telangana_official_land_document.pdf")
img_1_5 = np.array(pdf[0].render(scale=1.5).to_pil())
bgr_1_5 = cv2.cvtColor(img_1_5, cv2.COLOR_RGB2BGR)

print(f"Testing Page 1 at scale=1.5, shape={bgr_1_5.shape}...")
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

t0 = time.perf_counter()
res = ocr.predict(bgr_1_5)[0]
t_infer = time.perf_counter() - t0
print(f"Scale 1.5 single pass inference: {t_infer:.2f}s, found {len(res['rec_texts'])} texts")
for t, s in zip(res['rec_texts'][:10], res['rec_scores'][:10]):
    print(f"  [{s:.2f}] {t}")
