import os
# Fix modelscope dataset endpoint bug for newer python versions
os.environ.setdefault('HUB_DATASET_ENDPOINT', 'https://modelscope.cn/api/v1/datasets')
os.environ.setdefault('FLAGS_use_mkldnn', '0')
os.environ.setdefault('PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT', '0')

print("Attempting to import PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    print("PaddleOCR imported successfully!")
    
    print("Attempting to initialize PaddleOCR engine...")
    # Keep the smoke test aligned with the project extractor settings.
    ocr = PaddleOCR(
        lang='en',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print("PaddleOCR engine initialized successfully!")
except Exception as e:
    import traceback
    print("Error during PaddleOCR import/initialization:")
    traceback.print_exc()
