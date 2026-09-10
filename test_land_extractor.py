from land_document_extractor import OCRLine, extract_land_document_from_lines


SAMPLE_LINES = [
    OCRLine(text="D.NO : 1993 / 2018", score=0.96, x_min=0, y_min=10, x_max=100, y_max=30, page_num=1, page_height=400),
    OCRLine(text="TELANGANA", score=0.99, x_min=0, y_min=35, x_max=100, y_max=55, page_num=1, page_height=400),
    OCRLine(text="Sl No. 413 Dt. 03-08-2019 Rs. 50/-", score=0.97, x_min=0, y_min=60, x_max=100, y_max=80, page_num=1, page_height=400),
    OCRLine(text="MALLURU VENU GOPAL Licensed Stamp Vendor 379230", score=0.95, x_min=0, y_min=85, x_max=100, y_max=105, page_num=1, page_height=400),
    OCRLine(text="AGREEMENT OF SALE-CUM-GENERAL POWER OF ATTORNEY", score=0.99, x_min=0, y_min=120, x_max=100, y_max=140, page_num=1, page_height=400),
    OCRLine(text="THIS AGREEMENT OF SALE-CUM GENERAL POWER OF ATTORNEY IS MADE AND EXECUTED ON THIS 05TH DAY OF AUGUST-2019", score=0.96, x_min=0, y_min=145, x_max=100, y_max=170, page_num=1, page_height=400),
    OCRLine(text="GUDURU MALAKONDAIAH, S/o. CHINNA KONDAIAH, Age.52 Years, Occup: Agriculture, R/o. H.No. 2-234/1, Main Road, Mangapet Village, Dist. Warangal, Presently Mulugu District. Aadhar No. XXXX XXXX 5888.", score=0.95, x_min=0, y_min=180, x_max=100, y_max=220, page_num=1, page_height=400),
    OCRLine(text="(HEREINAFTER CALLED THE VENDOR/PRINCIPAL)", score=0.98, x_min=0, y_min=225, x_max=100, y_max=240, page_num=1, page_height=400),
    OCRLine(text="IN FAVOUR OF", score=0.98, x_min=0, y_min=245, x_max=100, y_max=260, page_num=1, page_height=400),
    OCRLine(text="1] RANGINENI VENKATESHWAR RAO, S/o. RAMCHANDER RAO, Age: 55 Years, Occup: Business, R/o. H.No. 11-19-97, New Grain Market Road, Kashibugga, Warangal City. Aadhar No. XXXX XXXX 2146.", score=0.96, x_min=0, y_min=265, x_max=100, y_max=305, page_num=1, page_height=400),
    OCRLine(text="2] KAKKERLA SURESH, S/o. NARSAIAH, Age: 47 Years, Occup: Business, R/o. H.No. 3-13, Kommala Village, Geesugonda Mandal, Dist. Warangal. Aadhar Card No. XXXX XXXX 6636.", score=0.96, x_min=0, y_min=310, x_max=100, y_max=350, page_num=1, page_height=400),
    OCRLine(text="Contd...2/p", score=0.99, x_min=0, y_min=360, x_max=100, y_max=380, page_num=1, page_height=400),
]


def main() -> None:
    raw_text = "\n".join(line.text for line in SAMPLE_LINES)
    result = extract_land_document_from_lines(SAMPLE_LINES, raw_text, "sample_document.png")

    assert result["document_type"] == "Agreement of Sale-cum-General Power of Attorney", f"Got: {result['document_type']}"
    assert result["document_category"] == "Property Transaction Document", f"Got: {result['document_category']}"
    assert result["state"] == "Telangana", f"Got: {result['state']}"
    assert result["property"]["status"] == "CONTINUES_ON_NEXT_PAGE"
    assert result["document_features"]["multi_page_document"] is True
    assert result["document_features"]["pii_detected"] is True

    # Print extracted fields for verification
    print(f"document_type:   {result['document_type']}")
    print(f"document_number: {result['document_number']}")
    print(f"stamp_value:     {result['stamp_value']}")
    print(f"document_date:   {result['document_date']}")
    print(f"execution_date:  {result['execution_date']}")
    print(f"parties_list:    {result['parties_list']}")

    print("\nland-document extraction parser test passed")


if __name__ == "__main__":
    main()

