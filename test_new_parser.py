import re

p1_text = """
C.S. 24618          no 18452/25       13207
                                       100 Rs.
Date : 22-08-2024    Serial No : 15,893   Denomination : 100
Purchased by :
M. RAGHAVENDER S/O. MUNIAH
R/O. 12-7-121/5
OLD MALKAJGIRI
SECUNDERABAD
R.R.DIST
For Whom :
M/S. SAI RAM DEVELOPERS PVT. LTD.
REP. BY ITS DIRECTOR
Mr. RAGHAVENDER M.
Sub Registrar P.O.Office Stamp Vendor S.R.O. QUTHBULLAPUR

S.A.L.E.D.E.E.D

This Deed of Sale is made and executed on this the 22nd
day of August , 2024 by :-

52,800/- Consideration

Sri M. RAGHAVENDER, S/o. Muniah, aged about 45 years, Occupation: Business, residing at H.No.12-7-121/5, Old Malkajgiri, Secunderabad - 500 047, Ranga Reddy District.
(Hereinafter called the 'VENDOR') of the First Part.

IN FAVOUR OF

M/s. SAI RAM DEVELOPERS PRIVATE LIMITED, Represented by its Director Mr. Raghavender M., aged about 42 years, Occupation: Business, having its Registered Office at Plot No. 45, 2nd Floor, Sai Arcade, Quthbullapur, Hyderabad - 500 055, Ranga Reddy District.
(Hereinafter called the 'PURCHASER') of the Second Part.
"""

# 1. Document Number
for m in re.finditer(r"\b(?:no|no\.|doc\.?\s*no\.?)?\s*([0-9]{3,6}\s*/\s*[0-9]{1,4})\b", p1_text, re.IGNORECASE):
    val = m.group(1).replace(" ", "")
    print("Found doc number candidate:", val)

# 2. Stamp Sold To
m_sold = re.search(r"PURCHASED\s+BY\s*[:\-]?\s*([A-Z][A-Za-z.\s]+?)(?=\s+(?:S/O|W/O|D/O|R/O|FOR\s+WHOM|FOR|\n|$))", p1_text, re.IGNORECASE)
if m_sold:
    raw = m_sold.group(1).strip()
    name = re.sub(r"\s+", " ", raw).title()
    print("Stamp Sold To:", name)

# 3. Execution Date
MONTH_MAP = {
    "JANUARY": "01", "FEBRUARY": "02", "MARCH": "03", "APRIL": "04",
    "MAY": "05", "JUNE": "06", "JULY": "07", "AUGUST": "08",
    "SEPTEMBER": "09", "OCTOBER": "10", "NOVEMBER": "11", "DECEMBER": "12",
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}
m_exec = re.search(
    r"(?:MADE\s+AND\s+)?EXECUTED\s+ON\s+THIS\s+THE.*?(\d{1,2})\s*(?:ST|ND|RD|TH)?.*?DAY\s+OF\s+([A-Z]+)[\s,.\-_]+(\d{4})",
    p1_text,
    re.IGNORECASE | re.DOTALL
)
if m_exec:
    d, mo_raw, y = m_exec.groups()
    mo = MONTH_MAP.get(mo_raw.upper())
    print("Execution Date:", f"{int(d):02d}-{mo}-{y}")

# 4. Vendor Extraction
m_vend = re.search(r"([^\n]+(?:\n[^\n]+){0,2})\s*\(\s*HEREINAFTER\s+CALLED\s+(?:THE\s+)?['\"]?VENDOR['\"]?\s*\)", p1_text, re.IGNORECASE)
if m_vend:
    v_raw = m_vend.group(1)
    m_comp = re.search(r"M/s\.?\s+([A-Za-z\s.]+?(?:PRIVATE\s+)?(?:LIMITED|LTD)\.?)", v_raw, re.IGNORECASE)
    if m_comp:
        print("Vendor (Company):", m_comp.group(0).strip())
    else:
        m_person = re.search(r"(?:SRI|SHRI|MR\.?|SMT\.?)\s+[A-Za-z.\s]+?(?=\s*,|\s+S/O|\s+W/O|\s+D/O|\s+AGED|\s+OCCUPATION|$)", v_raw, re.IGNORECASE)
        if m_person:
            p_name = re.sub(r"\s+", " ", m_person.group(0).strip())
            parts = p_name.split()
            norm_parts = [p.upper() if len(p) <= 2 and p.endswith('.') else p.capitalize() for p in parts]
            print("Vendor (Person):", " ".join(norm_parts))

doc1_p1 = """
M/s. SRINIDHI HOMES PRIVATE LIMITED, represented by its Director Sri P. SRINIVAS REDDY, S/o. Sri P. Narayana Reddy, aged about 40 years, Occupation: Business, R/o. Plot No. 64, Phase-I, Gunrock Enclave, Secunderabad - 500 009.
(Hereinafter called the 'VENDOR') of the First Part.

IN FAVOUR OF

SMT. B. SUVARNA W/O SRI. B. YADAIAH, aged about 35 years, Occupation: Housewife, Residing at Flat No. 203, Manasadhama Nilayam, H.No. 1-1-385/21, Gandhi Nagar, Hyderabad.
(Hereinafter called the 'PURCHASER') of the Second Part.
"""

doc2_p1 = """
Sri M. RAGHAVENDER, S/o. Muniah, aged about 45 years, Occupation: Business, residing at H.No.12-7-121/5, Old Malkajgiri, Secunderabad - 500 047, Ranga Reddy District.
(Hereinafter called the 'VENDOR') of the First Part.

IN FAVOUR OF

M/s. SAI RAM DEVELOPERS PRIVATE LIMITED, Represented by its Director Mr. Raghavender M., aged about 42 years, Occupation: Business, having its Registered Office at Plot No. 45, 2nd Floor, Sai Arcade, Quthbullapur, Hyderabad - 500 055, Ranga Reddy District.
(Hereinafter called the 'PURCHASER') of the Second Part.
"""

def parse_parties(text):
    parties = []
    
    # 1. Vendor
    m_v = re.search(r"(.+?)\s*\(\s*HEREINAFTER\s+CALLED\s+(?:THE\s+)?['\"]?VENDOR['\"]?\s*\)", text, re.IGNORECASE | re.DOTALL)
    if m_v:
        v_raw = m_v.group(1).strip()
        m_comp = re.search(r"M/s\.?\s+([A-Za-z\s.]+?(?:PRIVATE\s+)?(?:LIMITED|LTD)\.?)", v_raw, re.IGNORECASE)
        m_rep = re.search(r"REPRESENTED\s+BY\s+(?:ITS\s+DIRECTOR\s+)?((?:SRI|SHRI|MR\.?|SMT\.?)\s+[A-Za-z.\s]+?)(?=,|\s+S/O|\s+AGED|\s+OCCUPATION|$)", v_raw, re.IGNORECASE)
        if m_comp:
            c_name = m_comp.group(0).strip()
            c_name = re.sub(r"\bPRIVATE\b", "PVT.", c_name, flags=re.IGNORECASE)
            c_name = re.sub(r"\bLIMITED\b", "LTD.", c_name, flags=re.IGNORECASE)
            v_dict = {"name": c_name, "role": "Vendor"}
            if m_rep:
                v_dict["represented_by"] = re.sub(r"\s+", " ", m_rep.group(1).strip())
            parties.append(v_dict)
        else:
            m_pers = re.search(r"((?:SRI|SHRI|MR\.?|SMT\.?)\s+[A-Za-z.\s]+?)(?=,|\s+S/O|\s+W/O|\s+D/O|\s+AGED|\s+OCCUPATION|$)", v_raw, re.IGNORECASE)
            if m_pers:
                raw_p = re.sub(r"\s+", " ", m_pers.group(1).strip())
                parts = raw_p.split()
                norm_p = [p.upper() if len(p) <= 2 and p.endswith('.') else p.capitalize() for p in parts]
                parties.append({"name": " ".join(norm_p), "role": "Vendor"})

    # 2. Purchaser
    m_p = re.search(r"IN\s+FAVOUR\s+OF\s+(.+?)\s*\(\s*HEREINAFTER\s+CALLED\s+(?:THE\s+)?['\"]?PURCHASER['\"]?\s*\)", text, re.IGNORECASE | re.DOTALL)
    if m_p:
        p_raw = m_p.group(1).strip()
        m_comp = re.search(r"M/s\.?\s+([A-Za-z\s.]+?(?:PRIVATE\s+)?(?:LIMITED|LTD)\.?)", p_raw, re.IGNORECASE)
        m_rep = re.search(r"REPRESENTED\s+BY\s+(?:ITS\s+DIRECTOR\s+)?((?:SRI|SHRI|MR\.?|SMT\.?)\s+[A-Za-z.\s]+?)(?=,|\s+AGED|\s+OCCUPATION|$)", p_raw, re.IGNORECASE)
        if m_comp:
            c_name = m_comp.group(0).strip()
            p_dict = {"name": c_name, "role": "Purchaser"}
            if m_rep:
                p_dict["represented_by"] = re.sub(r"\s+", " ", m_rep.group(1).strip())
            parties.append(p_dict)
        else:
            m_pers = re.search(r"((?:SRI|SHRI|MR\.?|SMT\.?)\s+[A-Za-z.\s]+?)(?=,|\s+W/O|\s+S/O|\s+D/O|\s+AGED|\s+OCCUPATION|$)", p_raw, re.IGNORECASE)
            if m_pers:
                raw_p = re.sub(r"\s+", " ", m_pers.group(1).strip())
                parts = raw_p.split()
                norm_p = [p.upper() if len(p) <= 2 and p.endswith('.') else p.capitalize() for p in parts]
                p_name = " ".join(norm_p)
                parties.append({"name": p_name, "role": "Purchaser"})


    return parties

print("=== DOC 1 PARTIES ===")
for p in parse_parties(doc1_p1):
    print(" ", p)

print("\n=== DOC 2 (testdoc_1) PARTIES ===")
for p in parse_parties(doc2_p1):
    print(" ", p)

