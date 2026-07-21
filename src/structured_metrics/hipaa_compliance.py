import pgeocode
import re
import pandas as pd












def detect_hipaa_identifiers(df, columns_to_scan, country = "tn"):
    pgeocode.Nominatim(country.lower())

    import re

    patterns = {
        "TUN_CIN": re.compile(r"\b\d{8}\b"),
        
        "TUN_PHONE": re.compile(r"\b(?:(?:\+216|00216)?[24597]\d{7})\b"),
        
        "TUN_POSTAL_CODE": re.compile(r"\b\d{4}\b"),
        
        "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        "URL": re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"),
        
        "TUN_MATRICULE_FISCAL": re.compile(r"\b\d{7}/[A-Z]/[A-Z]/[A-Z]/\d{3}\b"),
    }

    TUN_POSTAL_CODE = re.compile(r"\b(?!(?:19|20)\d{2}\b)\d{4}\b")


    detected_phi = {}


    for col in columns_to_scan:
        if col not in df:
            continue
        


        