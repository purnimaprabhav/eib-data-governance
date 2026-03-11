import pandas as pd
from src.pii_engine import PIIDetector
from src.esg_mapper import ESGMapper

def run_governance_pipeline(csv_path):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # 2. Initialize AI Engines
    pii_engine = PIIDetector()
    esg_engine = ESGMapper()
    
    # 3. Analyze
    report = []
    pii_flags = {}
    
    # Map ESG Themes
    esg_mappings = esg_engine.map_columns(df.columns)
    
    for col in df.columns:
        # Detect PII
        found_pii = pii_engine.scan_column(df[col])
        
        report.append({
            "Column": col,
            "PII_Found": "Yes" if found_pii else "No",
            "PII_Types": found_pii,
            "ESG_Mapping": esg_mappings.get(col, {}).get("ESRS_Code", "N/A"),
            "Confidence": esg_mappings.get(col, {}).get("Confidence", 0)
        })
        
    return pd.DataFrame(report)


result = run_governance_pipeline("data/raw/company_esg_data.csv")
print(result)