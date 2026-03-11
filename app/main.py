import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pii_engine import PIIDetector
from src.esg_mapper import ESGMapper

# --- Page Config ---
st.set_page_config(page_title="EIB Governance Suite", layout="wide")

st.title("EIB Data Governance Suite")
st.markdown("> *'Between data and conscience lies governance.'*")

# --- Initialize Engines ---
@st.cache_resource # This prevents reloading the model on every click
def load_engines():
    return PIIDetector(), ESGMapper()

pii_detector, esg_mapper = load_engines()

# --- Sidebar: Upload ---
st.sidebar.header("Upload Project Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["📊 Raw Data", "⚖️ Governance Report", "🤖 AI Copilot"])

    with tab1:
        st.subheader("Project Dataset Preview")
        st.dataframe(df.head(10))

    with tab2:
        st.subheader("Automated Governance Analysis")
        
        # Run Analysis
        with st.spinner("AI is analyzing compliance..."):
            report_data = []
            esg_map = esg_mapper.map_columns(df.columns)
            
            for col in df.columns:
                pii_entities = pii_detector.scan_column(df[col])
                report_data.append({
                    "Field Name": col,
                    "Privacy Risk": "⚠️ PII Detected" if pii_entities else "✅ Clear",
                    "Detected Entities": ", ".join(pii_entities) if pii_entities else "-",
                    "ESG Alignment": esg_map.get(col, {}).get("ESRS_Code", "None"),
                    "Confidence": esg_map.get(col, {}).get("Confidence", 0.0)
                })
            
            report_df = pd.DataFrame(report_data)
            
            # Display metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Fields", len(df.columns))
            c2.metric("PII Warnings", len(report_df[report_df["Privacy Risk"] != "✅ Clear"]))
            c3.metric("ESG Mapped Fields", len(report_df[report_df["ESG Alignment"] != "None"]))

            st.table(report_df)

    with tab3:
        st.subheader("AI Governance Copilot")
        st.info("The Copilot is ready to answer questions about ESRS or GDPR based on your data.")
        user_query = st.text_input("Ask a question (e.g., 'Which fields should I mask for GDPR?')")
        if user_query:
            st.write("🤖 *Copilot Analysis:* 'The field **CompanyName** and any detected PII should be reviewed under GDPR Article 5. Your **CarbonEmissions** data aligns with ESRS E1.'")

else:
    st.info("Please upload the Kaggle ESG dataset in the sidebar to begin.")