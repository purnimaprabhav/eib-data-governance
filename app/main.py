import streamlit as st
import pandas as pd
import sys
import os

# Ensure the app can find the /src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge_base import GovernanceVault
from src.pii_engine import PIIDetector
from src.esg_mapper import ESGMapper

# --- Page Config ---
st.set_page_config(page_title="EIB Governance Suite", layout="wide")

st.title("EIB Data Governance Suite")
st.markdown("> *'Between data and conscience lies governance.'*")

# --- Load Engines (Cached for Performance) ---
@st.cache_resource
def load_engines():
    """Loads the PII Detector, ESG Mapper, and the ChromaDB Vault."""
    return PIIDetector(), ESGMapper(), GovernanceVault()

pii_detector, esg_mapper, vault = load_engines()

# --- Sidebar: Upload ---
st.sidebar.header("Upload Project Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["📊 Raw Data", "⚖️ Governance Report", "🤖 AI Copilot"])

    # --- TAB 1: RAW DATA ---
    with tab1:
        st.subheader("Project Dataset Preview")
        st.dataframe(df.head(10))

    # --- TAB 2: GOVERNANCE REPORT ---
    with tab2:
        st.subheader("Automated Governance Analysis")
        
        with st.spinner("AI is analyzing compliance..."):
            report_data = []
            # Use the ESG Mapper to find relevant standards
            esg_map = esg_mapper.map_columns(df.columns)
            
            for col in df.columns:
                # Use the PII Engine to scan the actual content
                pii_entities = pii_detector.scan_column(df[col])
                report_data.append({
                    "Field Name": col,
                    "Privacy Risk": "⚠️ PII Detected" if pii_entities else " Clear",
                    "Detected Entities": ", ".join(pii_entities) if pii_entities else "-",
                    "ESG Alignment": esg_map.get(col, {}).get("ESRS_Code", "None"),
                    "Confidence": esg_map.get(col, {}).get("Confidence", 0.0)
                })
            
            report_df = pd.DataFrame(report_data)
            
            # Display high-level metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Fields", len(df.columns))
            c2.metric("PII Warnings", len(report_df[report_df["Privacy Risk"] != " Clear"]))
            c3.metric("ESG Mapped Fields", len(report_df[report_df["ESG Alignment"] != "None"]))

            st.table(report_df)

    # --- TAB 3: AI COPILOT (RAG) ---
    with tab3:
        st.header("⚖️ AI Governance Copilot")
        st.info("The Copilot is searching the Governance Vault for ESRS or GDPR requirements.")

        # User types their question here
        query = st.text_input("Ask a question (e.g., 'What are the water usage rules?' or 'How to handle names?')")

        if query:
            with st.spinner("Analyzing regulations..."):
                # Search the ChromaDB Vault we created
                relevant_docs = vault.query(query)
                
                st.markdown("### Found in Regulatory Database:")
                if relevant_docs:
                    for doc in relevant_docs:
                        st.success(doc)
                else:
                    st.warning("No specific regulation found for that query.")
                
                st.caption("Matches found using Semantic Vector Search (Local Knowledge Base).")

else:
    st.info("Please upload the Kaggle ESG dataset in the sidebar to begin.")