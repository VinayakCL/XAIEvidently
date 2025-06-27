import streamlit as st
import pandas as pd
from drift_backend1 import DriftAnalyzer, LLMAnalyzer
import os

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit App Config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📊 Data Drift Column Explorer", layout="wide")
st.title("📊 Data Drift Column Explorer")

# Initialize backends with S3 configuration
@st.cache_resource
def get_analyzers():
    # Add your S3 configuration here
    api_url = st.secrets.get("API_URL", None)  # or use environment variables
    jwt_token = st.secrets.get("JWT_TOKEN", None)
    user_id = st.secrets.get("USER_ID", None)
    tenant_id = st.secrets.get("TENANT_ID", None)
    
    return DriftAnalyzer(api_url, jwt_token, user_id, tenant_id), LLMAnalyzer()

drift_analyzer, llm_analyzer = get_analyzers()

# Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "reference_df" not in st.session_state:
    st.session_state.reference_df = None
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "selected_col" not in st.session_state:
    st.session_state.selected_col = None
if "s3_data_loaded" not in st.session_state:
    st.session_state.s3_data_loaded = False

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar: Data Source Selection
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧰 Setup")
    st.markdown("### 1. Choose Data Source")
    
    data_source = st.radio(
        "Select data source:",
        ["🌐 Load from S3", "📁 Upload Files"],
        key="data_source_selector"
    )
    
    # ─────────────────────────────
    # S3 Data Loading Option
    # ─────────────────────────────
    if data_source == "🌐 Load from S3":
        st.markdown("### Load Data from S3")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load S3 Data", key="load_s3"):
                with st.spinner("Loading data from S3..."):
                    try:
                        ref_df, cur_df = drift_analyzer.load_data_from_s3(split_pct=50)
                        
                        if ref_df is not None and cur_df is not None:
                            st.session_state.df = pd.concat([ref_df, cur_df], ignore_index=True)
                            st.session_state.reference_df = ref_df
                            st.session_state.current_df = cur_df
                            st.session_state.s3_data_loaded = True
                            st.success("✅ S3 data loaded successfully!")
                        else:
                            st.error("❌ Failed to load data from S3")
                            st.session_state.s3_data_loaded = False
                    except Exception as e:
                        st.error(f"❌ S3 loading error: {str(e)}")
                        st.session_state.s3_data_loaded = False
        
        with col2:
            split_pct = st.slider("Reference split (%)", 10, 90, 50, 5, key="s3_split")
            if st.session_state.s3_data_loaded and st.button("🔄 Re-split Data", key="resplit_s3"):
                with st.spinner("Re-splitting data..."):
                    try:
                        ref_df, cur_df = drift_analyzer.load_data_from_s3(split_pct=split_pct)
                        if ref_df is not None and cur_df is not None:
                            st.session_state.reference_df = ref_df
                            st.session_state.current_df = cur_df
                            st.success(f"✅ Data re-split: {split_pct}% reference, {100-split_pct}% current")
                    except Exception as e:
                        st.error(f"❌ Re-split error: {str(e)}")
        
        # Show S3 data info if loaded
        if st.session_state.s3_data_loaded and st.session_state.df is not None:
            st.markdown("#### 📊 S3 Data Info")
            st.markdown(f"**Total Rows:** {len(st.session_state.df)}")
            st.markdown(f"**Columns:** {len(st.session_state.df.columns)}")
            st.markdown(f"**Reference:** {len(st.session_state.reference_df)} rows")
            st.markdown(f"**Current:** {len(st.session_state.current_df)} rows")
            
            st.markdown("#### ② Pick a column to inspect")
            options = ["🔄 Full Drift Summary (All Columns)"] + list(st.session_state.df.columns)
            st.session_state.selected_col = st.selectbox("Choose column", options, key="col_sel_s3")
    
    # ─────────────────────────────
    # File Upload Options
    # ─────────────────────────────
    elif data_source == "📁 Upload Files":
        st.markdown("### Upload Data Files")
        
        with st.expander("Option 1: Separate Reference and Current files"):
            train_file = st.file_uploader("📂 Upload Train (Reference)", type=["csv"], key="data_train")
            test_file = st.file_uploader("📂 Upload Test (Current)", type=["csv"], key="data_test")

            if train_file and test_file:
                ref_df = pd.read_csv(train_file)
                cur_df = pd.read_csv(test_file)

                if list(ref_df.columns) != list(cur_df.columns):
                    st.error("🚫 Reference and Current datasets must have the same columns.")
                    st.stop()

                st.session_state.df = pd.concat([ref_df, cur_df], ignore_index=True)
                st.session_state.reference_df, st.session_state.current_df = drift_analyzer.load_data(ref_df, cur_df)
                st.session_state.s3_data_loaded = False

                st.markdown(f"**Train Rows:** {len(ref_df)}   |   **Test Rows:** {len(cur_df)}")
                st.header("② Pick a column to inspect")
                options = ["🔄 Full Drift Summary (All Columns)"] + list(ref_df.columns)
                st.session_state.selected_col = st.selectbox("Choose column", options, key="col_sel_split")

        with st.expander("Option 2: Combined CSV file"):
            data_file = st.file_uploader("📂 Upload combined CSV", type=["csv"], key="data_combined")
            if data_file:
                df = pd.read_csv(data_file)
                st.session_state.df = df
                st.session_state.s3_data_loaded = False
                st.markdown(f"**Rows:** {len(df)}   |   **Columns:** {len(df.columns)}")

                split_pct = st.slider("Reference split (%)", 10, 90, 50, 5, key="file_split")
                st.session_state.reference_df, st.session_state.current_df = drift_analyzer.load_data(df, split_pct=split_pct)

                st.header("② Pick a column to inspect")
                options = ["🔄 Full Drift Summary (All Columns)"] + list(df.columns)
                st.session_state.selected_col = st.selectbox("Choose column", options, key="col_sel_combined")

# ────────────────────────────────────────────────────────────────────────────────
# Main Panel: Drift Report
# ────────────────────────────────────────────────────────────────────────────────
if st.session_state.df is not None and st.session_state.selected_col is not None:
    ref_df = st.session_state.reference_df
    cur_df = st.session_state.current_df
    col = st.session_state.selected_col

    # Show data source info
    data_source_info = "🌐 S3 Data" if st.session_state.s3_data_loaded else "📁 Uploaded Files"
    st.info(f"📊 Data Source: {data_source_info}")

    left, right = st.columns(2)
    with left:
        st.subheader("Reference preview")
        st.dataframe(ref_df.head(50))
    with right:
        st.subheader("Current preview")
        st.dataframe(cur_df.head(50))

    # ─────────────────────────────
    # Full Dataset Drift Summary
    # ─────────────────────────────
    if col == "🔄 Full Drift Summary (All Columns)":
        st.markdown("## 📄 Full Drift Summary (All Columns)")

        with st.spinner("Generating Evidently full drift report…"):
            html, drift_dict, json_path, upload_result = drift_analyzer.generate_full_drift_report() 
            
        # Show save status only (no upload since upload_result is always None)
        if json_path:
            st.info(f"📁 JSON report saved locally: {os.path.basename(json_path)}")
        else:
            st.warning("⚠️ JSON report could not be saved")

        st.components.v1.html(html, height=900, scrolling=True)

        if drift_dict:
            try:
                summary_text = drift_analyzer.get_drift_summary(drift_dict)
                with st.spinner("Generating AI analysis…"):
                    llm_response = llm_analyzer.analyze_full_drift(ref_df, cur_df, summary_text)
                
                st.markdown("### 🤖 AI-Powered Drift Summary")
                st.markdown(llm_response)
            except Exception as e:
                st.warning(f"LLM summary failed: {e}")

    # ─────────────────────────────
    # Single Column Drift Report
    # ─────────────────────────────
    else:
        st.markdown(f"## Drift report for **{col}**")

        with st.spinner("Generating Evidently report…"):
            html, drift_dict, json_path, upload_result = drift_analyzer.generate_column_drift_report(col)
        
        # Show save status only (no upload since upload_result is always None)
        if json_path:
            st.info(f"📁 JSON report saved locally: {os.path.basename(json_path)}")
        else:
            st.warning("⚠️ JSON report could not be saved")

        st.components.v1.html(html, height=850, scrolling=True)

        # Extract and summarize column-specific drift
        if drift_dict:
            try:
                summary = drift_analyzer.get_column_drift_summary(drift_dict, col)
                with st.spinner("Generating AI insight…"):
                    llm_response = llm_analyzer.analyze_column_drift(summary)

                st.markdown(f"### 🤖 AI-Powered {col} Drift Summary")
                st.markdown(llm_response)
            except Exception as e:
                st.warning(f"LLM summary failed: {e}")

elif st.session_state.df is not None:
    st.info("🛈 Pick a column in the sidebar to see its drift report.")
else:
    st.info("🛈 Please choose a data source and load your data to begin drift analysis.")