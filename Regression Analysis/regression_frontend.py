import streamlit as st
import pandas as pd
from regression_backend import RegressionAnalyzer, LLMAnalyzer
import os
import tempfile

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit App Config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 Regression Model Performance Analyzer", layout="wide")
st.title("📈 Regression Model Performance Analyzer")

# Initialize backends with S3 configuration
@st.cache_resource
def get_analyzers():
    # Add your S3 configuration here
    api_url = st.secrets.get("API_URL", None)  # or use environment variables
    jwt_token = st.secrets.get("JWT_TOKEN", None)
    user_id = st.secrets.get("USER_ID", None)
    tenant_id = st.secrets.get("TENANT_ID", None)
    
    return RegressionAnalyzer(api_url, jwt_token, user_id, tenant_id), LLMAnalyzer()

regression_analyzer, llm_analyzer = get_analyzers()

# Session State
if "reference_df" not in st.session_state:
    st.session_state.reference_df = None
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "s3_data_loaded" not in st.session_state:
    st.session_state.s3_data_loaded = False
if "uploaded_model" not in st.session_state:
    st.session_state.uploaded_model = None

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
                        ref_df, cur_df, model = regression_analyzer.load_data_from_s3(split_pct=70)
                        
                        if ref_df is not None and cur_df is not None:
                            st.session_state.reference_df = ref_df
                            st.session_state.current_df = cur_df
                            st.session_state.model = model
                            st.session_state.s3_data_loaded = True
                            st.success("✅ S3 data loaded successfully!")
                            
                            if model is not None:
                                st.success("✅ Model loaded from S3!")
                            else:
                                st.warning("⚠️ No model found in S3. Please upload a model file.")
                        else:
                            st.error("❌ Failed to load data from S3")
                            st.session_state.s3_data_loaded = False
                    except Exception as e:
                        st.error(f"❌ S3 loading error: {str(e)}")
                        st.session_state.s3_data_loaded = False
        
        with col2:
            split_pct = st.slider("Reference split (%)", 10, 90, 70, 5, key="s3_split")
            if st.session_state.s3_data_loaded and st.button("🔄 Re-split Data", key="resplit_s3"):
                with st.spinner("Re-splitting data..."):
                    try:
                        ref_df, cur_df, model = regression_analyzer.load_data_from_s3(split_pct=split_pct)
                        if ref_df is not None and cur_df is not None:
                            st.session_state.reference_df = ref_df
                            st.session_state.current_df = cur_df
                            if model is not None:
                                st.session_state.model = model
                            st.success(f"✅ Data re-split: {split_pct}% reference, {100-split_pct}% current")
                    except Exception as e:
                        st.error(f"❌ Re-split error: {str(e)}")
        
        # Show S3 data info if loaded
        if st.session_state.s3_data_loaded and st.session_state.reference_df is not None:
            st.markdown("#### 📊 S3 Data Info")
            st.markdown(f"**Reference Rows:** {len(st.session_state.reference_df)}")
            st.markdown(f"**Current Rows:** {len(st.session_state.current_df)}")
            st.markdown(f"**Columns:** {len(st.session_state.reference_df.columns)}")
            st.markdown(f"**Model Status:** {'✅ Loaded' if st.session_state.model else '❌ Not Found'}")
            
            # Column preprocessing options
            st.markdown("#### 🔧 Data Preprocessing")
            all_columns = list(st.session_state.reference_df.columns)
            drop_columns = st.multiselect(
                "Columns to drop (optional):",
                all_columns,
                key="drop_cols_s3"
            )
            
            if st.button("Apply Preprocessing", key="preprocess_s3"):
                if drop_columns:
                    st.session_state.reference_df = regression_analyzer.preprocess(
                        st.session_state.reference_df, drop_cols=drop_columns
                    )
                    st.session_state.current_df = regression_analyzer.preprocess(
                        st.session_state.current_df, drop_cols=drop_columns
                    )
                    st.success(f"✅ Dropped {len(drop_columns)} columns")
            
            # Target column selection
            st.markdown("#### 🎯 Target Column Selection")
            available_columns = list(st.session_state.reference_df.columns)
            st.session_state.target_column = st.selectbox(
                "Select target column:",
                available_columns,
                key="target_col_s3"
            )
    
    # ─────────────────────────────
    # File Upload Options
    # ─────────────────────────────
    elif data_source == "📁 Upload Files":
        st.markdown("### Upload Data Files")
        
        # Model file upload
        st.markdown("#### 🤖 Upload Model File")
        model_file = st.file_uploader(
            "Upload trained model (.pkl or .joblib):",
            type=["pkl", "joblib"],
            key="model_upload"
        )
        
        if model_file:
            st.session_state.uploaded_model = model_file.read()
            st.success("✅ Model uploaded successfully!")
        
        with st.expander("Option 1: Separate Reference and Current files"):
            train_file = st.file_uploader("📂 Upload Reference Dataset", type=["csv"], key="data_train")
            test_file = st.file_uploader("📂 Upload Current Dataset", type=["csv"], key="data_test")

            if train_file and test_file:
                ref_df = pd.read_csv(train_file)
                cur_df = pd.read_csv(test_file)

                if list(ref_df.columns) != list(cur_df.columns):
                    st.error("🚫 Reference and Current datasets must have the same columns.")
                    st.stop()

                st.session_state.reference_df, st.session_state.current_df = regression_analyzer.load_data(ref_df, cur_df)
                st.session_state.s3_data_loaded = False

                st.markdown(f"**Reference Rows:** {len(ref_df)}   |   **Current Rows:** {len(cur_df)}")
                
                # Column preprocessing for uploaded files
                st.markdown("#### 🔧 Data Preprocessing")
                all_columns = list(ref_df.columns)
                drop_columns = st.multiselect(
                    "Columns to drop (optional):",
                    all_columns,
                    key="drop_cols_upload1"
                )
                
                if st.button("Apply Preprocessing", key="preprocess_upload1"):
                    if drop_columns:
                        st.session_state.reference_df = regression_analyzer.preprocess(
                            st.session_state.reference_df, drop_cols=drop_columns
                        )
                        st.session_state.current_df = regression_analyzer.preprocess(
                            st.session_state.current_df, drop_cols=drop_columns
                        )
                        st.success(f"✅ Dropped {len(drop_columns)} columns")
                
                # Target column selection
                st.markdown("#### 🎯 Target Column Selection")
                available_columns = list(st.session_state.reference_df.columns if st.session_state.reference_df is not None else ref_df.columns)
                st.session_state.target_column = st.selectbox(
                    "Select target column:",
                    available_columns,
                    key="target_col_upload1"
                )

        with st.expander("Option 2: Combined CSV file"):
            data_file = st.file_uploader("📂 Upload combined CSV", type=["csv"], key="data_combined")
            if data_file:
                df = pd.read_csv(data_file)
                st.session_state.s3_data_loaded = False
                st.markdown(f"**Rows:** {len(df)}   |   **Columns:** {len(df.columns)}")

                split_pct = st.slider("Reference split (%)", 10, 90, 70, 5, key="file_split")
                st.session_state.reference_df, st.session_state.current_df = regression_analyzer.load_data(df, split_pct=split_pct)

                # Column preprocessing for combined file
                st.markdown("#### 🔧 Data Preprocessing")
                all_columns = list(df.columns)
                drop_columns = st.multiselect(
                    "Columns to drop (optional):",
                    all_columns,
                    key="drop_cols_upload2"
                )
                
                if st.button("Apply Preprocessing", key="preprocess_upload2"):
                    if drop_columns:
                        st.session_state.reference_df = regression_analyzer.preprocess(
                            st.session_state.reference_df, drop_cols=drop_columns
                        )
                        st.session_state.current_df = regression_analyzer.preprocess(
                            st.session_state.current_df, drop_cols=drop_columns
                        )
                        st.success(f"✅ Dropped {len(drop_columns)} columns")
                
                # Target column selection
                st.markdown("#### 🎯 Target Column Selection")
                available_columns = list(st.session_state.reference_df.columns if st.session_state.reference_df is not None else df.columns)
                st.session_state.target_column = st.selectbox(
                    "Select target column:",
                    available_columns,
                    key="target_col_upload2"
                )

# ────────────────────────────────────────────────────────────────────────────────
# Main Panel: Regression Analysis Report
# ────────────────────────────────────────────────────────────────────────────────
if (st.session_state.reference_df is not None and 
    st.session_state.current_df is not None and 
    st.session_state.target_column is not None):
    
    ref_df = st.session_state.reference_df
    cur_df = st.session_state.current_df
    target_col = st.session_state.target_column
    
    # Check if we have a model (from S3 or uploaded)
    model_available = st.session_state.model is not None or st.session_state.uploaded_model is not None
    
    # Show data source info
    data_source_info = "🌐 S3 Data" if st.session_state.s3_data_loaded else "📁 Uploaded Files"
    st.info(f"📊 Data Source: {data_source_info} | 🎯 Target: **{target_col}** | 🤖 Model: {'✅ Available' if model_available else '❌ Missing'}")

    # Data preview
    left, right = st.columns(2)
    with left:
        st.subheader("📋 Reference Dataset Preview")
        st.dataframe(ref_df.head(10))
        st.caption(f"Shape: {ref_df.shape}")
    with right:
        st.subheader("📋 Current Dataset Preview")
        st.dataframe(cur_df.head(10))
        st.caption(f"Shape: {cur_df.shape}")

    # Check if target column exists in both datasets
    if target_col not in ref_df.columns or target_col not in cur_df.columns:
        st.error(f"❌ Target column '{target_col}' not found in both datasets. Please select a valid target column.")
        st.stop()

    # Generate regression analysis button
    st.markdown("---")
    
    if model_available:
        if st.button("🚀 Generate Regression Performance Analysis", key="generate_analysis"):
            with st.spinner("Generating regression performance analysis..."):
                try:
                    # Determine which model to use
                    model_to_use = st.session_state.uploaded_model if st.session_state.uploaded_model else st.session_state.model
                    
                    # Generate regression report
                    html, regression_dict, json_path, upload_result = regression_analyzer.Regression_report(
                        target_column_name=target_col,
                        Model_File=model_to_use
                    )
                    
                    # Show save status
                    if json_path:
                        st.info(f"📁 JSON report saved locally: {os.path.basename(json_path)}")
                    else:
                        st.warning("⚠️ JSON report could not be saved")
                    
                    # Display the HTML report
                    st.markdown("## 📈 Regression Performance Report")
                    st.components.v1.html(html, height=900, scrolling=True)
                    
                    # Generate AI-powered summary
                    if regression_dict:
                        try:
                            with st.spinner("Generating AI-powered analysis..."):
                                summary_text = regression_analyzer.get_regression_summary(regression_dict)
                                
                                if summary_text and summary_text.strip():
                                    llm_response = llm_analyzer.analyze_Regrresion_Report(
                                        ref_df=ref_df,
                                        cur_df=cur_df,
                                        summary_text=summary_text
                                    )
                                    
                                    st.markdown("---")
                                    st.markdown("## 🤖 AI-Powered Regression Analysis")
                                    st.markdown(llm_response)
                                else:
                                    st.warning("⚠️ No regression metrics found for AI analysis")
                        
                        except Exception as e:
                            st.warning(f"⚠️ LLM analysis failed: {str(e)}")
                            st.markdown("### 📊 Raw Regression Summary")
                            st.text(summary_text if 'summary_text' in locals() else "No summary available")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.markdown("**Possible issues:**")
                    st.markdown("- Model file format incompatible")
                    st.markdown("- Feature mismatch between model and data")
                    st.markdown("- Target column contains invalid values")
    else:
        st.warning("⚠️ Please upload a model file (.pkl or .joblib) to generate regression analysis.")
        st.markdown("**Model Requirements:**")
        st.markdown("- Trained scikit-learn model")
        st.markdown("- Saved as .pkl or .joblib file")
        st.markdown("- Compatible with your dataset features")

elif st.session_state.reference_df is not None and st.session_state.current_df is not None:
    st.info("🛈 Please select a target column in the sidebar to begin regression analysis.")
elif st.session_state.reference_df is not None or st.session_state.current_df is not None:
    st.info("🛈 Please ensure both reference and current datasets are loaded.")
else:
    st.info("🛈 Please choose a data source and load your data to begin regression analysis.")

# ────────────────────────────────────────────────────────────────────────────────
# Footer with helpful information
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    ### 📋 Step-by-Step Guide:
    
    1. **Choose Data Source**: Select either S3 or file upload
    2. **Load Data**: Upload or load your reference and current datasets
    3. **Upload Model**: Provide a trained regression model (.pkl or .joblib)
    4. **Select Target**: Choose the target column for regression analysis
    5. **Preprocess** (Optional): Drop unnecessary columns
    6. **Generate Analysis**: Click the analysis button to get detailed performance metrics
    
    ### 🎯 What This Tool Analyzes:
    - **Model Performance Metrics**: MAE, MSE, RMSE, R², MAPE
    - **Performance Comparison**: Reference vs Current dataset performance
    - **Statistical Tests**: Detect significant performance changes
    - **AI Insights**: Automated interpretation of results
    
    ### 📁 Supported File Formats:
    - **Data**: CSV files
    - **Models**: .pkl, .joblib (scikit-learn models)
    
    ### 🔧 Requirements:
    - Reference and current datasets must have identical column structure
    - Model must be compatible with dataset features
    - Target column must exist in both datasets
    """)