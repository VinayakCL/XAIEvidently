import streamlit as st
import pandas as pd
import os
import json
import boto3
import tempfile
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently import Dataset

# ────────────────────────────────────────────────────────────────────────────────
# Claude 3.7 Sonnet via AWS Bedrock
# ────────────────────────────────────────────────────────────────────────────────
def invoke_claude_3_7_sonnet(prompt):
    try:
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')

        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 130000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            body=json.dumps(request_body)
        )

        response_body = json.loads(response['body'].read().decode("utf-8"))
        return response_body['content'][0]['text']
    except Exception as e:
        return f"Claude invocation failed: {e}"

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit App Config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📊 Data Drift Column Explorer", layout="wide")
st.title("📊 Data Drift Column Explorer")

# Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "reference_df" not in st.session_state:
    st.session_state.reference_df = None
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "selected_col" not in st.session_state:
    st.session_state.selected_col = None

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar: Upload Mode
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧰 Setup")
    st.markdown("### 1. Upload Data")

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
            st.session_state.reference_df = ref_df
            st.session_state.current_df = cur_df

            st.markdown(f"**Train Rows:** {len(ref_df)}   |   **Test Rows:** {len(cur_df)}")
            st.header("② Pick a column to inspect")
            options = ["🔄 Full Drift Summary (All Columns)"] + list(ref_df.columns)
            st.session_state.selected_col = st.selectbox("Choose column", options, key="col_sel_split")

    with st.expander("Option 2: Combined CSV file"):
        data_file = st.file_uploader("📂 Upload combined CSV", type=["csv"], key="data_combined")
        if data_file:
            df = pd.read_csv(data_file)
            st.session_state.df = df
            st.markdown(f"**Rows:** {len(df)}   |   **Columns:** {len(df.columns)}")

            split_pct = st.slider("Reference split (%)", 10, 90, 50, 5)
            split_idx = int(len(df) * split_pct / 100)
            st.session_state.reference_df = df.iloc[:split_idx]
            st.session_state.current_df = df.iloc[split_idx:]

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

    left, right = st.columns(2)
    with left:
        st.subheader("Reference preview")
        st.dataframe(ref_df.head())
    with right:
        st.subheader("Current preview")
        st.dataframe(cur_df.head())

    # ─────────────────────────────
    # Full Dataset Drift Summary
    # ─────────────────────────────
    if col == "🔄 Full Drift Summary (All Columns)":
        st.markdown("## 📄 Full Drift Summary (All Columns)")

        with st.spinner("Generating Evidently full drift report…"):
            ds_ref = Dataset.from_pandas(ref_df)
            ds_cur = Dataset.from_pandas(cur_df)
            report = Report(metrics=[DataDriftPreset()])
            result = report.run(reference_data=ds_ref, current_data=ds_cur)

            # Save HTML
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            result.save_html(tmp.name)
            html = tmp.read().decode("utf-8")
            tmp.close()
            os.unlink(tmp.name)

            # JSON for LLM
            try:
                drift_json = result.json()
                drift_save = result.save_json("file1.json")
                drift_dict = json.loads(drift_json)
            except Exception as e:
                st.warning(f"⚠️ Could not parse drift JSON: {e}")
                drift_dict = {}

        st.components.v1.html(html, height=900, scrolling=True)

        if drift_dict:
            try:
                summary_lines = []
                if 'metrics' in drift_dict:
                    for metric in drift_dict['metrics']:
                        if metric.get('metric') == 'DataDriftTable':
                            result = metric.get('result', {})
                            n_drifted = result.get('number_of_drifted_columns', 0)
                            total = result.get('number_of_columns', 0)
                            share = result.get('share_of_drifted_columns', 0.0)
                            summary_lines.append(f"- Drifted columns: {n_drifted} out of {total} ({share:.0%})")
                            for feature, info in result.get('drift_by_columns', {}).items():
                                status = info.get('drift_detected', False)
                                stat = info.get('stattest_name', 'Unknown')
                                summary_lines.append(f"  - {feature}: Drift={'Yes' if status else 'No'} | Test={stat}")
                summary_text = "\n".join(summary_lines)

                prompt = f"""
You are a senior data scientist and business consultant analyzing data drift for a dataset.
 
The organization has been monitoring their data over time to detect potential changes in data patterns, quality, or distributions that could impact their machine learning models and business decisions.
 
DATASET INFORMATION:
- Reference dataset shape: {ref_df.shape}
- Current dataset shape: {cur_df.shape}
- Number of features analyzed: {len(ref_df.columns)}
- Features: {', '.join(ref_df.columns[:10])}{'...' if len(ref_df.columns) > 10 else ''}
 
DRIFT ANALYSIS RESULTS:
{summary_text}
 
ANALYSIS METHOD:
- Framework: Evidently AI
- Method: Population Stability Index (PSI)
- Threshold: 0.1 (standard threshold for drift detection)
 
Please provide a comprehensive business analysis covering:
 
1. **Executive Summary** - Key findings in 2-3 sentences
2. **Critical Drift Issues** - Which features show significant drift and severity
3. **Business Impact** - Potential consequences for models and operations
4. **Root Cause Analysis** - Possible reasons for observed drift
5. **Recommendations** - Specific actions to address drift issues
6. **Risk Assessment** - Categorize risk levels and urgency
7. **Next Steps** - Immediate and long-term actions needed
 
Format your response in clear markdown with proper headings and bullet points.
"""
                llm_response = invoke_claude_3_7_sonnet(prompt)
                st.markdown("### 🤖 AI-Powered Drift Summary")
                st.markdown(llm_response)
            except Exception as e:
                st.warning(f"LLM summary failed: {e}")

    # ─────────────────────────────
    # Single Column Drift Report
    # ─────────────────────────────
    else:
        st.markdown(f"## Drift report for **{col}**")

        with st.spinner("Generating Evidently report …"):
            ref_col_df = ref_df[[col]].copy()
            cur_col_df = cur_df[[col]].copy()

            ds_ref = Dataset.from_pandas(ref_col_df)
            ds_cur = Dataset.from_pandas(cur_col_df)

            report = Report(metrics=[DataDriftPreset()])
            result = report.run(reference_data=ds_ref, current_data=ds_cur)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            result.save_html(tmp.name)
            html = tmp.read().decode("utf-8")
            tmp.close()
            os.unlink(tmp.name)

        st.components.v1.html(html, height=850, scrolling=True)

        summary_prompt = f"""
You are a helpful assistant. Analyze the data drift for column: {col}.
Provide a short summary for the differences in distributions between the reference and current dataset (if any).
Use statistical reasoning and focus on insights.
"""
        try:
            llm_response = invoke_claude_3_7_sonnet(summary_prompt)
            st.subheader("🧠 LLM Insight (Claude)")
            st.success(llm_response)
        except Exception as e:
            st.warning(f"LLM summary failed: {e}")

elif st.session_state.df is not None:
    st.info("🛈 Pick a column in the sidebar to see its drift report.")
