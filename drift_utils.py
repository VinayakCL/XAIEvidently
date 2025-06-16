import pandas as pd
import streamlit as st
from evidently import Report
from evidently.presets import DataDriftPreset
import boto3

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Map yes/no to 1/0 and one-hot encode categorical columns."""
    yes_no_cols = [col for col in df.columns if set(df[col].dropna().unique()) == {'yes', 'no'}]
    for col in yes_no_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    return pd.get_dummies(df, drop_first=True)


def load_and_run_drift_report(ref_path, cur_path=None):
    """
    Handles both single and dual dataset modes.
    Returns: (ref_df, cur_df, report_html)
    """
    # Load reference dataset
    try:
        df_ref = pd.read_csv(ref_path)
    except Exception as e:
        raise FileNotFoundError(f"❌ Could not load reference dataset:\n{e}")

    split_mode = cur_path is None

    if split_mode:
        st.info("🧪 Only reference file uploaded. Enter test size and column selection below to split.")
        split_ratio = st.slider("Split ratio for test set", 0.05, 0.5, 0.2, step=0.05)

        st.write("### Available Columns")
        all_columns = df_ref.columns.tolist()
        selected = st.multiselect("Select columns to check drift on", all_columns, default=all_columns)

        if not selected:
            st.error("Please select at least one column.")
            st.stop()

        df_subset = df_ref[selected]
        df_subset = preprocess(df_subset)

        # Sequential split (NOT random)
        n = len(df_subset)
        split_index = int(n * (1 - split_ratio))
        df_ref = df_subset.iloc[:split_index].reset_index(drop=True)
        df_cur = df_subset.iloc[split_index:].reset_index(drop=True)

    else:
        try:
            df_cur = pd.read_csv(cur_path)
        except Exception as e:
            raise FileNotFoundError(f"❌ Could not load current dataset:\n{e}")

        # Preprocess both
        df_ref = preprocess(df_ref)
        df_cur = preprocess(df_cur)

        # Align columns
        common = list(set(df_ref.columns).intersection(df_cur.columns))
        df_ref = df_ref[common]
        df_cur = df_cur[common]

    # === RUN EVIDENTLY DRIFT REPORT ===
    report = Report([DataDriftPreset(method="psi", threshold=0.1)], include_tests=True)
    report.run(reference_data=df_ref, current_data=df_cur)

    return df_ref, df_cur, report.get_html()


