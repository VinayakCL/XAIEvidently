import pandas as pd
import json
import chardet
import boto3
import tempfile
import os
import requests
import re
import mimetypes
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently import Dataset
import warnings
import requests
from io import BytesIO
 
from dotenv import load_dotenv
# loading variables from .env file
load_dotenv(dotenv_path="services/data_drift/.env")
 
warnings.filterwarnings("ignore", category=FutureWarning)
def load_flexible_csv(file):
    raw = file.read()
    encoding = chardet.detect(raw)["encoding"]
    file.seek(0)
 
    # Try comma first
    try:
        df = pd.read_csv(file, encoding=encoding)
        if df.shape[1] == 1:
            raise ValueError("Likely wrong delimiter, trying fallback...")
        return df
    except Exception:
        file.seek(0)
        try:
            df = pd.read_csv(file, sep=";", encoding=encoding)
            if df.shape[1] == 1:
                raise ValueError("Still not right — file might be malformed.")
            return df
        except Exception as e:
            raise RuntimeError(f"❌ Could not parse the CSV. Reason: {e}")
 
def s3_bucket_data():
    # Step 1: Call your FastAPI endpoint to get file URLs
    api_url = "http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com:8000/api/files_download/"
    response = requests.get(api_url)
 
    if response.status_code == 200:
        print("Successfully connected to the API.")
       
        try:
            json_data = response.json()
            files = json_data.get("files", [])
           
            if not files:
                print("No files found in API response.")
                return None
 
            dataframes = []
 
            # Step 2: Download each file and convert to DataFrame
            for file_info in files:
                file_name = file_info["file_name"]
                file_url = file_info["url"]
 
                print(f"Downloading file: {file_name}")
                file_response = requests.get(file_url)
 
                if file_response.status_code == 200:
                    file_content = file_response.content.decode("utf-8")
                    try:
                        df = pd.read_csv(BytesIO(file_content))
                        dataframes.append((file_name, df))
                        print(f"Loaded {file_name} into DataFrame.")
                    except Exception as e:
                        print(f"Error reading {file_name} as DataFrame: {e}")
                else:
                    print(f"Failed to download {file_name}. Status code: {file_response.status_code}")
 
            return dataframes  # List of (file_name, DataFrame)
 
        except Exception as e:
            print(f"Error processing API response: {e}")
            return None
 
    else:
        print(f"Failed to connect to the API. Status code: {response.status_code}")
        return None
s3_bucket_data()
 
# def s3_bucket_data():
#     response = requests.get("http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com:8000/api/files_download/")
 
#     if response.status_code == 200:
#         print("Successfully connected to the Evidently API.")
#         content_type = response.headers.get("Content-Type", "")
#         data = response.content.decode("utf-8")
#         # print(data)
       
#         # Convert to DataFrame - assuming CSV format
#         try:
#             df = pd.read_csv(StringIO(data))
#             return df
#         except Exception as e:
#             print(f"Error converting to DataFrame: {e}")
#             return data  # Return raw data if conversion fails
#     else:
#         print(f"Failed to connect to the Evidently API. Status code: {response.status_code}")
#         return None
 
class DriftAnalyzer:
    def __init__(self, api_url=None, jwt_token=None, user_id=None, tenant_id=None):
        self.reference_df = None
        self.current_df = None
        # S3 upload configuration
        self.api_url = api_url
        self.jwt_token = jwt_token
        self.user_id = user_id
        self.tenant_id = tenant_id
 
    def load_data_from_s3(self, split_pct=70, drop_cols=None, date_col=None, ref_range=None, cur_range=None):
        """Load data from S3 and prepare reference and current datasets for drift analysis."""
        try:
            data = s3_bucket_data()
 
            if data is None:
                print("No data returned from S3.")
                return None, None
 
            ref_df, cur_df = None, None
 
            if isinstance(data, list):
                if len(data) == 1:
                    file_name, df = data[0]
                    if not isinstance(df, pd.DataFrame):
                        print(f"{file_name} is not a valid DataFrame.")
                        return None, None
                    print(f"Single file detected: {file_name} — preprocessing and splitting.")
                    df = self.preprocess(df, drop_cols=drop_cols)
 
                    if date_col and ref_range and cur_range:
                        return self.split_by_date(df, date_col, ref_range, cur_range)
                    else:
                        return self.load_data(df, split_pct=split_pct)
 
                for file_name, df in data:
                    if not isinstance(df, pd.DataFrame):
                        print(f"{file_name} is not a valid DataFrame. Skipping.")
                        continue
 
                    if "ref" in file_name.lower():
                        print(f"Identified reference dataset: {file_name}")
                        ref_df = self.preprocess(df, drop_cols=drop_cols)
                    elif "cur" in file_name.lower():
                        print(f"Identified current dataset: {file_name}")
                        cur_df = self.preprocess(df, drop_cols=drop_cols)
                    else:
                        print(f"File {file_name} does not match 'ref' or 'cur'. Ignoring.")
 
                if ref_df is not None and cur_df is not None:
                    return self.load_data(ref_df, cur_df=cur_df)
                else:
                    print("Could not identify both reference and current datasets.")
                    return None, None
 
            else:
                print("Unexpected data format received.")
                return None, None
 
        except Exception as e:
            print(f"An error occurred while preparing data for drift analysis: {e}")
            return None, None
       
 
    def _save_and_upload_json(self, result, filename_prefix):
        """Save JSON report locally only."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{filename_prefix}_{timestamp}.json"
 
        try:
            # Ensure the folder exists
            folder = "result_json_files"
            os.makedirs(folder, exist_ok=True)
 
            # Construct a full file path with a filename (can use timestamp or UUID)
            temp_json_path = os.path.join(folder, json_filename)
 
            # Save the file
            result.save_json(temp_json_path)
            print(f"JSON saved locally: {temp_json_path}")
 
            return temp_json_path, None  # Always return None for upload_result
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return None, None
 
    def preprocess(self, df: pd.DataFrame, drop_cols: list = None) -> pd.DataFrame:
        """Drop user-specified columns from the dataset."""
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
        return df
 
    def split_by_date(self, df: pd.DataFrame, date_col: str, ref_range: tuple, cur_range: tuple):
        """Split the dataset into reference and current based on date ranges."""
        df[date_col] = pd.to_datetime(df[date_col])
        reference_df = df[(df[date_col] >= ref_range[0]) & (df[date_col] <= ref_range[1])]
        current_df = df[(df[date_col] >= cur_range[0]) & (df[date_col] <= cur_range[1])]
        self.reference_df = reference_df
        self.current_df = current_df
        return reference_df, current_df
 
    def load_data(self, ref_df, cur_df=None, split_pct=70):
        """Load and prepare data for drift analysis."""
        if cur_df is None:
            split_idx = int(len(ref_df) * split_pct / 100)
            self.reference_df = ref_df.iloc[:split_idx]
            self.current_df = ref_df.iloc[split_idx:]
        else:
            self.reference_df = ref_df
            self.current_df = cur_df
        return self.reference_df, self.current_df
 
    def generate_full_drift_report(self):
        """Generate full dataset drift report."""
        ds_ref = Dataset.from_pandas(self.reference_df)
        ds_cur = Dataset.from_pandas(self.current_df)
        report = Report(metrics=[DataDriftPreset()], include_tests= True)
        result = report.run(reference_data=ds_ref, current_data=ds_cur)
 
        # Save JSON locally only
        json_path, upload_result = self._save_and_upload_json(
            result, "full_drift_report"
        )
 
        # Get HTML
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.close()  # Close the file handle
        result.save_html(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp.name)
 
        # Get JSON for analysis
        drift_json = result.json()
        drift_dict = json.loads(drift_json) if drift_json else {}
 
        return html, drift_dict, json_path, upload_result
 
    def generate_column_drift_report(self, column):
        """Generate drift report for specific column."""
        ref_col_df = self.reference_df[[column]].copy()
        cur_col_df = self.current_df[[column]].copy()
 
        ds_ref = Dataset.from_pandas(ref_col_df)
        ds_cur = Dataset.from_pandas(cur_col_df)
        report = Report(metrics=[DataDriftPreset()], include_tests= True)
        result = report.run(reference_data=ds_ref, current_data=ds_cur)
 
        # Save JSON locally only
        json_path, upload_result = self._save_and_upload_json(
            result, f"column_drift_report_{column}"
        )
 
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.close()  # Close the file handle
        result.save_html(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp.name)
        drift_json = result.json()
        drift_dict = json.loads(drift_json) if drift_json else {}
 
        return html, drift_dict, json_path, upload_result
 
    def get_drift_summary(self, drift_dict):
        """Extract a complete drift summary across all columns."""
        summary_lines = []
 
        for test in drift_dict.get("tests", []):
            metric_cfg = test.get("metric_config", {})
            params = metric_cfg.get("params", {})
            col = params.get("column", "")
            desc = test.get("description", "")
 
            score_match = re.search(r"Drift score is ([0-9.]+)", desc)
            method_match = re.search(r"The drift detection method is (.+?)\.", desc)
            threshold_match = re.search(r"The drift threshold is ([0-9.]+)", desc)
 
            drift_score = float(score_match.group(1).rstrip(".")) if score_match else None
            threshold = float(threshold_match.group(1).rstrip(".")) if threshold_match else None
            method = method_match.group(1) if method_match else "Unknown"
 
            drift_flag = drift_score is not None and threshold is not None and drift_score > threshold
 
            summary_lines.append(
                f"- **Column**: `{col}`\n"
                f"  - Drift Detected: {'Yes ✅' if drift_flag else 'No ❌'}\n"
                f"  - Detection Method: {method}\n"
                f"  - Drift Score: {drift_score if drift_score is not None else 'N/A'}\n"
                f"  - Threshold: {threshold if threshold is not None else 'N/A'}"
            )
 
        return "\n\n".join(summary_lines) or "No drift test results found."
 
    def get_column_drift_summary(
        self,
        drift_dict: dict,
        column_name: str,
        psi_threshold: float = 0.10,
    ) -> str:
        """
        Produce a markdown summary for a specific column from the drift_dict,
        using whatever stat test or method Evidently reported.
        """
        for test in drift_dict.get("tests", []):
            metric_cfg = test.get("metric_config", {})
            params = metric_cfg.get("params", {})
            col = params.get("column", "")
            if col != column_name:
                continue
 
            desc = test.get("description", "")
            drift_score = None
            threshold = None
            method = None
 
            # Extract from description text using patterns
            score_match = re.search(r"Drift score is ([0-9.]+)", desc)
            method_match = re.search(r"The drift detection method is (.+?)\.", desc)
            threshold_match = re.search(r"The drift threshold is ([0-9.]+)", desc)
 
            # if score_match:
            #     drift_score = float(score_match.group(1))
            drift_score = float(score_match.group(1).rstrip(".")) if score_match else None
            threshold = float(threshold_match.group(1).rstrip(".")) if threshold_match else None
            
 
            drift_flag = drift_score is not None and threshold is not None and drift_score > threshold
 
            return (
                f"- **Column**: `{column_name}`\n"
                f"  - Drift Detected: {'Yes ✅' if drift_flag else 'No ❌'}\n"
                f"  - Detection Method: {method or 'Unknown'}\n"
                f"  - Drift Score: {drift_score if drift_score is not None else 'N/A'}\n"
                f"  - Threshold: {threshold if threshold is not None else 'N/A'}"
            )
 
        return f"- No drift information found for column `{column_name}`."
 
 
class LLMAnalyzer:
    def __init__(self):
        self.bedrock_runtime = None
        self._initialize_bedrock()
 
    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("REGION_LLM", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
            )
        except Exception as e:
            print(f"Bedrock initialization failed: {e}")
 
    def analyze_full_drift(self, ref_df, cur_df, summary_text):
        """Generate comprehensive drift analysis."""
        prompt = f"""
    You are a data analysis assistant helping interpret data drift results between a reference dataset and a current dataset.
 
    Below is metadata and a structured summary of the drift analysis:
 
    DATASET INFO:
    - Reference dataset shape: {ref_df.shape}
    - Current dataset shape: {cur_df.shape}
    - Number of features analyzed: {len(ref_df.columns)}
    - Sample features: {', '.join(ref_df.columns[:10])}{'...' if len(ref_df.columns) > 10 else ''}
 
    DRIFT RESULTS:
    {summary_text}
 
    Please analyze the results and provide:
 
    1. **High-Level Summary**
    - Is there meaningful drift in the dataset?
    - What percentage of columns show changes?
 
    2. **Column-Level Observations**
    - List affected columns with their drift score, threshold, and test used
    - Explain whether each shift is significant and what it suggests
 
    3. **Potential Causes**
    - Offer hypotheses for why changes occurred (seasonal patterns, source changes, etc.)
 
    4. **Data Quality and Operational Impact**
    - Suggest how drift may affect dashboards, decisions, or downstream use
 
    5. **Recommended Actions**
    - Highlight any next steps: verifying sources, cleaning data, or updating monitoring

    6. **Why This Method Was Used and not the other methods in this particular analysis **
- Explain why the chosen drift detection method is appropriate for this particular column  

    7.**How does This Method Works**
- Briefly describe how the method (e.g., KS test, PSI, Wasserstein distance) detects drift
 
    Present your analysis clearly in markdown using sections and bullet points. Assume no machine learning model is involved.
    """
        return self._invoke_claude(prompt)
 
 
    def analyze_column_drift(self, summary):
        """Generate column-specific drift analysis using summary."""
        prompt = f"""
    You are a data analysis expert reviewing drift for a single column in a dataset.
 
    Below is the extracted drift summary for the column:
 
    {summary}
 
    Please evaluate:
 
    1. **Drift Presence and Magnitude**
    - Is drift statistically significant?
    - Include the detection method, drift score, and threshold.
 
    2. **What Might Have Caused It**
    - Offer likely explanations for the observed change
 
    3. **Data Reliability**
    - How should this column be treated in future data monitoring?
 
    4. **Suggested Next Steps**
    - Recommend any immediate actions or longer-term data review
 
    Use clear markdown formatting with headings and short bullet points for easy readability.
    """
        return self._invoke_claude(prompt)
 
    def _invoke_claude(self, prompt):
        """Invoke Claude via AWS Bedrock."""
        if not self.bedrock_runtime:
            return "Claude invocation failed: Bedrock not initialized"
 
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 130000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }
 
            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps(request_body),
            )
 
            response_body = json.loads(response["body"].read().decode("utf-8"))
            print(response_body["content"][0]["text"])
            return response_body["content"][0]["text"]
        except Exception as e:
            return f"Claude invocation failed: {e}"
       
# analyzer = DriftAnalyzer()
# ref_df, cur_df = analyzer.load_data_from_s3()
 
# if ref_df is not None:
#     html, drift_dict, json_path, upload_result = analyzer.generate_full_drift_report()
 