import pandas as pd
import joblib, pickle, json, chardet, boto3, tempfile, os, requests, re
from datetime import datetime
from evidently.presets import *
from evidently import Dataset, DataDefinition, Regression, Report
import warnings
import requests
from io import BytesIO
from evidently.metrics import *

 
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
    api_url = "http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com:8000/api/files_download/"
    access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJSZWdBbmFseXNpc0BjaXJydXNsYWJzLmlvIiwiZXhwIjoxNzUxNjEwNDkwfQ.F3V385gjm05MKo28nlV_HKq0vRg-BuV5iqzCyJ85hDo"

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        print("Successfully connected to the API.")

        try:
            json_data = response.json()
            files = json_data.get("files", [])

            if not files:
                print("No files found in API response.")
                return None

            dataframes = []

            for file_info in files:
                file_name = file_info["file_name"]
                file_url = file_info["url"]

                print(f"Downloading file: {file_name}")
                file_response = requests.get(file_url)

                if file_response.status_code == 200:
                    file_content = file_response.content

                    # ✅ Handle model files separately
                    if file_name.endswith((".pkl", ".joblib")):
                        dataframes.append((file_name, file_content))
                        print(f"Loaded {file_name} as binary model file.")
                        continue

                    try:
                        df = load_flexible_csv(BytesIO(file_content))
                        dataframes.append((file_name, df))
                        print(f"Loaded {file_name} into DataFrame.")
                    except Exception as e:
                        print(f"Error reading {file_name} as DataFrame: {e}")
                else:
                    print(f"Failed to download {file_name}. Status code: {file_response.status_code}")

            return dataframes

        except Exception as e:
            print(f"Error processing API response: {e}")
            return None

    else:
        print(f"Failed to connect to the API. Status code: {response.status_code}")
        return None

 
class RegressionAnalyzer:
    def __init__(self, api_url=None, jwt_token=None, user_id=None, tenant_id=None):
        self.reference_df = None
        self.current_df = None
        # S3 upload configuration
        self.api_url = api_url
        self.jwt_token = jwt_token
        self.user_id = user_id
        self.tenant_id = tenant_id

    def load_data_from_s3(self, split_pct=70, drop_cols=None, date_col=None, ref_range=None, cur_range=None):
        """Load data and model from S3, and prepare reference and current datasets for drift analysis."""
        try:
            data = s3_bucket_data()

            if data is None:
                print("No data returned from S3.")
                return None, None, None  # include model

            ref_df, cur_df, model = None, None, None

            if isinstance(data, list):
                for file_name, content in data:
                    # ✅ Model loading
                    if file_name.endswith((".pkl", ".joblib")):
                        print(f"📦 Detected model file: {file_name}")
                        if isinstance(content, bytes):
                            model = self.load_model_safely(content)
                            print(f"✅ Model loaded: {type(model)}")
                        else:
                            print(f"❌ Skipping {file_name} - not bytes, type={type(content)}")
                        continue  # move on to next file

                    # ✅ Dataset loading
                    if not isinstance(content, pd.DataFrame):
                        print(f"{file_name} is not a valid DataFrame. Skipping.")
                        continue

                    if "ref" in file_name.lower():
                        print(f"📘 Identified reference dataset: {file_name}")
                        ref_df = self.preprocess(content, drop_cols=drop_cols)
                    elif "cur" in file_name.lower():
                        print(f"📗 Identified current dataset: {file_name}")
                        cur_df = self.preprocess(content, drop_cols=drop_cols)
                    else:
                        print(f"📄 File {file_name} does not match 'ref' or 'cur'. Ignoring.")


                if ref_df is not None and cur_df is not None:
                    ref_df, cur_df = self.load_data(ref_df, cur_df=cur_df)
                    return ref_df, cur_df, model
                elif ref_df is not None:
                    print("Current dataset not found — splitting reference dataset.")
                    ref_df, cur_df = self.load_data(ref_df, split_pct=split_pct)
                    return ref_df, cur_df, model
                else:
                    print("Could not identify any usable dataset.")
                    return None, None, model


            return None, None, None

        except Exception as e:
            print(f"An error occurred while preparing data for Regression analysis: {e}")
            return None, None, None

       
 
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
    
    def load_model_safely(self, file_content):
        try:
            print(f"🔍 Model content type: {type(file_content)}")
            if isinstance(file_content, bytes):
                try:
                    model = joblib.load(BytesIO(file_content))
                    print("✅ Model loaded with joblib")
                    return model
                except Exception as e:
                    print(f"⚠️ Joblib failed: {e}, trying pickle...")
                    return pickle.load(BytesIO(file_content))
            else:
                print(f"❌ Unsupported type for model loading: {type(file_content)}")
                return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None



 
    def Regression_report(self, target_column_name: str, Model_File):
        """Generate full regression dataset drift report with dynamic target and model."""

        # Step 1: Rename target column to 'target' in both DataFrames
        ref_df = self.reference_df.copy()
        cur_df = self.current_df.copy()

        if target_column_name not in ref_df.columns or target_column_name not in cur_df.columns:
            raise ValueError(f"Target column '{target_column_name}' not found in both datasets.")

        ref_df = ref_df.rename(columns={target_column_name: 'target'})
        cur_df = cur_df.rename(columns={target_column_name: 'target'})

        # Step 2: Use the already loaded model (don't load it again)
        if Model_File is None:
            raise ValueError("❌ Model is None. Ensure the model was loaded correctly from S3.")
        
        # Check if Model_File is already a loaded model object
        if hasattr(Model_File, 'predict'):
            model = Model_File
            print(f"✅ Using already loaded model: {type(model)}")
        else:
            # If it's still bytes, load it
            model = self.load_model_safely(Model_File)
            if model is None:
                raise ValueError("❌ Model could not be loaded. Ensure the uploaded file is a valid pickle/joblib model.")

        # Step 3: Prepare feature inputs for prediction (don't modify original DataFrames)
        X_ref_full = ref_df.drop(columns=['target']).copy()
        X_cur_full = cur_df.drop(columns=['target']).copy()

        # Step 4: Align features based on training feature names
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_ref_full.columns

        # Create aligned versions (add missing, drop extra)
        X_ref = X_ref_full.copy()
        X_cur = X_cur_full.copy()

        for col in expected_features:
            if col not in X_ref.columns:
                X_ref[col] = 0
            if col not in X_cur.columns:
                X_cur[col] = 0

        X_ref = X_ref[expected_features]
        X_cur = X_cur[expected_features]

        # Step 5: Predict
        ref_df['prediction'] = model.predict(X_ref)
        cur_df['prediction'] = model.predict(X_cur)


        # Step 5: Define schema and run report
        data_definition = DataDefinition(regression=[Regression(target='target', prediction_labels='prediction')])
        ds_ref = Dataset.from_pandas(ref_df, data_definition=data_definition)
        ds_cur = Dataset.from_pandas(cur_df, data_definition=data_definition)

        report = Report(metrics=[RegressionPreset()], include_tests=True)
        result = report.run(reference_data=ds_ref, current_data=ds_cur)

        # Step 6: Save JSON locally
        json_path, upload_result = self._save_and_upload_json(result, "Regression_report")

        # Step 7: Save and return HTML
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.close()
        result.save_html(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp.name)

        # Step 8: Get result as dict
        regression_json = result.json()
        regression_dict = json.loads(regression_json) if regression_json else {}

        return html, regression_dict, json_path, upload_result
    
    def get_regression_summary(self, regression_dict):
        """Extract a summary of regression performance metrics from the Evidently report."""
        summary_text = []
        
        try:
            # Debug: Print the keys to understand structure
            print("DEBUG: regression_dict keys:", list(regression_dict.keys()))
            
            # Method 1: Extract from tests section
            if "tests" in regression_dict:
                for i, test in enumerate(regression_dict["tests"]):
                    if not isinstance(test, dict):
                        continue
                    
                    test_name = test.get("name", f"Test_{i}")
                    status = test.get("status", "Unknown")
                    description = test.get("description", "")
                    
                    summary_text.append(f"### 📊 {test_name}")
                    summary_text.append(f"- **Status**: {status}")
                    
                    if description:
                        summary_text.append(f"- **Details**: {description}")
                    
                    # Extract parameters if available
                    if "parameters" in test:
                        params = test["parameters"]
                        if isinstance(params, dict):
                            for key, value in params.items():
                                if isinstance(value, (int, float)):
                                    summary_text.append(f"- **{key}**: {value:.4f}")
                    
                    summary_text.append("---")
            
            # Method 2: Extract from metrics section
            if "metrics" in regression_dict:
                metrics = regression_dict["metrics"]
                summary_text.append("### 📈 Performance Metrics")
                
                # Function to recursively extract numerical values
                def extract_numbers(obj, prefix=""):
                    results = []
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_prefix = f"{prefix}.{key}" if prefix else key
                            if isinstance(value, (int, float)):
                                results.append(f"- **{key}**: {value:.4f}")
                            elif isinstance(value, dict):
                                # Check for reference/current structure
                                if "current" in value and "reference" in value:
                                    results.append(f"- **{key}**:")
                                    results.append(f"  - Reference: {value['reference']:.4f}")
                                    results.append(f"  - Current: {value['current']:.4f}")
                                else:
                                    nested = extract_numbers(value, current_prefix)
                                    results.extend(nested)
                    return results
                
                metric_results = extract_numbers(metrics)
                if metric_results:
                    summary_text.extend(metric_results)
                else:
                    summary_text.append("- No numerical metrics found")
                
                summary_text.append("---")
            
            # If no structured data found, try to get any useful info
            if len(summary_text) == 0:
                summary_text.append("### ⚠️ Unable to extract structured metrics")
                summary_text.append("Raw data keys available:")
                for key in regression_dict.keys():
                    summary_text.append(f"- {key}: {type(regression_dict[key])}")
            
        except Exception as e:
            print(f"Error extracting regression summary: {e}")
            return f"Error: {e}"
        
        result = "\n".join(summary_text)
        print("DEBUG: Summary result:", result[:500] + "..." if len(result) > 500 else result)
        return result if result.strip() else "No regression metrics could be extracted."

 
 
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
 
    def analyze_Regrresion_Report(self, ref_df, cur_df, summary_text):
        """Generate comprehensive drift analysis."""
        prompt = f"""
            You are a data analysis assistant helping interpret regression model performance results between a reference dataset and a current dataset.
            
            Below is metadata and a structured summary of the performance analysis:
            
            DATASET INFO:
            - Reference dataset shape: {ref_df.shape}
            - Current dataset shape: {cur_df.shape}
            - Number of features analyzed: {len(ref_df.columns)}
            - Sample features: {', '.join(ref_df.columns[:10])}{'...' if len(ref_df.columns) > 10 else ''}
            
            REGRESSION PERFORMANCE RESULTS:
            {summary_text}
            
            Please analyze the results and provide:
            
            1. **High-Level Summary**
            - Has the model's predictive performance changed meaningfully between the reference and current datasets?
            - Highlight any significant shifts in metrics like MAE, MSE, RMSE, R², and MAPE.
            - Are the changes within expected variation, or do they suggest performance degradation?
            
            2. **Metric-Level Observations**
            - List key metrics for both reference and current datasets, with mean and standard deviation.
            - For each metric, explain if the difference is substantial and what it implies for real-world predictions.
            
            3. **Possible Causes for Performance Change**
            - Suggest plausible reasons for differences in model performance. These might include:
            - Changes in data distribution or feature ranges
            - Seasonal or temporal effects
            - Data quality issues (e.g., missing values, new outliers)
            - Concept drift (underlying real-world relationships changed)
            
            4. **Operational and Business Impact**
            - Discuss how performance shifts may affect real users, automated systems, or decision-making.
            - Identify potential risks like inaccurate forecasts, poor targeting, or misallocations.
            
            5. **Recommended Actions**
            - Suggest next steps to improve reliability, such as:
            - Retraining the model
            - Validating current data integrity
            - Adding new features or regularization
            - Updating performance monitoring thresholds
            
            6. **Why This Evaluation Approach Was Used**
            - Explain why this performance evaluation method was chosen (e.g., RegressionPreset).
            - Describe what makes it suitable for this regression use case.
            
            7. **How the Metrics Are Computed**
            - Briefly explain the key regression metrics used in the analysis:
            - **MAE**: Mean absolute difference between predicted and actual values.
            - **MAPE**: Percentage-based error metric, useful for scale-invariant insights.
            - **RMSE**: Penalizes larger errors more than MAE.
            - **R²**: Indicates proportion of variance explained by the model.
            
            Present your analysis clearly in markdown using structured sections and bullet points. Do not refer to "drift detection" — this is a regression evaluation context.
            """
        return self._invoke_claude(prompt)
    
    def _invoke_claude(self, prompt):
        """Invoke Claude model via AWS Bedrock."""
        if not self.bedrock_runtime:
            return "❌ Bedrock client not initialized. Please check your AWS credentials."
        
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=body
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('content', [{}])[0].get('text', 'No response generated')
            
        except Exception as e:
            return f"❌ Error invoking Claude: {str(e)}"




# Only run the analysis if this file is executed directly, not when imported
if __name__ == "__main__":
    analyzer = RegressionAnalyzer()
    LLM_Summary = LLMAnalyzer()

    # Load data from S3
    ref_df, cur_df, model = analyzer.load_data_from_s3()

    if ref_df is not None and cur_df is not None:
        # Specify your target column name
        target_column = "MedHouseVal"  # Replace with your actual target column
        
        # Generate regression report
        html, regression_dict, json_path, upload_result = analyzer.Regression_report(
            target_column_name=target_column, 
            Model_File=model  # Pass the loaded model
        )
        
        # Get regression summary from the report
        summary_text = analyzer.get_regression_summary(regression_dict)
        #print(summary_text)
        # Call LLM analyzer with the summary
        llm_analysis = LLM_Summary.analyze_Regrresion_Report(
            ref_df=ref_df,
            cur_df=cur_df, 
            summary_text=summary_text
        )
        
        # Print or use the LLM analysis
        print("=== LLM Analysis ===")
        print(llm_analysis)
        
        # Optionally save the analysis to a file
        with open("llm_regression_analysis.txt", "w") as f:
            f.write(llm_analysis)
            
    else:
        print("Could not load data for regression analysis")