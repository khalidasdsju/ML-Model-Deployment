import os
import sys
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HF.utils.s3_utils import S3Utils
from HF.logger import logging

def upload_model_to_s3(aws_access_key_id, aws_secret_access_key, bucket_name="hf-predication-model2025"):
    """
    Upload the model and important files to S3
    
    Parameters:
    -----------
    aws_access_key_id : str
        AWS access key ID
    aws_secret_access_key : str
        AWS secret access key
    bucket_name : str, optional
        S3 bucket name
    """
    try:
        # Initialize S3 client
        s3_utils = S3Utils(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Create a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Upload model files
        model_files = []
        
        # Upload the best model
        if os.path.exists("best_xgboost_model.pkl"):
            s3_key = f"models/{timestamp}/best_xgboost_model.pkl"
            s3_uri = s3_utils.upload_file("best_xgboost_model.pkl", s3_key)
            model_files.append(s3_uri)
        
        # Upload saved models
        if os.path.exists("saved_models/latest/xgboost_model.pkl"):
            s3_key = f"models/{timestamp}/xgboost_model.pkl"
            s3_uri = s3_utils.upload_file("saved_models/latest/xgboost_model.pkl", s3_key)
            model_files.append(s3_uri)
        
        # Upload model configuration
        if os.path.exists("config/prediction_params.yaml"):
            s3_key = f"config/{timestamp}/prediction_params.yaml"
            s3_uri = s3_utils.upload_file("config/prediction_params.yaml", s3_key)
            model_files.append(s3_uri)
        
        # Upload feature importance file
        if os.path.exists("data/feature_importance_cv.csv"):
            s3_key = f"data/{timestamp}/feature_importance_cv.csv"
            s3_uri = s3_utils.upload_file("data/feature_importance_cv.csv", s3_key)
            model_files.append(s3_uri)
        
        # Upload model comparison file
        if os.path.exists("data/model_comparison.csv"):
            s3_key = f"data/{timestamp}/model_comparison.csv"
            s3_uri = s3_utils.upload_file("data/model_comparison.csv", s3_key)
            model_files.append(s3_uri)
        
        # Upload reports
        report_files = []
        
        # Upload analysis summary
        if os.path.exists("reports/analysis_summary.txt"):
            s3_key = f"reports/{timestamp}/analysis_summary.txt"
            s3_uri = s3_utils.upload_file("reports/analysis_summary.txt", s3_key)
            report_files.append(s3_uri)
        
        # Upload technical analysis
        if os.path.exists("reports/technical_analysis.txt"):
            s3_key = f"reports/{timestamp}/technical_analysis.txt"
            s3_uri = s3_utils.upload_file("reports/technical_analysis.txt", s3_key)
            report_files.append(s3_uri)
        
        # Upload clinical interpretation
        if os.path.exists("reports/clinical_interpretation.txt"):
            s3_key = f"reports/{timestamp}/clinical_interpretation.txt"
            s3_uri = s3_utils.upload_file("reports/clinical_interpretation.txt", s3_key)
            report_files.append(s3_uri)
        
        # Upload executive summary
        if os.path.exists("reports/executive_summary.txt"):
            s3_key = f"reports/{timestamp}/executive_summary.txt"
            s3_uri = s3_utils.upload_file("reports/executive_summary.txt", s3_key)
            report_files.append(s3_uri)
        
        # Upload combined report
        if os.path.exists("reports/combined_report.txt"):
            s3_key = f"reports/{timestamp}/combined_report.txt"
            s3_uri = s3_utils.upload_file("reports/combined_report.txt", s3_key)
            report_files.append(s3_uri)
        
        # Upload visualizations
        visualization_files = []
        
        # Upload all PNG files in the visualizations folder
        if os.path.exists("visualizations"):
            for file in os.listdir("visualizations"):
                if file.endswith(".png"):
                    s3_key = f"visualizations/{timestamp}/{file}"
                    s3_uri = s3_utils.upload_file(f"visualizations/{file}", s3_key)
                    visualization_files.append(s3_uri)
        
        # Print summary
        print("\n=== Upload Summary ===")
        print(f"Timestamp: {timestamp}")
        print(f"Model files uploaded: {len(model_files)}")
        for uri in model_files:
            print(f"  - {uri}")
        
        print(f"\nReport files uploaded: {len(report_files)}")
        for uri in report_files:
            print(f"  - {uri}")
        
        print(f"\nVisualization files uploaded: {len(visualization_files)}")
        for uri in visualization_files:
            print(f"  - {uri}")
        
        print(f"\nTotal files uploaded: {len(model_files) + len(report_files) + len(visualization_files)}")
        
        # Create a manifest file
        manifest_content = f"""# Heart Failure Model S3 Upload Manifest
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Timestamp: {timestamp}
Bucket: {bucket_name}

## Model Files
{chr(10).join([f"- {uri}" for uri in model_files])}

## Report Files
{chr(10).join([f"- {uri}" for uri in report_files])}

## Visualization Files
{chr(10).join([f"- {uri}" for uri in visualization_files])}
"""
        
        # Save manifest locally
        manifest_path = f"s3_manifest_{timestamp}.txt"
        with open(manifest_path, "w") as f:
            f.write(manifest_content)
        
        # Upload manifest
        s3_key = f"manifests/{timestamp}_manifest.txt"
        s3_uri = s3_utils.upload_file(manifest_path, s3_key)
        
        print(f"\nManifest file uploaded: {s3_uri}")
        print(f"Local manifest saved to: {manifest_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error uploading to S3: {e}")
        print(f"Error uploading to S3: {e}")
        return False

def download_model_from_s3(aws_access_key_id, aws_secret_access_key, timestamp=None, bucket_name="hf-predication-model2025"):
    """
    Download the model and important files from S3
    
    Parameters:
    -----------
    aws_access_key_id : str
        AWS access key ID
    aws_secret_access_key : str
        AWS secret access key
    timestamp : str, optional
        Timestamp of the version to download. If None, will use the latest version
    bucket_name : str, optional
        S3 bucket name
    """
    try:
        # Initialize S3 client
        s3_utils = S3Utils(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # If timestamp is not provided, find the latest version
        if timestamp is None:
            # List all manifests
            manifests = s3_utils.list_files("manifests/")
            
            if not manifests:
                print("No manifests found in S3")
                return False
            
            # Sort manifests by timestamp (assuming format: manifests/TIMESTAMP_manifest.txt)
            manifests.sort(reverse=True)
            
            # Extract timestamp from the latest manifest
            latest_manifest = manifests[0]
            timestamp = latest_manifest.split("/")[1].split("_")[0]
            
            print(f"Using latest version: {timestamp}")
        
        # Download model files
        model_files = []
        
        # Download the best model
        s3_key = f"models/{timestamp}/best_xgboost_model.pkl"
        try:
            local_path = s3_utils.download_file(s3_key, "best_xgboost_model.pkl")
            model_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download saved model
        s3_key = f"models/{timestamp}/xgboost_model.pkl"
        try:
            # Create directory if it doesn't exist
            os.makedirs("saved_models/latest", exist_ok=True)
            
            local_path = s3_utils.download_file(s3_key, "saved_models/latest/xgboost_model.pkl")
            model_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download model configuration
        s3_key = f"config/{timestamp}/prediction_params.yaml"
        try:
            # Create directory if it doesn't exist
            os.makedirs("config", exist_ok=True)
            
            local_path = s3_utils.download_file(s3_key, "config/prediction_params.yaml")
            model_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download feature importance file
        s3_key = f"data/{timestamp}/feature_importance_cv.csv"
        try:
            # Create directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            local_path = s3_utils.download_file(s3_key, "data/feature_importance_cv.csv")
            model_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download model comparison file
        s3_key = f"data/{timestamp}/model_comparison.csv"
        try:
            local_path = s3_utils.download_file(s3_key, "data/model_comparison.csv")
            model_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download reports
        report_files = []
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Download analysis summary
        s3_key = f"reports/{timestamp}/analysis_summary.txt"
        try:
            local_path = s3_utils.download_file(s3_key, "reports/analysis_summary.txt")
            report_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download technical analysis
        s3_key = f"reports/{timestamp}/technical_analysis.txt"
        try:
            local_path = s3_utils.download_file(s3_key, "reports/technical_analysis.txt")
            report_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download clinical interpretation
        s3_key = f"reports/{timestamp}/clinical_interpretation.txt"
        try:
            local_path = s3_utils.download_file(s3_key, "reports/clinical_interpretation.txt")
            report_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download executive summary
        s3_key = f"reports/{timestamp}/executive_summary.txt"
        try:
            local_path = s3_utils.download_file(s3_key, "reports/executive_summary.txt")
            report_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download combined report
        s3_key = f"reports/{timestamp}/combined_report.txt"
        try:
            local_path = s3_utils.download_file(s3_key, "reports/combined_report.txt")
            report_files.append(local_path)
        except Exception as e:
            logging.warning(f"Could not download {s3_key}: {e}")
        
        # Download visualizations
        visualization_files = []
        
        # Create visualizations directory if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)
        
        # List visualization files
        visualization_keys = s3_utils.list_files(f"visualizations/{timestamp}/")
        
        # Download each visualization file
        for s3_key in visualization_keys:
            try:
                file_name = os.path.basename(s3_key)
                local_path = s3_utils.download_file(s3_key, f"visualizations/{file_name}")
                visualization_files.append(local_path)
            except Exception as e:
                logging.warning(f"Could not download {s3_key}: {e}")
        
        # Print summary
        print("\n=== Download Summary ===")
        print(f"Timestamp: {timestamp}")
        print(f"Model files downloaded: {len(model_files)}")
        for path in model_files:
            print(f"  - {path}")
        
        print(f"\nReport files downloaded: {len(report_files)}")
        for path in report_files:
            print(f"  - {path}")
        
        print(f"\nVisualization files downloaded: {len(visualization_files)}")
        for path in visualization_files:
            print(f"  - {path}")
        
        print(f"\nTotal files downloaded: {len(model_files) + len(report_files) + len(visualization_files)}")
        
        return True
    except Exception as e:
        logging.error(f"Error downloading from S3: {e}")
        print(f"Error downloading from S3: {e}")
        return False

def list_s3_versions(aws_access_key_id, aws_secret_access_key, bucket_name="hf-predication-model2025"):
    """
    List all versions of the model in S3
    
    Parameters:
    -----------
    aws_access_key_id : str
        AWS access key ID
    aws_secret_access_key : str
        AWS secret access key
    bucket_name : str, optional
        S3 bucket name
    """
    try:
        # Initialize S3 client
        s3_utils = S3Utils(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # List all manifests
        manifests = s3_utils.list_files("manifests/")
        
        if not manifests:
            print("No manifests found in S3")
            return []
        
        # Sort manifests by timestamp (assuming format: manifests/TIMESTAMP_manifest.txt)
        manifests.sort(reverse=True)
        
        # Print versions
        print("\n=== Available Versions ===")
        versions = []
        
        for manifest in manifests:
            # Extract timestamp from manifest
            timestamp = manifest.split("/")[1].split("_")[0]
            versions.append(timestamp)
            
            # Download manifest to get details
            try:
                local_path = s3_utils.download_file(manifest, f"temp_manifest_{timestamp}.txt")
                
                # Read manifest
                with open(local_path, "r") as f:
                    lines = f.readlines()
                
                # Extract generation date
                generation_date = "Unknown"
                for line in lines:
                    if line.startswith("Generated:"):
                        generation_date = line.split("Generated:")[1].strip()
                        break
                
                print(f"Version: {timestamp}")
                print(f"  Generated: {generation_date}")
                print(f"  Manifest: {manifest}")
                
                # Count files
                model_count = 0
                report_count = 0
                visualization_count = 0
                
                in_model_section = False
                in_report_section = False
                in_visualization_section = False
                
                for line in lines:
                    if "## Model Files" in line:
                        in_model_section = True
                        in_report_section = False
                        in_visualization_section = False
                        continue
                    elif "## Report Files" in line:
                        in_model_section = False
                        in_report_section = True
                        in_visualization_section = False
                        continue
                    elif "## Visualization Files" in line:
                        in_model_section = False
                        in_report_section = False
                        in_visualization_section = True
                        continue
                    
                    if line.strip().startswith("- s3://"):
                        if in_model_section:
                            model_count += 1
                        elif in_report_section:
                            report_count += 1
                        elif in_visualization_section:
                            visualization_count += 1
                
                print(f"  Files: {model_count + report_count + visualization_count} (Models: {model_count}, Reports: {report_count}, Visualizations: {visualization_count})")
                print()
                
                # Remove temporary manifest
                os.remove(local_path)
            except Exception as e:
                logging.warning(f"Could not download manifest {manifest}: {e}")
                print(f"Version: {timestamp}")
                print(f"  Manifest: {manifest}")
                print(f"  Error: Could not download manifest")
                print()
        
        return versions
    except Exception as e:
        logging.error(f"Error listing S3 versions: {e}")
        print(f"Error listing S3 versions: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload/download heart failure model to/from S3")
    parser.add_argument("--action", choices=["upload", "download", "list"], required=True,
                        help="Action to perform: upload, download, or list versions")
    parser.add_argument("--access-key", required=True,
                        help="AWS access key ID")
    parser.add_argument("--secret-key", required=True,
                        help="AWS secret access key")
    parser.add_argument("--bucket", default="hf-predication-model2025",
                        help="S3 bucket name")
    parser.add_argument("--timestamp", 
                        help="Timestamp of the version to download (only for download action)")
    
    args = parser.parse_args()
    
    if args.action == "upload":
        upload_model_to_s3(args.access_key, args.secret_key, args.bucket)
    elif args.action == "download":
        download_model_from_s3(args.access_key, args.secret_key, args.timestamp, args.bucket)
    elif args.action == "list":
        list_s3_versions(args.access_key, args.secret_key, args.bucket)
