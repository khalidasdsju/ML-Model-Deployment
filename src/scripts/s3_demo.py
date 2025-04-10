import os
import sys
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HF.utils.s3_utils import S3Utils
from HF.logger import logging

def test_s3_connection(aws_access_key_id, aws_secret_access_key, bucket_name="hf-predication-model2025"):
    """
    Test the connection to AWS S3
    
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
        print(f"Testing connection to S3 bucket: {bucket_name}")
        
        # Initialize S3 client
        s3_utils = S3Utils(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Create a test file
        test_file_path = "s3_test.txt"
        with open(test_file_path, "w") as f:
            f.write(f"S3 connection test file\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Upload the test file
        s3_key = f"tests/s3_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        s3_uri = s3_utils.upload_file(test_file_path, s3_key)
        
        print(f"Test file uploaded to: {s3_uri}")
        
        # List files in the bucket
        files = s3_utils.list_files()
        
        print(f"\nFiles in bucket {bucket_name}:")
        for file in files:
            print(f"  - {file}")
        
        # Clean up
        os.remove(test_file_path)
        
        print("\nS3 connection test successful!")
        return True
    except Exception as e:
        print(f"Error testing S3 connection: {e}")
        return False

def list_s3_models(aws_access_key_id, aws_secret_access_key, bucket_name="hf-predication-model2025"):
    """
    List all models in the S3 bucket
    
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
        print(f"Listing models in S3 bucket: {bucket_name}")
        
        # Initialize S3 client
        s3_utils = S3Utils(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # List model files
        model_files = s3_utils.list_files("models/")
        
        if not model_files:
            print("No models found in S3 bucket")
            return []
        
        # Group models by timestamp
        models_by_timestamp = {}
        for file in model_files:
            # Extract timestamp from path (models/TIMESTAMP/...)
            parts = file.split("/")
            if len(parts) >= 3:
                timestamp = parts[1]
                if timestamp not in models_by_timestamp:
                    models_by_timestamp[timestamp] = []
                models_by_timestamp[timestamp].append(file)
        
        # Print models by timestamp
        print(f"\nModels in bucket {bucket_name}:")
        for timestamp, files in sorted(models_by_timestamp.items(), reverse=True):
            print(f"\nTimestamp: {timestamp}")
            for file in files:
                print(f"  - {file}")
        
        return model_files
    except Exception as e:
        print(f"Error listing S3 models: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AWS S3 integration")
    parser.add_argument("--action", choices=["test", "list"], required=True,
                        help="Action to perform: test connection or list models")
    parser.add_argument("--access-key", required=True,
                        help="AWS access key ID")
    parser.add_argument("--secret-key", required=True,
                        help="AWS secret access key")
    parser.add_argument("--bucket", default="hf-predication-model2025",
                        help="S3 bucket name")
    
    args = parser.parse_args()
    
    if args.action == "test":
        test_s3_connection(args.access_key, args.secret_key, args.bucket)
    elif args.action == "list":
        list_s3_models(args.access_key, args.secret_key, args.bucket)
