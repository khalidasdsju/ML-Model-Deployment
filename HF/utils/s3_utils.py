import os
import boto3
import botocore
from HF.logger import logging
from HF.exception import HFException
import sys

class S3Utils:
    """Utility class for AWS S3 operations"""
    
    def __init__(self, bucket_name="hf-predication-model2025", 
                 aws_access_key_id=None, aws_secret_access_key=None):
        """
        Initialize S3 client
        
        Parameters:
        -----------
        bucket_name : str
            S3 bucket name
        aws_access_key_id : str, optional
            AWS access key ID. If None, will use environment variables or AWS configuration
        aws_secret_access_key : str, optional
            AWS secret access key. If None, will use environment variables or AWS configuration
        """
        try:
            self.bucket_name = bucket_name
            
            # Initialize S3 client
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key
                )
            else:
                self.s3_client = boto3.client('s3')
            
            # Check if bucket exists, create if it doesn't
            self._ensure_bucket_exists()
            
            logging.info(f"S3Utils initialized with bucket: {bucket_name}")
        except Exception as e:
            logging.error(f"Error initializing S3Utils: {e}")
            raise HFException(f"Error initializing S3Utils: {e}", sys)
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, create it if it doesn't"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logging.info(f"Bucket {self.bucket_name} exists")
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logging.info(f"Bucket {self.bucket_name} does not exist, creating...")
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                logging.info(f"Bucket {self.bucket_name} created")
            else:
                logging.error(f"Error checking bucket: {e}")
                raise HFException(f"Error checking bucket: {e}", sys)
    
    def upload_file(self, file_path, s3_key=None):
        """
        Upload a file to S3
        
        Parameters:
        -----------
        file_path : str
            Local file path
        s3_key : str, optional
            S3 object key. If None, will use the file name
            
        Returns:
        --------
        str
            S3 URI of the uploaded file
        """
        try:
            # If s3_key is not provided, use the file name
            if s3_key is None:
                s3_key = os.path.basename(file_path)
            
            # Upload file
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            
            # Generate S3 URI
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            
            logging.info(f"File {file_path} uploaded to {s3_uri}")
            
            return s3_uri
        except Exception as e:
            logging.error(f"Error uploading file {file_path}: {e}")
            raise HFException(f"Error uploading file {file_path}: {e}", sys)
    
    def download_file(self, s3_key, local_path=None):
        """
        Download a file from S3
        
        Parameters:
        -----------
        s3_key : str
            S3 object key
        local_path : str, optional
            Local file path. If None, will use the S3 key as the file name
            
        Returns:
        --------
        str
            Local path of the downloaded file
        """
        try:
            # If local_path is not provided, use the S3 key as the file name
            if local_path is None:
                local_path = os.path.basename(s3_key)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            logging.info(f"File {s3_key} downloaded to {local_path}")
            
            return local_path
        except Exception as e:
            logging.error(f"Error downloading file {s3_key}: {e}")
            raise HFException(f"Error downloading file {s3_key}: {e}", sys)
    
    def list_files(self, prefix=""):
        """
        List files in S3 bucket
        
        Parameters:
        -----------
        prefix : str, optional
            Prefix to filter objects
            
        Returns:
        --------
        list
            List of S3 object keys
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
                logging.info(f"Found {len(files)} files with prefix {prefix}")
                return files
            else:
                logging.info(f"No files found with prefix {prefix}")
                return []
        except Exception as e:
            logging.error(f"Error listing files with prefix {prefix}: {e}")
            raise HFException(f"Error listing files with prefix {prefix}: {e}", sys)
    
    def delete_file(self, s3_key):
        """
        Delete a file from S3
        
        Parameters:
        -----------
        s3_key : str
            S3 object key
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logging.info(f"File {s3_key} deleted")
            return True
        except Exception as e:
            logging.error(f"Error deleting file {s3_key}: {e}")
            raise HFException(f"Error deleting file {s3_key}: {e}", sys)
    
    def get_file_url(self, s3_key, expiration=3600):
        """
        Generate a presigned URL for a file
        
        Parameters:
        -----------
        s3_key : str
            S3 object key
        expiration : int, optional
            URL expiration time in seconds
            
        Returns:
        --------
        str
            Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            logging.info(f"Generated presigned URL for {s3_key}")
            return url
        except Exception as e:
            logging.error(f"Error generating presigned URL for {s3_key}: {e}")
            raise HFException(f"Error generating presigned URL for {s3_key}: {e}", sys)
