#!/usr/bin/env python3
"""
Google Cloud Storage Service for InspecAI
Handles uploading reports and images to GCS buckets
"""

import os
from datetime import timedelta
from typing import Optional
from google.cloud import storage
from google.oauth2 import service_account

# Configuration from environment variables
GCS_CREDENTIALS_PATH = os.environ.get("GCS_CREDENTIALS_PATH", "inspec-ai_storage.json")
GCS_PROJECT_ID = os.environ.get("GCS_PROJECT_ID", "calcium-bridge-483903-d9")
GCS_REPORTS_BUCKET = os.environ.get("GCS_REPORTS_BUCKET", "inspecai-reports")
GCS_IMAGES_BUCKET = os.environ.get("GCS_IMAGES_BUCKET", "inspecai-images")

# Signed URL expiration time (in minutes)
SIGNED_URL_EXPIRATION = int(os.environ.get("GCS_SIGNED_URL_EXPIRATION", 60))


class GCSService:
    """
    Google Cloud Storage service for uploading and managing files
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to reuse the GCS client"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.client = None
        self.reports_bucket = None
        self.images_bucket = None
        self._initialized = True
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize GCS client with service account credentials"""
        try:
            # Check if credentials file exists
            if os.path.exists(GCS_CREDENTIALS_PATH):
                credentials = service_account.Credentials.from_service_account_file(
                    GCS_CREDENTIALS_PATH
                )
                self.client = storage.Client(
                    project=GCS_PROJECT_ID,
                    credentials=credentials
                )
                print(f"[GCS] Initialized with service account from {GCS_CREDENTIALS_PATH}")
            else:
                # Fall back to Application Default Credentials (ADC)
                self.client = storage.Client(project=GCS_PROJECT_ID)
                print("[GCS] Initialized with Application Default Credentials")
            
            # Get bucket references
            self.reports_bucket = self.client.bucket(GCS_REPORTS_BUCKET)
            print(f"[GCS] Reports bucket: {GCS_REPORTS_BUCKET}")
            
            # Images bucket is optional
            if GCS_IMAGES_BUCKET:
                self.images_bucket = self.client.bucket(GCS_IMAGES_BUCKET)
                print(f"[GCS] Images bucket: {GCS_IMAGES_BUCKET}")
                
        except Exception as e:
            print(f"[GCS] Failed to initialize GCS client: {e}")
            print("[GCS] File uploads will be skipped, local storage only")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if GCS is available and configured"""
        return self.client is not None
    
    def upload_report(self, local_path: str, report_id: str) -> Optional[str]:
        """
        Upload a PDF report to the reports bucket
        
        Args:
            local_path: Local file path of the PDF
            report_id: Unique report ID for naming
            
        Returns:
            GCS URI (gs://bucket/path) or None if upload failed
        """
        if not self.is_available() or not self.reports_bucket:
            print("[GCS] GCS not available, skipping report upload")
            return None
        
        try:
            # Define blob name (path in bucket)
            blob_name = f"reports/{report_id}.pdf"
            blob = self.reports_bucket.blob(blob_name)
            
            # Upload file
            blob.upload_from_filename(local_path, content_type="application/pdf")
            
            gcs_uri = f"gs://{GCS_REPORTS_BUCKET}/{blob_name}"
            print(f"[GCS] Uploaded report to {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            print(f"[GCS] Failed to upload report: {e}")
            return None
    
    def upload_image(self, local_path: str, image_id: str, image_type: str = "outputs") -> Optional[str]:
        """
        Upload an image to the images bucket
        
        Args:
            local_path: Local file path of the image
            image_id: Unique image ID for naming
            image_type: "uploads" for original images, "outputs" for processed images
            
        Returns:
            GCS URI (gs://bucket/path) or None if upload failed
        """
        if not self.is_available() or not self.images_bucket:
            print("[GCS] GCS images bucket not available, skipping image upload")
            return None
        
        try:
            # Determine file extension
            _, ext = os.path.splitext(local_path)
            if not ext:
                ext = ".jpg"
            
            # Define blob name
            blob_name = f"{image_type}/{image_id}{ext}"
            blob = self.images_bucket.blob(blob_name)
            
            # Determine content type
            content_type = "image/jpeg"
            if ext.lower() == ".png":
                content_type = "image/png"
            elif ext.lower() == ".webp":
                content_type = "image/webp"
            
            # Upload file
            blob.upload_from_filename(local_path, content_type=content_type)
            
            gcs_uri = f"gs://{GCS_IMAGES_BUCKET}/{blob_name}"
            print(f"[GCS] Uploaded image to {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            print(f"[GCS] Failed to upload image: {e}")
            return None
    
    def get_signed_url(self, bucket_name: str, blob_name: str, expiration_minutes: int = None) -> Optional[str]:
        """
        Generate a signed URL for temporary access to a file
        
        Args:
            bucket_name: Name of the bucket
            blob_name: Path to the blob in the bucket
            expiration_minutes: URL expiration time in minutes
            
        Returns:
            Signed URL string or None if generation failed
        """
        if not self.is_available():
            return None
        
        try:
            if expiration_minutes is None:
                expiration_minutes = SIGNED_URL_EXPIRATION
            
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="GET"
            )
            
            return url
            
        except Exception as e:
            print(f"[GCS] Failed to generate signed URL: {e}")
            return None
    
    def get_report_signed_url(self, report_id: str, expiration_minutes: int = None) -> Optional[str]:
        """
        Generate a signed URL for a report PDF
        
        Args:
            report_id: The report ID
            expiration_minutes: URL expiration time in minutes
            
        Returns:
            Signed URL string or None
        """
        blob_name = f"reports/{report_id}.pdf"
        return self.get_signed_url(GCS_REPORTS_BUCKET, blob_name, expiration_minutes)
    
    def get_image_signed_url(self, image_id: str, image_type: str = "outputs", expiration_minutes: int = None) -> Optional[str]:
        """
        Generate a signed URL for an image
        
        Args:
            image_id: The image ID
            image_type: "uploads" or "outputs"
            expiration_minutes: URL expiration time in minutes
            
        Returns:
            Signed URL string or None
        """
        blob_name = f"{image_type}/{image_id}.jpg"
        return self.get_signed_url(GCS_IMAGES_BUCKET, blob_name, expiration_minutes)
    
    def get_public_url(self, bucket_name: str, blob_name: str) -> str:
        """
        Get the public URL for a file (requires bucket to be public)
        
        Args:
            bucket_name: Name of the bucket
            blob_name: Path to the blob in the bucket
            
        Returns:
            Public URL string
        """
        return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    
    def get_report_public_url(self, report_id: str) -> str:
        """Get public URL for a report PDF"""
        return self.get_public_url(GCS_REPORTS_BUCKET, f"reports/{report_id}.pdf")
    
    def delete_file(self, bucket_name: str, blob_name: str) -> bool:
        """
        Delete a file from GCS
        
        Args:
            bucket_name: Name of the bucket
            blob_name: Path to the blob in the bucket
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
            print(f"[GCS] Deleted {bucket_name}/{blob_name}")
            return True
            
        except Exception as e:
            print(f"[GCS] Failed to delete file: {e}")
            return False
    
    def file_exists(self, bucket_name: str, blob_name: str) -> bool:
        """
        Check if a file exists in GCS
        
        Args:
            bucket_name: Name of the bucket
            blob_name: Path to the blob in the bucket
            
        Returns:
            True if file exists, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.exists()
            
        except Exception as e:
            print(f"[GCS] Failed to check file existence: {e}")
            return False


# Singleton instance getter
def get_gcs_service() -> GCSService:
    """Get the singleton GCS service instance"""
    return GCSService()


# Quick test
if __name__ == "__main__":
    print("Testing GCS Service...")
    
    gcs = get_gcs_service()
    
    if gcs.is_available():
        print("✓ GCS client initialized successfully")
        print(f"  Project: {GCS_PROJECT_ID}")
        print(f"  Reports Bucket: {GCS_REPORTS_BUCKET}")
        print(f"  Images Bucket: {GCS_IMAGES_BUCKET or 'Not configured'}")
    else:
        print("✗ GCS client not available")
        print("  Check your credentials file path and permissions")
