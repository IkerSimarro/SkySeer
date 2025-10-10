import os
import cv2
import zipfile
from io import BytesIO
import pandas as pd
import tempfile
import shutil

def get_video_info(uploaded_file):
    """
    Extract basic information from uploaded video file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        dict: Video information dictionary
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            os.unlink(tmp_path)
            return {"error": "Could not read video file"}
        
        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        duration_seconds = frame_count / fps if fps > 0 else 0
        duration_formatted = format_duration(duration_seconds)
        
        # Get file size
        file_size_bytes = len(uploaded_file.getbuffer())
        file_size_formatted = format_file_size(file_size_bytes)
        
        # Calculate bitrate estimate
        bitrate_kbps = (file_size_bytes * 8) / (duration_seconds * 1000) if duration_seconds > 0 else 0
        bitrate_formatted = f"{bitrate_kbps:.1f} kbps" if bitrate_kbps > 0 else "Unknown"
        
        cap.release()
        os.unlink(tmp_path)
        
        return {
            'duration': duration_formatted,
            'duration_seconds': duration_seconds,
            'fps': f"{fps:.1f}" if fps > 0 else "Unknown",
            'fps_numeric': fps,
            'resolution': f"{width}x{height}",
            'file_size': file_size_formatted,
            'format': uploaded_file.name.split('.')[-1].upper(),
            'bitrate': bitrate_formatted,
            'filename': uploaded_file.name
        }
        
    except Exception as e:
        return {"error": f"Error reading video: {str(e)}"}

def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_file_size(bytes_size):
    """Format file size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def create_download_zip(clip_paths, results_df):
    """
    Create a ZIP file containing all classified clips
    
    Args:
        clip_paths (list): List of paths to video clips
        results_df (pd.DataFrame): Results dataframe with classifications
        
    Returns:
        BytesIO: ZIP file buffer
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create folders for each classification
        classifications = results_df['classification'].unique()
        
        # Add clips organized by classification
        for idx, row in results_df.iterrows():
            clip_id = row['clip_id']
            classification = row['classification']
            confidence = row['confidence']
            
            # Find corresponding clip file
            clip_filename = f"clip_{clip_id:04d}.mp4"
            clip_path = None
            
            for path in clip_paths:
                if clip_filename in path:
                    clip_path = path
                    break
            
            if clip_path and os.path.exists(clip_path):
                # Create descriptive filename
                safe_classification = classification.replace('/', '_')
                new_filename = f"{safe_classification}/{clip_filename.replace('.mp4', '')}_{safe_classification}_conf{confidence:.2f}.mp4"
                
                zip_file.write(clip_path, new_filename)
        
        # Add CSV report
        csv_buffer = BytesIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        zip_file.writestr("analysis_report.csv", csv_buffer.getvalue())
        
        # Add README file
        readme_content = """SkySeer AI Pipeline - Analysis Results
=====================================

This ZIP file contains the results of your sky anomaly detection analysis.

Folder Structure:
- ANOMALY_UAP/: Clips flagged as potential unidentified aerial phenomena
- Satellite/: Clips classified as satellites
- Meteor/: Clips classified as meteors  
- Plane/: Clips classified as aircraft
- Junk/: Clips classified as noise/artifacts

Files:
- analysis_report.csv: Detailed analysis data with confidence scores
- Video clips are named with format: clipXXXX_Classification_confX.XX.mp4

Confidence Scores:
- 0.9+ : High confidence
- 0.7-0.9 : Medium-high confidence  
- 0.5-0.7 : Medium confidence
- <0.5 : Low confidence

For questions about this analysis, please refer to the SkySeer documentation.
"""
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer

def cleanup_temp_files():
    """Clean up temporary files and directories"""
    temp_dirs = ['temp_uploads', 'processed_clips', 'results']
    
    for directory in temp_dirs:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
            except Exception as e:
                print(f"Warning: Could not clean up {directory}: {e}")

def validate_video_file(file_path):
    """
    Validate if a file is a readable video
    
    Args:
        file_path (str): Path to video file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "Could not open video file"
        
        # Try to read first frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False, "Could not read video frames"
        
        # Check basic properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        cap.release()
        
        if fps <= 0 or frame_count <= 0:
            return False, "Invalid video properties"
        
        return True, "Video is valid"
        
    except Exception as e:
        return False, f"Error validating video: {str(e)}"

def estimate_processing_time(file_size_mb, duration_seconds):
    """
    Estimate processing time based on file size and duration
    
    Args:
        file_size_mb (float): File size in megabytes
        duration_seconds (float): Video duration in seconds
        
    Returns:
        str: Estimated processing time
    """
    # Simple estimation based on empirical data
    # Processing is roughly 0.1x real-time for motion detection
    # Plus additional time for ML processing
    
    base_processing_time = duration_seconds * 0.1  # Motion detection
    ml_processing_time = min(30, duration_seconds * 0.05)  # ML processing (capped at 30s)
    
    total_time = base_processing_time + ml_processing_time
    
    return format_duration(total_time)
