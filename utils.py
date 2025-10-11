import os
import cv2
import zipfile
from io import BytesIO
import pandas as pd
import numpy as np
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

def create_download_zip(clip_paths, results_df, classification_filter=None):
    """
    Create a ZIP file containing classified clips
    
    Args:
        clip_paths (list): List of paths to video clips
        results_df (pd.DataFrame): Results dataframe with classifications
        classification_filter (str, optional): Only include clips of this classification
        
    Returns:
        BytesIO: ZIP file buffer
    """
    zip_buffer = BytesIO()
    
    # Filter results if classification specified
    if classification_filter:
        results_df = results_df[results_df['classification'] == classification_filter]
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Track which clip files we've already added to avoid duplicates
        added_clips = set()
        
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
            
            # If specific clip not found, use the first available clip
            # (with object tracking, all objects are in one video)
            if not clip_path and clip_paths:
                clip_path = clip_paths[0]
            
            if clip_path and os.path.exists(clip_path):
                # Only add each unique clip file once
                if clip_path not in added_clips:
                    # Create descriptive filename with object ID
                    safe_classification = classification.replace('/', '_')
                    new_filename = f"{safe_classification}/object_{clip_id}_{safe_classification}_conf{confidence:.2f}.mp4"
                    
                    # Add the clip
                    zip_file.write(clip_path, new_filename)
                    added_clips.add(clip_path)
        
        # Add CSV report
        csv_buffer = BytesIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        zip_file.writestr("analysis_report.csv", csv_buffer.getvalue())
        
        # Add README file
        readme_content = """SkySeer AI Pipeline - Analysis Results
=====================================

This ZIP file contains the results of your sky object detection analysis.

Folder Structure:
- Satellite/: Clips classified as satellites
- Meteor/: Clips classified as meteors  
- Plane/: Clips classified as aircraft
- Junk/: Clips classified as noise/artifacts
- Star/: Clips classified as stars (if detected)

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

def analyze_video_content(uploaded_file, video_info):
    """
    Analyze video content by sampling frames to detect brightness, noise, and motion characteristics
    
    Args:
        uploaded_file: Streamlit uploaded file object
        video_info (dict): Basic video info from get_video_info()
        
    Returns:
        dict: Content analysis results
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            os.unlink(tmp_path)
            return {'error': True}
        
        fps = video_info.get('fps_numeric', 30)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Sample 10 frames evenly distributed across the video
        sample_indices = [int(i * frame_count / 10) for i in range(10)] if frame_count > 10 else [0]
        
        brightness_values = []
        contrast_values = []
        noise_levels = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness: average pixel intensity
            brightness = np.mean(gray)
            brightness_values.append(brightness)
            
            # Contrast: standard deviation of pixel intensities
            contrast = np.std(gray)
            contrast_values.append(contrast)
            
            # Noise level: apply Laplacian to detect high-frequency noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise = np.var(laplacian)
            noise_levels.append(noise)
        
        cap.release()
        os.unlink(tmp_path)
        
        return {
            'avg_brightness': np.mean(brightness_values) if brightness_values else 0,
            'avg_contrast': np.mean(contrast_values) if contrast_values else 0,
            'avg_noise': np.mean(noise_levels) if noise_levels else 0,
            'brightness_std': np.std(brightness_values) if brightness_values else 0,
            'error': False
        }
        
    except Exception as e:
        return {'error': True, 'message': str(e)}

def recommend_settings(video_info, uploaded_file=None):
    """
    Analyze video and recommend optimal settings for detection
    
    Args:
        video_info (dict): Video information from get_video_info()
        uploaded_file: Streamlit uploaded file object (optional, for deep analysis)
        
    Returns:
        dict: Recommended settings with explanations
    """
    recommendations = {
        'sensitivity': 4,
        'min_duration': 1.5,
        'max_duration': 15.0,
        'max_duration_enabled': True,  # Always enabled now
        'frame_skip': 3,
        'explanations': []
    }
    
    duration_seconds = video_info.get('duration_seconds', 0)
    fps_numeric = video_info.get('fps_numeric', 30)
    resolution = video_info.get('resolution', '1920x1080')
    
    # Parse resolution
    try:
        width, height = map(int, resolution.split('x'))
        total_pixels = width * height
    except:
        total_pixels = 1920 * 1080
    
    # Deep content analysis if uploaded file provided
    content_analysis = None
    if uploaded_file:
        content_analysis = analyze_video_content(uploaded_file, video_info)
    
    # Recommendation 1: Frame skip based on duration and FPS
    if duration_seconds > 600:  # > 10 minutes
        recommendations['frame_skip'] = 6
        recommendations['explanations'].append(
            "📹 Very long video (>10min) - using frame skip=6 for faster processing"
        )
    elif duration_seconds > 300:  # > 5 minutes
        recommendations['frame_skip'] = 5
        recommendations['explanations'].append(
            "📹 Long video (>5min) - using frame skip=5 for efficient processing"
        )
    elif duration_seconds > 180:  # > 3 minutes
        recommendations['frame_skip'] = 4
        recommendations['explanations'].append(
            "📹 Medium video (>3min) - using frame skip=4 for balanced speed/accuracy"
        )
    elif fps_numeric >= 60:
        recommendations['frame_skip'] = 4
        recommendations['explanations'].append(
            "📹 High FPS video (60+) - using frame skip=4 to handle extra frames"
        )
    else:
        recommendations['explanations'].append(
            "📹 Short/standard video - using frame skip=3 for good accuracy"
        )
    
    # Recommendation 2: Sensitivity based on resolution and brightness
    if content_analysis and not content_analysis.get('error'):
        brightness = content_analysis['avg_brightness']
        noise = content_analysis['avg_noise']
        
        # Very dark video (night sky)
        if brightness < 30:
            if noise > 100:
                recommendations['sensitivity'] = 3
                recommendations['explanations'].append(
                    "🌑 Very dark & noisy video - using low sensitivity (3) to reduce false positives"
                )
            else:
                recommendations['sensitivity'] = 5
                recommendations['explanations'].append(
                    "🌑 Very dark but clean video - using moderate sensitivity (5)"
                )
        # Dark video (twilight/dusk)
        elif brightness < 80:
            recommendations['sensitivity'] = 4
            recommendations['explanations'].append(
                "🌆 Dark video detected - using sensitivity=4 for twilight conditions"
            )
        # Brighter video
        else:
            recommendations['sensitivity'] = 3
            recommendations['explanations'].append(
                "☀️ Relatively bright video - lowering sensitivity to avoid false detections"
            )
    elif total_pixels > 2073600:  # Fallback to resolution-based
        recommendations['sensitivity'] = 3
        recommendations['explanations'].append(
            "🎯 High resolution video (>1080p) - using lower sensitivity to reduce noise"
        )
    else:
        recommendations['explanations'].append(
            "🎯 Standard video - using moderate sensitivity (4)"
        )
    
    # Recommendation 3: Duration settings based on FPS and content
    if fps_numeric > 0:
        if fps_numeric >= 60:
            recommendations['min_duration'] = 1.0
            recommendations['explanations'].append(
                "⏱️ High FPS (60+) - can detect shorter events (min 1.0s)"
            )
        elif fps_numeric >= 30:
            recommendations['min_duration'] = 1.5
            recommendations['explanations'].append(
                "⏱️ Standard FPS (30+) - using conservative min duration (1.5s)"
            )
        else:
            recommendations['min_duration'] = 2.0
            recommendations['explanations'].append(
                "⏱️ Low FPS (<30) - using longer min duration (2.0s) for reliability"
            )
    
    # Recommendation 4: Max duration based on video length
    if duration_seconds > 600:  # Long videos might have slower satellites
        recommendations['max_duration'] = 25.0
        recommendations['explanations'].append(
            "⭐ Long video - max duration 25s (allows slower-moving satellites)"
        )
    else:
        recommendations['max_duration'] = 15.0
        recommendations['explanations'].append(
            "⭐ Max duration 15s to filter stationary stars"
        )
    
    # Recommendation 5: Noise-specific advice
    if content_analysis and not content_analysis.get('error'):
        noise = content_analysis['avg_noise']
        if noise > 150:
            recommendations['explanations'].append(
                "⚠️ High noise detected - consider increasing min duration to 2.0s"
            )
    
    # General advice
    recommendations['explanations'].append(
        "💡 Settings optimized for <10 obvious detections per video"
    )
    
    return recommendations
