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

def get_object_time_ranges(metadata):
    """
    Extract time ranges (start/end frames) for each object from metadata
    
    Args:
        metadata (list): List of detection dictionaries with frame_number and clip_id
        
    Returns:
        dict: {clip_id: {'start_frame': int, 'end_frame': int, 'fps': float}}
    """
    from collections import defaultdict
    
    object_frames = defaultdict(list)
    object_fps = {}
    
    # Group frame numbers by clip_id
    for detection in metadata:
        clip_id = detection['clip_id']
        frame_num = detection['frame_number']
        fps = detection.get('fps', 30)
        
        object_frames[clip_id].append(frame_num)
        object_fps[clip_id] = fps
    
    # Calculate start/end frames for each object
    time_ranges = {}
    for clip_id, frames in object_frames.items():
        time_ranges[clip_id] = {
            'start_frame': min(frames),
            'end_frame': max(frames),
            'fps': object_fps[clip_id]
        }
    
    return time_ranges

def extract_video_segment(source_video_path, start_frame, end_frame, output_path):
    """
    Extract a segment from a video file
    
    Args:
        source_video_path (str): Path to source video
        start_frame (int): Starting frame number
        end_frame (int): Ending frame number
        output_path (str): Path for output video
        
    Returns:
        bool: True if successful, False otherwise
    """
    import cv2
    
    try:
        # Open source video
        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1
        
        # Clean up
        cap.release()
        out.release()
        
        return True
        
    except Exception as e:
        print(f"Error extracting segment: {e}")
        return False

def create_download_zip(clip_paths, results_df, classification_filter=None, metadata=None):
    """
    Create a ZIP file containing classified clips
    
    Args:
        clip_paths (list): List of paths to video clips (should be single combined video)
        results_df (pd.DataFrame): Results dataframe with classifications
        classification_filter (str, optional): Only include clips of this classification
        metadata (list, optional): Detection metadata for extracting time ranges
        
    Returns:
        BytesIO: ZIP file buffer
    """
    import tempfile
    
    zip_buffer = BytesIO()
    
    # Filter results if classification specified
    if classification_filter:
        results_df = results_df[results_df['classification'] == classification_filter]
    
    # Check if we have the combined video and metadata for segment extraction
    has_combined_video = clip_paths and len(clip_paths) == 1 and os.path.exists(clip_paths[0])
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if has_combined_video and metadata:
            # Extract time ranges for each object
            time_ranges = get_object_time_ranges(metadata)
            source_video = clip_paths[0]
            
            # Create temp directory for extracted segments
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract and add segments for each object
                for idx, row in results_df.iterrows():
                    clip_id = row['clip_id']
                    classification = row['classification']
                    confidence = row['confidence']
                    
                    # Get time range for this object
                    if clip_id in time_ranges:
                        time_range = time_ranges[clip_id]
                        start_frame = time_range['start_frame']
                        end_frame = time_range['end_frame']
                        
                        # Create temp output path
                        temp_output = os.path.join(temp_dir, f"segment_{clip_id}.mp4")
                        
                        # Extract segment
                        if extract_video_segment(source_video, start_frame, end_frame, temp_output):
                            # Add to ZIP with descriptive name
                            safe_classification = classification.replace('/', '_')
                            new_filename = f"{safe_classification}/object_{clip_id}_{safe_classification}_conf{confidence:.2f}.mp4"
                            zip_file.write(temp_output, new_filename)
        else:
            # Fallback: use old method if metadata not available
            added_clips = set()
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
                
                if not clip_path and clip_paths:
                    clip_path = clip_paths[0]
                
                if clip_path and os.path.exists(clip_path):
                    if clip_path not in added_clips:
                        safe_classification = classification.replace('/', '_')
                        new_filename = f"{safe_classification}/object_{clip_id}_{safe_classification}_conf{confidence:.2f}.mp4"
                        zip_file.write(clip_path, new_filename)
                        added_clips.add(clip_path)
        
        # Add CSV report
        csv_buffer = BytesIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        zip_file.writestr("analysis_report.csv", csv_buffer.getvalue())
        
        # Generate detection summary
        summary_lines = ["SkySeer AI - Detection Summary", "=" * 50, ""]
        
        # Count detections by classification
        classification_counts = results_df['classification'].value_counts().to_dict()
        summary_lines.append("DETECTION COUNTS:")
        for classification, count in sorted(classification_counts.items()):
            summary_lines.append(f"  {classification}: {count}")
        
        summary_lines.append("")
        summary_lines.append("DETAILED DETECTIONS:")
        summary_lines.append("-" * 50)
        
        # Add details for each detection
        for idx, row in results_df.iterrows():
            clip_id = row['clip_id']
            classification = row['classification']
            confidence = row['confidence']
            duration = row.get('duration', 0)
            avg_speed = row.get('avg_speed', 0)
            
            summary_lines.append(f"\nObject ID: {clip_id}")
            summary_lines.append(f"  Classification: {classification}")
            summary_lines.append(f"  Confidence: {confidence:.1%}")
            summary_lines.append(f"  Duration: {duration:.2f}s")
            summary_lines.append(f"  Average Speed: {avg_speed:.1f} px/frame")
        
        summary_lines.append("")
        summary_lines.append("=" * 50)
        summary_lines.append("Analysis complete. See analysis_report.csv for full details.")
        
        summary_content = "\n".join(summary_lines)
        zip_file.writestr("SUMMARY.txt", summary_content)
        
        # Add README file
        readme_content = """SkySeer AI Pipeline - Analysis Results
=====================================

This ZIP file contains the results of your sky object detection analysis.

Files:
- SUMMARY.txt: Quick overview of all detections
- analysis_report.csv: Detailed analysis data with confidence scores
- Video clips are organized by classification in separate folders

Folder Structure:
- Satellite/: Clips classified as satellites
- Meteor/: Clips classified as meteors  
- Plane/: Clips classified as aircraft
- Junk/: Clips classified as noise/artifacts

Video Clip Naming:
- Format: object_ID_Classification_confX.XX.mp4
- ID: Unique object identifier
- Classification: Detected object type
- conf: Confidence score (0.0-1.0)

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

def add_colored_rectangles_to_clips(clip_paths, metadata, results_df, progress_callback=None):
    """
    Add color-coded rectangles to video clips based on classification
    
    Args:
        clip_paths (list): List of paths to video clips
        metadata (list): Detection metadata with bbox information
        results_df (pd.DataFrame): Results with classifications
        progress_callback (callable): Optional callback(current, total) for progress updates
        
    Returns:
        list: Updated clip paths
    """
    # Color mapping for classifications
    color_map = {
        'Satellite': (0, 255, 0),      # Green
        'Meteor': (0, 0, 255),          # Red
        'Plane': (255, 0, 255),         # Magenta
        'Star': (0, 255, 255),          # Yellow
        'Junk': (128, 128, 128)         # Gray
    }
    
    # Create classification lookup by clip_id
    classification_lookup = {}
    for idx, row in results_df.iterrows():
        classification_lookup[row['clip_id']] = row['classification']
    
    # Group metadata by clip_id
    from collections import defaultdict
    clip_metadata = defaultdict(list)
    for item in metadata:
        clip_metadata[item['clip_id']].append(item)
    
    updated_clips = []
    total_clips = len(clip_paths)
    
    for clip_index, clip_path in enumerate(clip_paths):
        # Extract clip_id from filename
        filename = os.path.basename(clip_path)
        try:
            clip_id = int(filename.split('_')[1].split('.')[0])
        except:
            updated_clips.append(clip_path)
            continue
        
        # Get classification for this clip
        classification = classification_lookup.get(clip_id, 'Junk')
        color = color_map.get(classification, (128, 128, 128))
        
        # Open original clip
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            updated_clips.append(clip_path)
            continue
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create temporary output file
        temp_path = clip_path.replace('.mp4', '_colored.mp4')
        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        
        # Get bbox data for this clip
        bbox_data = clip_metadata.get(clip_id, [])
        bbox_by_frame = {item['frame_number']: item for item in bbox_data}
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Draw colored rectangle if bbox data exists for this frame
            if frame_num in bbox_by_frame:
                item = bbox_by_frame[frame_num]
                x = item['bbox_x']
                y = item['bbox_y']
                w = item['bbox_width']
                h = item['bbox_height']
                
                pad = 8
                cv2.rectangle(frame,
                            (max(0, x-pad), max(0, y-pad)),
                            (min(frame_width-1, x+w+pad), min(frame_height-1, y+h+pad)),
                            color, 2)
                
                # Add classification label
                label = f"{classification}:{clip_id}"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            writer.write(frame)
        
        cap.release()
        writer.release()
        
        # Replace original with colored version
        if os.path.exists(temp_path):
            os.remove(clip_path)
            os.rename(temp_path, clip_path)
        
        updated_clips.append(clip_path)
        
        # Send progress update after each clip to keep connection alive
        if progress_callback:
            progress_callback(clip_index + 1, total_clips)
    
    return updated_clips

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
