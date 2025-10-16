import cv2
import os
import numpy as np
from collections import deque, defaultdict
import csv
from datetime import datetime
from object_tracker import ObjectTracker

class VideoProcessor:
    def __init__(self, sensitivity=5, min_duration=1.0, max_duration=None, frame_skip=3):
        """
        Initialize video processor with configurable parameters
        
        Args:
            sensitivity (int): Motion detection sensitivity (1-10)
            min_duration (float): Minimum clip duration in seconds
            max_duration (float): Maximum clip duration in seconds (None = no limit)
            frame_skip (int): Process every Nth frame for performance
        """
        self.sensitivity = sensitivity
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.frame_skip = frame_skip
        
        # Convert sensitivity to contour area threshold
        # Higher sensitivity = lower threshold (detects smaller objects)
        # Lowered threshold to detect small, distant satellites
        self.min_contour_area = max(8, 50 - (sensitivity * 7))
        
    def process_video(self, video_path, progress_callback=None):
        """
        Process video to detect and track individual objects
        
        Args:
            video_path (str): Path to input video file
            progress_callback (callable): Optional callback function(current, total) for progress updates
            
        Returns:
            tuple: (list of clip paths, metadata list)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        
        # Setup MOG2 background subtractor for better motion detection
        # Lower varThreshold to detect small, dim satellites
        backSub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=45,  # Lower threshold - better detection of small satellites
            detectShadows=False
        )
        
        # Initialize object tracker
        # Account for frame skipping when calculating max_disappeared
        # We want 2 seconds of real time, but we only process every frame_skip frames
        # So: max_disappeared = (fps / frame_skip) * desired_seconds
        max_disappeared_frames = int((fps / self.frame_skip) * 2.0)  # 2 seconds of tracked frames
        tracker = ObjectTracker(max_disappeared=max_disappeared_frames, max_distance=150)
        
        # Metadata collection per object
        object_metadata = defaultdict(list)
        video_fps = fps
        
        # Results storage
        clips_folder = "processed_clips"
        os.makedirs(clips_folder, exist_ok=True)
        
        # Maximum contour area to filter out large objects (clouds, etc.)
        max_contour_area = int(frame_width * frame_height * 0.005)
        
        frame_count = 0
        processed_frame_count = 0  # Track output video frame number
        motion_active = False
        clip_writer = None
        clip_filename = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Close video writer if active
                if clip_writer is not None:
                    clip_writer.release()
                break
            
            frame_count += 1
            
            # Frame skipping for performance
            if frame_count % self.frame_skip != 0:
                continue
            
            # Send progress updates every 30 PROCESSED frames to keep connection alive
            # (every 30 * frame_skip actual frames)
            if progress_callback and frame_count % (30 * self.frame_skip) == 0:
                if total_frames > 0:
                    progress_callback(frame_count, total_frames)
                else:
                    # If total_frames unknown, estimate growing total for monotonic progress
                    # Use current_frame * 1.5 so progress gradually increases (66% -> 70% -> 75%...)
                    estimated_total = int(frame_count * 1.5) + 1000  # +1000 to start slower
                    progress_callback(frame_count, estimated_total)
            
            # Motion detection pipeline
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Background subtraction
            fgMask = backSub.apply(gray_blur)
            
            # Morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Detect objects and extract their properties
            current_frame_objects = []
            current_frame_centroids = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (sensitivity-based)
                if self.min_contour_area < area < max_contour_area:
                    # Extract object properties
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        centroid_x = int(M["m10"] / M["m00"])
                        centroid_y = int(M["m01"] / M["m00"])
                    else:
                        centroid_x, centroid_y = x + w//2, y + h//2
                    
                    # Calculate aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Calculate brightness (mean intensity in bounding box)
                    roi = gray[y:y+h, x:x+w]
                    mean_brightness = np.mean(roi) if roi.size > 0 else 0
                    max_brightness = np.max(roi) if roi.size > 0 else 0
                    
                    # Store object data (without clip_id yet)
                    object_data = {
                        'frame_number': frame_count,  # INPUT frame for speed calculations
                        'output_frame_number': processed_frame_count + 1,  # OUTPUT frame for video overlay
                        'area': area,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                        'bbox_x': x,
                        'bbox_y': y,
                        'bbox_width': w,
                        'bbox_height': h,
                        'aspect_ratio': aspect_ratio,
                        'mean_brightness': mean_brightness,
                        'max_brightness': max_brightness,
                        'fps': video_fps
                    }
                    current_frame_objects.append(object_data)
                    current_frame_centroids.append((centroid_x, centroid_y))
            
            # Update tracker with detected centroids
            tracked_objects = tracker.update(current_frame_centroids)
            
            # Assign object_id (as clip_id) to each detection
            if len(tracked_objects) > 0 and len(current_frame_objects) > 0:
                # Match detections to tracked objects by centroid
                for i, (centroid_x, centroid_y) in enumerate(current_frame_centroids):
                    # Find closest tracked object
                    min_dist = float('inf')
                    assigned_id = None
                    
                    for obj_id, (tx, ty) in tracked_objects.items():
                        dist = np.sqrt((centroid_x - tx)**2 + (centroid_y - ty)**2)
                        if dist < min_dist:
                            min_dist = dist
                            assigned_id = obj_id
                    
                    if assigned_id is not None:
                        # Add clip_id and store metadata
                        current_frame_objects[i]['clip_id'] = assigned_id
                        object_metadata[assigned_id].append(current_frame_objects[i])
                        
                        # Draw bounding rectangle with object ID
                        obj = current_frame_objects[i]
                        x, y, w, h = obj['bbox_x'], obj['bbox_y'], obj['bbox_width'], obj['bbox_height']
                        pad = 8
                        cv2.rectangle(frame, 
                                    (max(0, x-pad), max(0, y-pad)),
                                    (min(frame_width-1, x+w+pad), min(frame_height-1, y+h+pad)),
                                    (0, 0, 255), 2)
                        cv2.putText(frame, f"ID:{assigned_id}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Start video writer if we have detections
            if len(tracked_objects) > 0 and not motion_active:
                motion_active = True
                clip_filename = os.path.join(clips_folder, "clip_0001.mp4")
                # Calculate output FPS to achieve exactly 10x speedup
                # Output duration = input duration / 10
                output_fps = 10.0 * fps / self.frame_skip
                clip_writer = cv2.VideoWriter(
                    clip_filename,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    output_fps,
                    (frame_width, frame_height)
                )
            
            # Write frame if writer is active
            if clip_writer is not None:
                clip_writer.write(frame)
                processed_frame_count += 1  # Track output video frame number
        
        cap.release()
        
        # Filter objects by duration (min and max)
        # Account for frame skipping: we only get detections every frame_skip frames
        min_detections = int((fps / self.frame_skip) * self.min_duration)
        max_detections = int((fps / self.frame_skip) * self.max_duration) if self.max_duration else float('inf')
        filtered_metadata = []
        
        for obj_id, detections in object_metadata.items():
            num_detections = len(detections)
            if min_detections <= num_detections <= max_detections:
                filtered_metadata.extend(detections)
        
        # Prepare motion clips list
        motion_clips = [clip_filename] if clip_filename and os.path.exists(clip_filename) else []
        
        return motion_clips, filtered_metadata
