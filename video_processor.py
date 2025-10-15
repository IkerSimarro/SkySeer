import cv2
import os
import numpy as np
from collections import deque, defaultdict
import csv
from datetime import datetime
from object_tracker import ObjectTracker
import gc

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
        # Further optimized for small satellites and faint meteors
        self.min_contour_area = max(3, 60 - (sensitivity * 7))
        
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
        # Lower varThreshold to detect small/faint satellites and meteors
        backSub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=45,  # Lower threshold - detects smaller/fainter objects
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
        
        # Store frames for each object with trimming buffer (0.5 second = fps * 0.5 frames)
        # MEMORY OPTIMIZATION: Reduced from 1 second to 0.5 seconds to support Full HD videos
        # Clamp to minimum of 1 frame to handle very low FPS videos (≤2 FPS)
        buffer_frames = max(1, int(fps * 0.5))  # 0.5 second buffer before/after detection, minimum 1 frame
        object_frames = defaultdict(lambda: {'frames': deque(), 'frame_numbers': deque()})
        all_frames_buffer = deque(maxlen=buffer_frames * 2)  # Circular buffer for pre-detection frames
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Store frame in circular buffer (for pre-detection context)
            all_frames_buffer.append((frame_count, frame.copy()))
            
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
                        'frame_number': frame_count,
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
            
            # Assign object_id (as clip_id) to each detection and collect frames
            active_object_ids = set()
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
                        active_object_ids.add(assigned_id)
                        
                        # Note: Rectangles will be added later with color-coding based on classification
            
            # Store frames for active objects
            for obj_id in active_object_ids:
                # If this is first detection, add buffer frames before
                if obj_id not in object_frames or len(object_frames[obj_id]['frames']) == 0:
                    # Add frames from circular buffer (0.5 second before detection)
                    for buff_frame_num, buff_frame in all_frames_buffer:
                        if buff_frame_num >= frame_count - buffer_frames:
                            object_frames[obj_id]['frames'].append(buff_frame.copy())
                            object_frames[obj_id]['frame_numbers'].append(buff_frame_num)
                
                # Add current frame
                object_frames[obj_id]['frames'].append(frame.copy())
                object_frames[obj_id]['frame_numbers'].append(frame_count)
            
            # For objects that were active but are no longer detected, add post-buffer frames
            # Keep adding frames for buffer_frames after last detection
            for obj_id in list(object_frames.keys()):
                if obj_id not in active_object_ids:
                    if len(object_frames[obj_id]['frame_numbers']) > 0:
                        last_frame = object_frames[obj_id]['frame_numbers'][-1]
                        # Add frames for 0.5 second after last detection
                        if frame_count <= last_frame + buffer_frames:
                            object_frames[obj_id]['frames'].append(frame.copy())
                            object_frames[obj_id]['frame_numbers'].append(frame_count)
        
        cap.release()
        
        # Filter objects by duration (min and max)
        # Account for frame skipping: we only get detections every frame_skip frames
        min_detections = int((fps / self.frame_skip) * self.min_duration)
        max_detections = int((fps / self.frame_skip) * self.max_duration) if self.max_duration else float('inf')
        filtered_metadata = []
        valid_object_ids = set()
        
        for obj_id, detections in object_metadata.items():
            num_detections = len(detections)
            if min_detections <= num_detections <= max_detections:
                filtered_metadata.extend(detections)
                valid_object_ids.add(obj_id)
        
        # Create individual trimmed clips for each valid object
        motion_clips = []
        for obj_id in valid_object_ids:
            if obj_id in object_frames and len(object_frames[obj_id]['frames']) > 0:
                clip_filename = os.path.join(clips_folder, f"clip_{obj_id:04d}.mp4")
                
                # Create video writer for this object's clip
                clip_writer = cv2.VideoWriter(
                    clip_filename,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height)
                )
                
                # Write all frames for this object (already trimmed to 1 sec before/after)
                for frame in object_frames[obj_id]['frames']:
                    clip_writer.write(frame)
                
                clip_writer.release()
                motion_clips.append(clip_filename)
                
                # MEMORY OPTIMIZATION: Clear frames from memory after writing to disk
                object_frames[obj_id]['frames'].clear()
        
        # MEMORY OPTIMIZATION: Force garbage collection after processing
        object_frames.clear()
        all_frames_buffer.clear()
        gc.collect()
        
        return motion_clips, filtered_metadata
