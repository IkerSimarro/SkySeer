import cv2
import os
import numpy as np
from collections import deque
import csv
from datetime import datetime

class VideoProcessor:
    def __init__(self, sensitivity=5, min_duration=1.0, frame_skip=3):
        """
        Initialize video processor with configurable parameters
        
        Args:
            sensitivity (int): Motion detection sensitivity (1-10)
            min_duration (float): Minimum clip duration in seconds
            frame_skip (int): Process every Nth frame for performance
        """
        self.sensitivity = sensitivity
        self.min_duration = min_duration
        self.frame_skip = frame_skip
        
        # Convert sensitivity to contour area threshold
        # Higher sensitivity = lower threshold (detects smaller objects)
        self.min_contour_area = max(1, 15 - sensitivity)
        
    def process_video(self, video_path):
        """
        Process video to detect and extract motion clips
        
        Args:
            video_path (str): Path to input video file
            
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
        
        # Setup MOG2 background subtractor for better motion detection
        backSub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=32,  # Adjusted based on sensitivity
            detectShadows=False
        )
        
        # Clip management variables
        min_frames_for_clip = int(fps * self.min_duration)
        pre_buffer_frames = fps // 2  # 0.5 seconds before motion
        post_buffer_frames = fps // 2  # 0.5 seconds after motion
        
        pre_motion_buffer = deque(maxlen=pre_buffer_frames)
        post_motion_buffer = deque()
        
        frame_count = 0
        clip_index = 0
        motion_active = False
        clip_writer = None
        clip_frames_count = 0
        post_motion_count = 0
        
        # Metadata collection
        metadata = []
        current_clip_metadata = []
        video_fps = fps  # Store FPS for feature extraction
        
        # Results storage
        motion_clips = []
        clips_folder = "processed_clips"
        os.makedirs(clips_folder, exist_ok=True)
        
        # Maximum contour area to filter out large objects (clouds, etc.)
        max_contour_area = int(frame_width * frame_height * 0.005)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Handle final clip cleanup
                if motion_active and clip_writer is not None:
                    if clip_frames_count >= min_frames_for_clip:
                        # Write remaining post-motion frames
                        for f in post_motion_buffer:
                            clip_writer.write(f)
                        clip_writer.release()
                        
                        # Save metadata for this clip
                        for meta in current_clip_metadata:
                            meta['clip_id'] = clip_index
                        metadata.extend(current_clip_metadata)
                        
                    else:
                        # Clip too short, discard
                        clip_writer.release()
                        clip_filename = os.path.join(clips_folder, f"clip_{clip_index:04d}.mp4")
                        if os.path.exists(clip_filename):
                            os.remove(clip_filename)
                break
            
            frame_count += 1
            
            # Frame skipping for performance
            if frame_count % self.frame_skip != 0:
                pre_motion_buffer.append(frame.copy())
                continue
            
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
            
            motion_detected = False
            frame_objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (sensitivity-based)
                if self.min_contour_area < area < max_contour_area:
                    motion_detected = True
                    
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
                    
                    # Store object data
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
                        'fps': video_fps
                    }
                    frame_objects.append(object_data)
                    
                    # Draw bounding rectangle on frame
                    pad = 8
                    cv2.rectangle(frame, 
                                (max(0, x-pad), max(0, y-pad)),
                                (min(frame_width-1, x+w+pad), min(frame_height-1, y+h+pad)),
                                (0, 0, 255), 2)
            
            # Add frame to pre-motion buffer
            pre_motion_buffer.append(frame.copy())
            
            # Handle clip recording
            if motion_detected:
                # Add frame objects to current clip metadata
                current_clip_metadata.extend(frame_objects)
                clip_frames_count += 1
                
                if not motion_active:
                    # Start new clip
                    clip_index += 1
                    clip_filename = os.path.join(clips_folder, f"clip_{clip_index:04d}.mp4")
                    
                    clip_writer = cv2.VideoWriter(
                        clip_filename,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (frame_width, frame_height)
                    )
                    
                    motion_active = True
                    motion_clips.append(clip_filename)
                    
                    # Write pre-motion buffer
                    for f in pre_motion_buffer:
                        clip_writer.write(f)
                
                # Write current frame
                clip_writer.write(frame)
                post_motion_count = 0
                post_motion_buffer.clear()
                
            else:
                # No motion in current frame
                if motion_active:
                    post_motion_buffer.append(frame.copy())
                    post_motion_count += 1
                    
                    # Check if we should end the clip
                    if post_motion_count >= post_buffer_frames:
                        if clip_frames_count >= min_frames_for_clip:
                            # Valid clip - write post-motion frames and close
                            for f in post_motion_buffer:
                                clip_writer.write(f)
                            clip_writer.release()
                            
                            # Save metadata for this clip
                            for meta in current_clip_metadata:
                                meta['clip_id'] = clip_index
                            metadata.extend(current_clip_metadata)
                            
                        else:
                            # Clip too short - discard
                            clip_writer.release()
                            if os.path.exists(clip_filename):
                                os.remove(clip_filename)
                            motion_clips.pop()  # Remove from clips list
                        
                        # Reset for next clip
                        clip_writer = None
                        motion_active = False
                        clip_frames_count = 0
                        post_motion_buffer.clear()
                        current_clip_metadata = []
        
        cap.release()
        
        return motion_clips, metadata
