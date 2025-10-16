import pandas as pd
import numpy as np
from collections import defaultdict
import math
from star_detector import StarDetector

class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor for motion analysis"""
        self.star_detector = StarDetector()
    
    def extract_features(self, metadata):
        """
        Extract movement features from motion detection metadata
        
        Args:
            metadata (list): List of detection metadata dictionaries
            
        Returns:
            pd.DataFrame: DataFrame with extracted features for each clip
        """
        if not metadata:
            return pd.DataFrame()
        
        # Detect star groups first
        star_clip_ids = self.star_detector.detect_star_groups(metadata)
        
        # Group metadata by clip_id
        clips_data = defaultdict(list)
        for item in metadata:
            clips_data[item['clip_id']].append(item)
        
        features_list = []
        
        for clip_id, clip_detections in clips_data.items():
            if not clip_detections:
                continue
            
            # Sort detections by frame number
            clip_detections.sort(key=lambda x: x['frame_number'])
            
            # Get FPS from first detection (all should have same FPS)
            video_fps = clip_detections[0].get('fps', 30) if clip_detections else 30
            
            # Extract basic trajectory features
            features = self._extract_clip_features(clip_id, clip_detections, video_fps)
            
            # Add star detection feature
            features['is_star_group'] = 1 if clip_id in star_clip_ids else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _extract_clip_features(self, clip_id, detections, fps=30):
        """
        Extract comprehensive features for a single clip
        
        Args:
            clip_id (int): Unique identifier for the clip
            detections (list): List of detection dictionaries for this clip
            fps (float): Video frame rate for accurate speed/duration calculations
            
        Returns:
            dict: Dictionary of extracted features
        """
        if len(detections) < 2:
            return self._create_minimal_features(clip_id, detections, fps)
        
        # Extract position data
        positions = [(d['centroid_x'], d['centroid_y']) for d in detections]
        frame_numbers = [d['frame_number'] for d in detections]
        areas = [d['area'] for d in detections]
        aspect_ratios = [d['aspect_ratio'] for d in detections]
        
        # Extract brightness data
        mean_brightness_values = [d.get('mean_brightness', 0) for d in detections]
        max_brightness_values = [d.get('max_brightness', 0) for d in detections]
        
        # Calculate speeds between consecutive positions
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = math.sqrt(dx*dx + dy*dy)
            frame_diff = frame_numbers[i] - frame_numbers[i-1]
            speed = distance / max(frame_diff, 1)  # pixels per frame
            speeds.append(speed)
        
        # Calculate trajectory features
        avg_speed = np.mean(speeds) if speeds else 0
        speed_std = np.std(speeds) if len(speeds) > 1 else 0
        max_speed = max(speeds) if speeds else 0
        min_speed = min(speeds) if speeds else 0
        
        # Speed consistency (lower std = more consistent)
        speed_consistency = 1.0 / (1.0 + speed_std) if speed_std > 0 else 1.0
        
        # Calculate total path length and duration
        total_distance = sum([math.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                      (positions[i][1] - positions[i-1][1])**2) 
                            for i in range(1, len(positions))])
        
        duration = (frame_numbers[-1] - frame_numbers[0]) / float(fps)  # Use actual FPS
        
        # Trajectory linearity (straight line vs actual path)
        if len(positions) >= 2:
            start_pos = positions[0]
            end_pos = positions[-1]
            straight_line_distance = math.sqrt((end_pos[0] - start_pos[0])**2 + 
                                             (end_pos[1] - start_pos[1])**2)
            linearity = straight_line_distance / max(total_distance, 1)
        else:
            linearity = 1.0
        
        # Direction changes (indicator of erratic movement)
        direction_changes = 0
        if len(positions) >= 3:
            for i in range(1, len(positions) - 1):
                # Calculate vectors
                v1 = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
                v2 = (positions[i+1][0] - positions[i][0], positions[i+1][1] - positions[i][1])
                
                # Calculate angle between vectors
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                    angle = math.acos(cos_angle)
                    
                    # Count significant direction changes (> 30 degrees)
                    if angle > math.pi / 6:
                        direction_changes += 1
        
        # Object size statistics
        avg_area = np.mean(areas)
        area_std = np.std(areas) if len(areas) > 1 else 0
        max_area = max(areas)
        min_area = min(areas)
        
        # Size consistency
        size_consistency = 1.0 / (1.0 + area_std / max(avg_area, 1))
        
        # Aspect ratio statistics
        avg_aspect_ratio = np.mean(aspect_ratios)
        aspect_ratio_std = np.std(aspect_ratios) if len(aspect_ratios) > 1 else 0
        
        # Acceleration analysis
        accelerations = []
        if len(speeds) > 1:
            for i in range(1, len(speeds)):
                accel = abs(speeds[i] - speeds[i-1])
                accelerations.append(accel)
        
        avg_acceleration = np.mean(accelerations) if accelerations else 0
        max_acceleration = max(accelerations) if accelerations else 0
        
        # Brightness statistics (for distinguishing planes from satellites)
        avg_brightness = np.mean(mean_brightness_values) if mean_brightness_values else 0
        brightness_std = np.std(mean_brightness_values) if len(mean_brightness_values) > 1 else 0
        max_brightness = max(max_brightness_values) if max_brightness_values else 0
        
        # Brightness consistency (low variance = constant like satellite, high variance = blinking like plane)
        brightness_consistency = 1.0 / (1.0 + brightness_std / max(avg_brightness, 1))
        
        # Enhanced blinking pattern detection for planes
        brightness_variation_coeff = (brightness_std / max(avg_brightness, 1)) if avg_brightness > 0 else 0
        
        # Detect periodic blinking (on/off pattern typical of plane lights)
        blinking_pattern_score = 0.0
        if len(mean_brightness_values) >= 3 and avg_brightness > 1.0:  # Safeguard against zero brightness
            # Calculate brightness differences between consecutive frames
            brightness_diffs = [abs(mean_brightness_values[i] - mean_brightness_values[i-1]) 
                               for i in range(1, len(mean_brightness_values))]
            
            # High differences indicate blinking
            avg_diff = np.mean(brightness_diffs) if brightness_diffs else 0
            if avg_diff > avg_brightness * 0.15:  # 15% change threshold
                blinking_pattern_score = min(avg_diff / avg_brightness, 1.0)
        
        # Combine blinking indicators
        blinking_score = max(brightness_variation_coeff, blinking_pattern_score)
        
        # Movement pattern classification hints  
        # Use duration as PRIMARY discriminator between satellites and planes
        
        # Satellite: consistent, linear, HIGH score for 3-15s duration (typical satellite pass)
        # Score peaks at 3-15s, but maintain decent score for longer passes
        if duration < 3:
            duration_satellite_factor = 0.5
        elif duration < 15:
            duration_satellite_factor = 1.4  # Strong boost for typical satellite range (3-15s)
        elif duration < 25:
            duration_satellite_factor = 1.0  # Still good for longer satellites
        else:
            duration_satellite_factor = 0.8  # Gentle decline for very long tracks
        
        # CRITICAL: Satellites must have minimum speed (typically 1-25 px/frame)
        # Use gentle penalties to filter slow objects without blocking real satellites
        if avg_speed < 0.3:
            speed_penalty = 0.2  # Heavy penalty for extremely slow objects
        elif avg_speed < 0.6:
            speed_penalty = 0.6  # Moderate penalty for very slow objects
        elif avg_speed > 35:
            speed_penalty = 0.6  # Penalize very fast objects (likely meteors)
        else:
            speed_penalty = 1.0  # Normal satellite speed range (0.6-35 px/frame)
        
        satellite_consistency = size_consistency * brightness_consistency
        satellite_score = speed_consistency * linearity * satellite_consistency * duration_satellite_factor * speed_penalty
        
        # Meteor: VERY high speed, extremely linear, brief duration, bright streaks
        # Meteors are very fast streaks (>15 px/frame), ultra-linear, brief (<5s)
        # They travel small patches of sky and often leave bright trails
        if avg_speed > 15:
            speed_factor = min(avg_speed / 10.0, 5.0)  # Strong boost for fast objects, cap at 5x
        elif avg_speed > 8:
            speed_factor = avg_speed / 15.0  # Moderate score for medium-fast
        else:
            speed_factor = 0.2  # Low score for slow objects
        
        # Meteors are brief - favor <5s duration, heavily penalize longer
        if duration < 2:
            duration_factor = 2.0  # Strong boost for very brief
        elif duration < 5:
            duration_factor = 1.0  # Normal for brief
        else:
            duration_factor = 0.2  # Heavy penalty for long duration
        
        # Brightness bonus for bright streaks
        brightness_factor = 1.0 + min(max_brightness / 200.0, 1.0)  # Up to 2x for bright objects
        
        # Linearity is critical for meteors (straight streaks)
        linearity_factor = linearity ** 2  # Square to heavily favor linear paths
        
        meteor_score = speed_factor * linearity_factor * duration_factor * brightness_factor
        
        # Anomaly indicators: erratic movement, inconsistent speed/size
        anomaly_indicators = (
            direction_changes / max(len(positions), 1) +  # Direction change rate
            (1.0 - speed_consistency) +  # Speed inconsistency
            (1.0 - size_consistency) +   # Size inconsistency
            max_acceleration / max(avg_speed, 1)  # High acceleration relative to speed
        )
        
        return {
            'clip_id': clip_id,
            'duration': duration,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'speed_consistency': speed_consistency,
            'total_distance': total_distance,
            'linearity': linearity,
            'direction_changes': direction_changes,
            'avg_area': avg_area,
            'max_area': max_area,
            'min_area': min_area,
            'size_consistency': size_consistency,
            'avg_aspect_ratio': avg_aspect_ratio,
            'aspect_ratio_std': aspect_ratio_std,
            'avg_acceleration': avg_acceleration,
            'max_acceleration': max_acceleration,
            'avg_brightness': avg_brightness,
            'brightness_std': brightness_std,
            'max_brightness': max_brightness,
            'brightness_consistency': brightness_consistency,
            'brightness_variation_coeff': brightness_variation_coeff,
            'blinking_score': blinking_score,  # Enhanced blinking detection
            'satellite_score': satellite_score,
            'meteor_score': meteor_score,
            'anomaly_indicators': anomaly_indicators,
            'detection_count': len(detections)
        }
    
    def _create_minimal_features(self, clip_id, detections, fps=30):
        """Create minimal feature set for clips with insufficient data"""
        if not detections:
            return {
                'clip_id': clip_id,
                'duration': 0,
                'avg_speed': 0,
                'max_speed': 0,
                'min_speed': 0,
                'speed_consistency': 0,
                'total_distance': 0,
                'linearity': 0,
                'direction_changes': 0,
                'avg_area': 0,
                'max_area': 0,
                'min_area': 0,
                'size_consistency': 0,
                'avg_aspect_ratio': 1,
                'aspect_ratio_std': 0,
                'avg_acceleration': 0,
                'max_acceleration': 0,
                'avg_brightness': 0,
                'brightness_std': 0,
                'max_brightness': 0,
                'brightness_consistency': 0,
                'brightness_variation_coeff': 0,
                'blinking_score': 0,  # No blinking for insufficient data
                'satellite_score': 0,
                'meteor_score': 0,
                'anomaly_indicators': 1,  # High anomaly score for insufficient data
                'detection_count': len(detections)
            }
        
        # Single detection case
        detection = detections[0]
        return {
            'clip_id': clip_id,
            'duration': 1.0 / float(fps),  # Single frame duration using actual FPS
            'avg_speed': 0,
            'max_speed': 0,
            'min_speed': 0,
            'speed_consistency': 0,
            'total_distance': 0,
            'linearity': 0,
            'direction_changes': 0,
            'avg_area': detection['area'],
            'max_area': detection['area'],
            'min_area': detection['area'],
            'size_consistency': 1,
            'avg_aspect_ratio': detection['aspect_ratio'],
            'aspect_ratio_std': 0,
            'avg_acceleration': 0,
            'max_acceleration': 0,
            'avg_brightness': detection.get('mean_brightness', 0),
            'brightness_std': 0,
            'max_brightness': detection.get('max_brightness', 0),
            'brightness_consistency': 1,
            'brightness_variation_coeff': 0,
            'blinking_score': 0,  # No blinking for single detection
            'satellite_score': 0,
            'meteor_score': 0,
            'anomaly_indicators': 0.5,  # Medium anomaly score for single detection
            'detection_count': 1
        }
