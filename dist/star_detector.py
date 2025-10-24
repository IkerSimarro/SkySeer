import numpy as np
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Set

class StarDetector:
    def __init__(self, min_group_size=3, velocity_tolerance=0.2, direction_tolerance=20, 
                 min_temporal_overlap=0.5, max_spatial_distance=500, min_speed=0.01):
        """
        Initialize star detector to identify stationary stars based on group movement
        
        Args:
            min_group_size (int): Minimum number of objects moving together to classify as stars (default: 3)
            velocity_tolerance (float): Maximum relative difference in velocity (0-1, default: 0.2 = 20%)
            direction_tolerance (float): Maximum direction difference in degrees (default: 20)
            min_temporal_overlap (float): Minimum fraction of frames that must overlap (default: 0.5 = 50%)
            max_spatial_distance (float): Maximum average distance between group members in pixels (default: 500)
            min_speed (float): Minimum speed in pixels/frame to consider (default: 0.01, very low to catch stars)
        """
        self.min_group_size = min_group_size
        self.velocity_tolerance = velocity_tolerance
        self.direction_tolerance = direction_tolerance
        self.min_temporal_overlap = min_temporal_overlap
        self.max_spatial_distance = max_spatial_distance
        self.min_speed = min_speed
    
    def detect_star_groups(self, metadata: List[Dict]) -> Set[int]:
        """
        Detect which clip IDs contain stars based on group movement analysis
        
        Args:
            metadata (list): List of detection metadata dictionaries
            
        Returns:
            set: Set of clip IDs that are classified as stars
        """
        if not metadata:
            return set()
        
        # Group metadata by clip_id
        clips_data = defaultdict(list)
        for item in metadata:
            clips_data[item['clip_id']].append(item)
        
        # Build candidate groups by finding mutually overlapping, similarly moving objects
        candidate_groups = []
        processed_clips = set()
        
        for clip_id in sorted(clips_data.keys()):
            if clip_id in processed_clips or len(clips_data[clip_id]) < 2:
                continue
            
            clip_velocity = self._calculate_velocity_vector(clips_data[clip_id])
            if clip_velocity is None:
                continue
            
            clip_frame_range = self._get_frame_range(clips_data[clip_id])
            
            # Build a group starting with this clip
            current_group = [clip_id]
            
            # Find all clips that are similar to the current clip
            for other_clip_id in sorted(clips_data.keys()):
                if other_clip_id == clip_id or other_clip_id in processed_clips or len(clips_data[other_clip_id]) < 2:
                    continue
                
                other_velocity = self._calculate_velocity_vector(clips_data[other_clip_id])
                if other_velocity is None:
                    continue
                
                other_frame_range = self._get_frame_range(clips_data[other_clip_id])
                
                # Check if this clip is compatible with ALL members of the current group
                is_compatible = True
                
                for group_member_id in current_group:
                    member_velocity = self._calculate_velocity_vector(clips_data[group_member_id])
                    member_frame_range = self._get_frame_range(clips_data[group_member_id])
                    
                    # Check temporal overlap with this group member
                    if not self._has_sufficient_temporal_overlap(other_frame_range, member_frame_range):
                        is_compatible = False
                        break
                    
                    # Check velocity similarity with this group member
                    if not self._are_velocities_similar(other_velocity, member_velocity):
                        is_compatible = False
                        break
                    
                    # Check spatial proximity with this group member
                    if not self._are_spatially_close(clips_data[other_clip_id], clips_data[group_member_id]):
                        is_compatible = False
                        break
                
                # If compatible with all group members, add to group
                if is_compatible:
                    current_group.append(other_clip_id)
            
            # Check if group has minimum size and sustained co-movement
            if len(current_group) >= self.min_group_size:
                # Verify sustained co-movement by checking absolute overlap duration
                if self._has_sustained_comovement(current_group, clips_data):
                    candidate_groups.append(current_group)
                    processed_clips.update(current_group)
        
        # Collect all clip IDs from valid star groups
        star_clip_ids = set()
        for group in candidate_groups:
            star_clip_ids.update(group)
        
        return star_clip_ids
    
    def _get_frame_range(self, detections: List[Dict]) -> Tuple[int, int]:
        """
        Get the frame range for a sequence of detections
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            tuple: (min_frame, max_frame)
        """
        frame_numbers = [d['frame_number'] for d in detections]
        return (min(frame_numbers), max(frame_numbers))
    
    def _has_sufficient_temporal_overlap(self, range1: Tuple[int, int], 
                                        range2: Tuple[int, int]) -> bool:
        """
        Check if two frame ranges have sufficient temporal overlap
        
        Args:
            range1 (tuple): First frame range (min, max)
            range2 (tuple): Second frame range (min, max)
            
        Returns:
            bool: True if ranges overlap sufficiently
        """
        min1, max1 = range1
        min2, max2 = range2
        
        # Calculate overlap
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        if overlap_start > overlap_end:
            return False
        
        overlap_duration = overlap_end - overlap_start
        
        # Calculate minimum required overlap based on shortest range
        range1_duration = max1 - min1
        range2_duration = max2 - min2
        min_duration = min(range1_duration, range2_duration)
        
        # Check if overlap is sufficient
        if min_duration == 0:
            return False
        
        overlap_ratio = overlap_duration / min_duration
        return overlap_ratio >= self.min_temporal_overlap
    
    def _are_spatially_close(self, detections1: List[Dict], 
                            detections2: List[Dict]) -> bool:
        """
        Check if two sets of detections are spatially close (in the same region)
        
        Args:
            detections1 (list): First set of detections
            detections2 (list): Second set of detections
            
        Returns:
            bool: True if detections are spatially close
        """
        # Calculate average positions
        avg_x1 = np.mean([d['centroid_x'] for d in detections1])
        avg_y1 = np.mean([d['centroid_y'] for d in detections1])
        
        avg_x2 = np.mean([d['centroid_x'] for d in detections2])
        avg_y2 = np.mean([d['centroid_y'] for d in detections2])
        
        # Calculate distance between average positions
        distance = math.sqrt((avg_x2 - avg_x1)**2 + (avg_y2 - avg_y1)**2)
        
        return distance <= self.max_spatial_distance
    
    def _has_sustained_comovement(self, group: List[int], clips_data: Dict) -> bool:
        """
        Verify that a group has sustained co-movement by checking actual overlapping frames
        
        Args:
            group (list): List of clip IDs in the group
            clips_data (dict): Dictionary mapping clip IDs to their detections
            
        Returns:
            bool: True if group has sustained co-movement
        """
        if len(group) < 2:
            return False
        
        # Get actual frame sets for each group member
        member_frame_sets = []
        for clip_id in group:
            frames = set(d['frame_number'] for d in clips_data[clip_id])
            member_frame_sets.append(frames)
        
        # Since stars move slowly, detections won't be at exactly the same frames
        # Instead, we'll check if detections are close in time (within a tolerance)
        # Build a time window approach: for each frame where ANY member is detected,
        # check if ALL members have detections within a small window around it
        
        frame_tolerance = 5  # Allow 5 frames of tolerance for detection timing
        
        # Get all unique frames where any member is detected
        all_frames = set()
        for frames in member_frame_sets:
            all_frames.update(frames)
        
        if not all_frames:
            return False
        
        # Find frames where all members are "nearby" (within tolerance)
        cooccurring_frames = []
        
        for target_frame in sorted(all_frames):
            all_members_present = True
            
            for member_frames in member_frame_sets:
                # Check if this member has a detection within tolerance of target_frame
                has_nearby_detection = any(
                    abs(f - target_frame) <= frame_tolerance 
                    for f in member_frames
                )
                
                if not has_nearby_detection:
                    all_members_present = False
                    break
            
            if all_members_present:
                cooccurring_frames.append(target_frame)
        
        if not cooccurring_frames:
            return False
        
        # Get FPS from first detection (assuming all have same FPS)
        fps = clips_data[group[0]][0].get('fps', 30)
        
        # Check if we have enough co-occurring frames (at least 1 second worth)
        min_required_cooccur_count = int(fps * 0.5)  # At least 0.5 seconds of actual co-occurrences
        
        if len(cooccurring_frames) < min_required_cooccur_count:
            return False
        
        # Also check the time span to ensure detections cover at least 1 second
        min_cooccur_frame = min(cooccurring_frames)
        max_cooccur_frame = max(cooccurring_frames)
        cooccur_span_frames = max_cooccur_frame - min_cooccur_frame + 1
        
        min_required_span = fps * 1.0
        
        return cooccur_span_frames >= min_required_span
    
    def _calculate_velocity_vector(self, detections: List[Dict]) -> Tuple[float, float, float]:
        """
        Calculate average velocity vector for a sequence of detections
        
        Args:
            detections (list): List of detection dictionaries for a single object
            
        Returns:
            tuple: (vx, vy, speed) or None if cannot calculate
        """
        if len(detections) < 2:
            return None
        
        # Sort by frame number
        detections = sorted(detections, key=lambda x: x['frame_number'])
        
        # Calculate velocity vectors between consecutive frames
        velocities = []
        
        for i in range(1, len(detections)):
            dx = detections[i]['centroid_x'] - detections[i-1]['centroid_x']
            dy = detections[i]['centroid_y'] - detections[i-1]['centroid_y']
            dt = detections[i]['frame_number'] - detections[i-1]['frame_number']
            
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                velocities.append((vx, vy))
        
        if not velocities:
            return None
        
        # Calculate average velocity
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        avg_speed = math.sqrt(avg_vx**2 + avg_vy**2)
        
        # Filter out very slow motion (but keep threshold low for stars)
        if avg_speed < self.min_speed:
            return None
        
        return (avg_vx, avg_vy, avg_speed)
    
    def _are_velocities_similar(self, vel1: Tuple[float, float, float], 
                                vel2: Tuple[float, float, float]) -> bool:
        """
        Check if two velocity vectors are similar (same speed and direction)
        
        Args:
            vel1 (tuple): First velocity vector (vx, vy, speed)
            vel2 (tuple): Second velocity vector (vx, vy, speed)
            
        Returns:
            bool: True if velocities are similar
        """
        vx1, vy1, speed1 = vel1
        vx2, vy2, speed2 = vel2
        
        # Check if speeds are similar (within tolerance)
        avg_speed = (speed1 + speed2) / 2
        if avg_speed == 0:
            return False
        
        speed_diff = abs(speed1 - speed2) / avg_speed
        if speed_diff > self.velocity_tolerance:
            return False
        
        # Check if directions are similar
        angle1 = math.atan2(vy1, vx1) * 180 / math.pi
        angle2 = math.atan2(vy2, vx2) * 180 / math.pi
        
        # Calculate angular difference (handle wrap-around at 360 degrees)
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        if angle_diff > self.direction_tolerance:
            return False
        
        return True
    
    def get_star_field_info(self, metadata: List[Dict], star_clip_ids: Set[int]) -> Dict:
        """
        Get information about the detected star field
        
        Args:
            metadata (list): List of detection metadata dictionaries
            star_clip_ids (set): Set of clip IDs classified as stars
            
        Returns:
            dict: Information about the star field including average velocity and count
        """
        if not star_clip_ids:
            return {
                'star_count': 0,
                'avg_velocity_x': 0,
                'avg_velocity_y': 0,
                'avg_speed': 0,
                'direction_deg': 0
            }
        
        # Group metadata by clip_id
        clips_data = defaultdict(list)
        for item in metadata:
            clips_data[item['clip_id']].append(item)
        
        # Calculate velocities for all stars
        velocities = []
        for clip_id in star_clip_ids:
            if clip_id in clips_data:
                vel = self._calculate_velocity_vector(clips_data[clip_id])
                if vel is not None:
                    velocities.append(vel)
        
        if not velocities:
            return {
                'star_count': len(star_clip_ids),
                'avg_velocity_x': 0,
                'avg_velocity_y': 0,
                'avg_speed': 0,
                'direction_deg': 0
            }
        
        # Calculate average star field motion
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        avg_speed = np.mean([v[2] for v in velocities])
        direction_deg = math.atan2(avg_vy, avg_vx) * 180 / math.pi
        
        return {
            'star_count': len(star_clip_ids),
            'avg_velocity_x': float(avg_vx),
            'avg_velocity_y': float(avg_vy),
            'avg_speed': float(avg_speed),
            'direction_deg': float(direction_deg)
        }
