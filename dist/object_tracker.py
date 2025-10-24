import numpy as np
from scipy.spatial import distance as dist
from collections import defaultdict

class ObjectTracker:
    def __init__(self, max_disappeared=15, max_distance=50):
        """
        Simple centroid-based object tracker
        
        Args:
            max_disappeared (int): Maximum frames an object can be missing before deregistration
            max_distance (float): Maximum distance to consider same object
        """
        self.next_object_id = 1
        self.objects = {}  # object_id -> centroid
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """Register a new object with next available ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections (list): List of (centroid_x, centroid_y) tuples
            
        Returns:
            dict: Mapping of object_id to centroid for current frame
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return {}
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for centroid in detections:
                self.register(centroid)
        else:
            # Get current object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance between each pair of object/detection centroids
            D = dist.cdist(np.array(object_centroids), np.array(detections))
            
            # Find the smallest distance for each object
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match objects to detections
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Check if distance is within threshold
                if D[row, col] > self.max_distance:
                    continue
                
                # Update object position
                object_id = object_ids[row]
                self.objects[object_id] = detections[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(detections[col])
        
        # Return current tracked objects
        return self.objects.copy()
