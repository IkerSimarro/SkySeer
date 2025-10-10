import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import math


class TrajectoryVisualizer:
    """Visualize object trajectories on interactive star charts"""
    
    def __init__(self, frame_width=1920, frame_height=1080):
        """
        Initialize trajectory visualizer
        
        Args:
            frame_width (int): Video frame width
            frame_height (int): Video frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def create_trajectory_plot(self, metadata, classification='All'):
        """
        Create interactive trajectory plot showing object paths
        
        Args:
            metadata (list): List of detection metadata dictionaries
            classification (str): Filter by classification or 'All'
            
        Returns:
            plotly.graph_objects.Figure: Interactive trajectory plot
        """
        if not metadata:
            return self._create_empty_plot()
        
        # Group by clip_id
        clips_data = {}
        for item in metadata:
            clip_id = item.get('clip_id', -1)
            if clip_id not in clips_data:
                clips_data[clip_id] = []
            clips_data[clip_id].append(item)
        
        # Create figure with dark background (star field)
        fig = go.Figure()
        
        # Add starfield background effect
        fig.update_layout(
            plot_bgcolor='#0a0e27',
            paper_bgcolor='#0a0e27',
            font=dict(color='white'),
            xaxis=dict(
                range=[0, self.frame_width],
                showgrid=False,
                zeroline=False,
                title="X Position (pixels)"
            ),
            yaxis=dict(
                range=[self.frame_height, 0],  # Inverted for image coordinates
                showgrid=False,
                zeroline=False,
                title="Y Position (pixels)",
                scaleanchor="x",
                scaleratio=1
            ),
            title="Object Trajectory Visualization",
            hovermode='closest',
            height=600
        )
        
        # Color mapping for classifications
        color_map = {
            'Satellite': '#1f77b4',  # Blue
            'Meteor': '#ff7f0e',     # Orange
            'Plane': '#2ca02c',      # Green
            'Junk': '#d62728',       # Red
            'ANOMALY_UAP': '#9467bd' # Purple
        }
        
        # Plot each trajectory
        for clip_id, detections in clips_data.items():
            if len(detections) < 2:
                continue
            
            # Sort by frame number
            detections.sort(key=lambda x: x.get('frame_number', 0))
            
            # Extract positions
            x_coords = [d.get('centroid_x', 0) for d in detections]
            y_coords = [d.get('centroid_y', 0) for d in detections]
            frames = [d.get('frame_number', 0) for d in detections]
            
            # Determine color based on classification (if available)
            clip_class = detections[0].get('classification', 'Unknown')
            color = color_map.get(clip_class, '#808080')
            
            # Calculate trajectory metrics for hover info
            distances = []
            for i in range(1, len(x_coords)):
                dx = x_coords[i] - x_coords[i-1]
                dy = y_coords[i] - y_coords[i-1]
                dist = math.sqrt(dx*dx + dy*dy)
                distances.append(dist)
            
            avg_speed = np.mean(distances) if distances else 0
            
            # Add trajectory line
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                name=f'Clip {clip_id} ({clip_class})',
                line=dict(color=color, width=2),
                marker=dict(
                    size=6,
                    color=color,
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                hovertemplate=(
                    f'<b>Clip {clip_id}</b><br>' +
                    'Position: (%{x:.0f}, %{y:.0f})<br>' +
                    f'Classification: {clip_class}<br>' +
                    f'Avg Speed: {avg_speed:.1f} px/frame<br>' +
                    '<extra></extra>'
                )
            ))
            
            # Add start and end markers
            fig.add_trace(go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='lime',
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                name=f'Start {clip_id}',
                hovertemplate=f'<b>Start</b><br>Frame: {frames[0]}<extra></extra>',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='white')
                ),
                name=f'End {clip_id}',
                hovertemplate=f'<b>End</b><br>Frame: {frames[-1]}<extra></extra>',
                showlegend=False
            ))
        
        return fig
    
    def create_speed_heatmap(self, metadata):
        """
        Create heatmap showing speed distribution across the frame
        
        Args:
            metadata (list): List of detection metadata
            
        Returns:
            plotly.graph_objects.Figure: Speed heatmap
        """
        if not metadata:
            return self._create_empty_plot()
        
        # Create grid for heatmap
        grid_size = 50
        x_bins = np.linspace(0, self.frame_width, grid_size)
        y_bins = np.linspace(0, self.frame_height, grid_size)
        
        speed_grid = np.zeros((grid_size-1, grid_size-1))
        count_grid = np.zeros((grid_size-1, grid_size-1))
        
        # Group by clip and calculate speeds
        clips_data = {}
        for item in metadata:
            clip_id = item.get('clip_id', -1)
            if clip_id not in clips_data:
                clips_data[clip_id] = []
            clips_data[clip_id].append(item)
        
        # Calculate speeds and bin them
        for clip_id, detections in clips_data.items():
            detections.sort(key=lambda x: x.get('frame_number', 0))
            
            for i in range(1, len(detections)):
                x1, y1 = detections[i-1].get('centroid_x', 0), detections[i-1].get('centroid_y', 0)
                x2, y2 = detections[i].get('centroid_x', 0), detections[i].get('centroid_y', 0)
                
                speed = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Find bin indices
                x_idx = np.digitize(x2, x_bins) - 1
                y_idx = np.digitize(y2, y_bins) - 1
                
                if 0 <= x_idx < grid_size-1 and 0 <= y_idx < grid_size-1:
                    speed_grid[y_idx, x_idx] += speed
                    count_grid[y_idx, x_idx] += 1
        
        # Average speeds
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_speed_grid = np.where(count_grid > 0, speed_grid / count_grid, 0)
        
        fig = go.Figure(data=go.Heatmap(
            z=avg_speed_grid,
            x=x_bins[:-1],
            y=y_bins[:-1],
            colorscale='Viridis',
            colorbar=dict(title="Avg Speed<br>(px/frame)")
        ))
        
        fig.update_layout(
            title="Speed Distribution Heatmap",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=500
        )
        
        return fig
    
    def create_direction_plot(self, metadata):
        """
        Create polar plot showing movement directions
        
        Args:
            metadata (list): List of detection metadata
            
        Returns:
            plotly.graph_objects.Figure: Direction distribution plot
        """
        if not metadata:
            return self._create_empty_plot()
        
        # Group by clip
        clips_data = {}
        for item in metadata:
            clip_id = item.get('clip_id', -1)
            if clip_id not in clips_data:
                clips_data[clip_id] = []
            clips_data[clip_id].append(item)
        
        # Calculate directions
        directions = []
        speeds = []
        
        for clip_id, detections in clips_data.items():
            detections.sort(key=lambda x: x.get('frame_number', 0))
            
            if len(detections) < 2:
                continue
            
            # Overall direction from start to end
            x1, y1 = detections[0].get('centroid_x', 0), detections[0].get('centroid_y', 0)
            x2, y2 = detections[-1].get('centroid_x', 0), detections[-1].get('centroid_y', 0)
            
            dx, dy = x2 - x1, y2 - y1
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 360
            
            speed = math.sqrt(dx*dx + dy*dy) / len(detections)
            
            directions.append(angle)
            speeds.append(speed)
        
        # Create polar histogram
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=speeds,
            theta=directions,
            mode='markers',
            marker=dict(
                size=10,
                color=speeds,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Speed")
            ),
            hovertemplate='Direction: %{theta:.0f}°<br>Speed: %{r:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(title="Speed (px/frame)"),
                angularaxis=dict(direction="clockwise")
            ),
            title="Movement Direction Distribution",
            height=500
        )
        
        return fig
    
    def create_timeline_plot(self, metadata, fps=30):
        """
        Create timeline plot showing when objects were detected
        
        Args:
            metadata (list): List of detection metadata
            fps (float): Video frame rate
            
        Returns:
            plotly.graph_objects.Figure: Timeline plot
        """
        if not metadata:
            return self._create_empty_plot()
        
        # Convert fps to float if it's a string
        try:
            fps = float(fps)
        except (ValueError, TypeError):
            fps = 30.0
        
        # Group by clip
        clips_data = {}
        for item in metadata:
            clip_id = item.get('clip_id', -1)
            if clip_id not in clips_data:
                clips_data[clip_id] = []
            clips_data[clip_id].append(item)
        
        # Prepare timeline data
        timeline_data = []
        
        for clip_id, detections in clips_data.items():
            detections.sort(key=lambda x: x.get('frame_number', 0))
            
            start_frame = detections[0].get('frame_number', 0)
            end_frame = detections[-1].get('frame_number', 0)
            
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            
            classification = detections[0].get('classification', 'Unknown')
            
            timeline_data.append({
                'clip_id': clip_id,
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'classification': classification
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Color mapping
        color_map = {
            'Satellite': '#1f77b4',
            'Meteor': '#ff7f0e',
            'Plane': '#2ca02c',
            'Junk': '#d62728',
            'ANOMALY_UAP': '#9467bd'
        }
        
        fig = go.Figure()
        
        for _, row in df.iterrows():
            color = color_map.get(row['classification'], '#808080')
            
            fig.add_trace(go.Scatter(
                x=[row['start'], row['end']],
                y=[row['clip_id'], row['clip_id']],
                mode='lines+markers',
                line=dict(color=color, width=10),
                marker=dict(size=8, color=color),
                name=f"Clip {row['clip_id']}",
                hovertemplate=(
                    f"<b>Clip {row['clip_id']}</b><br>" +
                    f"Classification: {row['classification']}<br>" +
                    f"Start: {row['start']:.2f}s<br>" +
                    f"Duration: {row['duration']:.2f}s<br>" +
                    "<extra></extra>"
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Detection Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Clip ID",
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    def _create_empty_plot(self):
        """Create empty plot when no data available"""
        fig = go.Figure()
        fig.add_annotation(
            text="No trajectory data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
