from datetime import datetime
from db_models import AnalysisSession, DetectionClip, ObjectDetection, init_database, get_session
import pandas as pd


class DatabaseService:
    """Service layer for database operations"""
    
    def __init__(self):
        """Initialize database and create tables if they don't exist"""
        try:
            session, engine = init_database()
            session.close()
            engine.dispose()
        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
    
    def save_analysis_session(self, video_info, results_df, clips):
        """
        Save complete analysis session to database
        
        Args:
            video_info (dict): Video metadata (filename, duration, fps, resolution)
            results_df (pd.DataFrame): Classification results dataframe
            clips (list): List of clip file paths
            
        Returns:
            int: Session ID
        """
        session = None
        try:
            session = get_session()
            
            # Count classifications
            classification_counts = results_df['classification'].value_counts().to_dict()
            
            # Create analysis session with safe type conversions
            video_fps = video_info.get('fps_numeric', 30)
            if isinstance(video_fps, str) or video_fps is None:
                video_fps = 30  # Default if not available
            
            video_duration = video_info.get('duration_seconds', 0)
            if isinstance(video_duration, str) or video_duration is None:
                video_duration = 0
            
            analysis_session = AnalysisSession(
                video_filename=video_info.get('filename', 'unknown.mp4'),
                video_duration=float(video_duration),
                video_fps=float(video_fps),
                video_resolution=video_info.get('resolution', 'unknown'),
                total_detections=len(results_df),
                satellites_count=classification_counts.get('Satellite', 0),
                meteors_count=classification_counts.get('Meteor', 0),
                planes_count=classification_counts.get('Plane', 0),
                anomalies_count=classification_counts.get('ANOMALY_UAP', 0),
                junk_count=classification_counts.get('Junk', 0)
            )
            
            session.add(analysis_session)
            session.flush()  # Get the session ID
            
            # Save each detection clip
            for idx, row in results_df.iterrows():
                clip_id = row['clip_id']
                
                # Find corresponding clip file
                clip_filename = None
                for clip_path in clips:
                    if f"clip_{clip_id:04d}" in clip_path:
                        clip_filename = clip_path
                        break
                
                # Safe conversion helper for handling NaN/None values
                def safe_float(val, default=0.0):
                    try:
                        if pd.isna(val):
                            return default
                        return float(val)
                    except:
                        return default
                
                def safe_int(val, default=0):
                    try:
                        if pd.isna(val):
                            return default
                        return int(val)
                    except:
                        return default
                
                detection_clip = DetectionClip(
                    session_id=analysis_session.id,
                    clip_id=clip_id,
                    clip_filename=clip_filename,
                    classification=str(row.get('classification', 'Unknown')),
                    confidence=safe_float(row.get('confidence'), 0),
                    anomaly_score=safe_float(row.get('anomaly_score'), 0),
                    is_anomaly=bool(row.get('is_anomaly', False)),
                    duration=safe_float(row.get('duration'), 0),
                    avg_speed=safe_float(row.get('avg_speed'), 0),
                    max_speed=safe_float(row.get('max_speed'), 0),
                    speed_consistency=safe_float(row.get('speed_consistency'), 0),
                    linearity=safe_float(row.get('linearity'), 0),
                    direction_changes=safe_int(row.get('direction_changes'), 0),
                    avg_area=safe_float(row.get('avg_area'), 0),
                    max_area=safe_float(row.get('max_area'), 0),
                    size_consistency=safe_float(row.get('size_consistency'), 0),
                    avg_aspect_ratio=safe_float(row.get('avg_aspect_ratio'), 1.0),
                    detection_count=safe_int(row.get('detection_count'), 0)
                )
                
                session.add(detection_clip)
            
            session.commit()
            session_id = analysis_session.id
            session.close()
            
            return session_id
            
        except Exception as e:
            print(f"Error saving analysis session: {e}")
            if session:
                session.rollback()
                session.close()
            return None
    
    def get_all_sessions(self):
        """Get all analysis sessions ordered by date (newest first)"""
        session = None
        try:
            session = get_session()
            sessions = session.query(AnalysisSession).order_by(AnalysisSession.processing_date.desc()).all()
            session.close()
            return sessions
        except Exception as e:
            print(f"Error retrieving sessions: {e}")
            return []
    
    def get_session_by_id(self, session_id):
        """Get specific analysis session with all clips"""
        session = None
        try:
            session = get_session()
            analysis_session = session.query(AnalysisSession).filter_by(id=session_id).first()
            session.close()
            return analysis_session
        except Exception as e:
            print(f"Error retrieving session: {e}")
            return None
    
    def get_all_detections(self, classification_filter=None, min_confidence=0.0):
        """
        Get all detection clips with optional filters
        
        Args:
            classification_filter (list): List of classifications to include (e.g., ['Satellite', 'ANOMALY_UAP'])
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            list: List of DetectionClip objects
        """
        session = None
        try:
            session = get_session()
            query = session.query(DetectionClip).filter(DetectionClip.confidence >= min_confidence)
            
            if classification_filter:
                query = query.filter(DetectionClip.classification.in_(classification_filter))
            
            clips = query.order_by(DetectionClip.detected_at.desc()).all()
            session.close()
            return clips
        except Exception as e:
            print(f"Error retrieving detections: {e}")
            return []
    
    def get_statistics(self, days=30):
        """
        Get detection statistics for the last N days
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            dict: Statistics dictionary
        """
        session = None
        try:
            session = get_session()
            
            # Get sessions from last N days
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            sessions = session.query(AnalysisSession).filter(
                AnalysisSession.processing_date >= cutoff_date
            ).all()
            
            # Calculate statistics
            total_sessions = len(sessions)
            total_detections = sum(s.total_detections for s in sessions)
            total_satellites = sum(s.satellites_count for s in sessions)
            total_meteors = sum(s.meteors_count for s in sessions)
            total_planes = sum(s.planes_count for s in sessions)
            total_anomalies = sum(s.anomalies_count for s in sessions)
            total_junk = sum(s.junk_count for s in sessions)
            
            stats = {
                'total_sessions': total_sessions,
                'total_detections': total_detections,
                'satellites': total_satellites,
                'meteors': total_meteors,
                'planes': total_planes,
                'anomalies': total_anomalies,
                'junk': total_junk,
                'days': days
            }
            
            session.close()
            return stats
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}
    
    def delete_session(self, session_id):
        """Delete an analysis session and all its clips"""
        session = None
        try:
            session = get_session()
            analysis_session = session.query(AnalysisSession).filter_by(id=session_id).first()
            
            if analysis_session:
                session.delete(analysis_session)
                session.commit()
                session.close()
                return True
            
            session.close()
            return False
            
        except Exception as e:
            print(f"Error deleting session: {e}")
            if session:
                session.rollback()
                session.close()
            return False
