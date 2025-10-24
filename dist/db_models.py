import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import NullPool

Base = declarative_base()

class AnalysisSession(Base):
    """Represents a video analysis session"""
    __tablename__ = 'analysis_sessions'
    
    id = Column(Integer, primary_key=True)
    video_filename = Column(String(500), nullable=False)
    video_duration = Column(Float)
    video_fps = Column(Float)
    video_resolution = Column(String(50))
    processing_date = Column(DateTime, default=datetime.utcnow)
    total_detections = Column(Integer, default=0)
    satellites_count = Column(Integer, default=0)
    meteors_count = Column(Integer, default=0)
    planes_count = Column(Integer, default=0)
    anomalies_count = Column(Integer, default=0)
    junk_count = Column(Integer, default=0)
    
    # Relationships
    clips = relationship("DetectionClip", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<AnalysisSession(id={self.id}, video={self.video_filename}, date={self.processing_date})>"


class DetectionClip(Base):
    """Represents a detected motion clip with ML classification"""
    __tablename__ = 'detection_clips'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'), nullable=False)
    clip_id = Column(Integer, nullable=False)
    clip_filename = Column(String(500))
    
    # Classification results
    classification = Column(String(50), nullable=False)
    confidence = Column(Float)
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean, default=False)
    
    # Movement features
    duration = Column(Float)
    avg_speed = Column(Float)
    max_speed = Column(Float)
    speed_consistency = Column(Float)
    linearity = Column(Float)
    direction_changes = Column(Integer)
    
    # Object characteristics
    avg_area = Column(Float)
    max_area = Column(Float)
    size_consistency = Column(Float)
    avg_aspect_ratio = Column(Float)
    
    # Detection metadata
    detection_count = Column(Integer)
    first_frame = Column(Integer)
    last_frame = Column(Integer)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="clips")
    detections = relationship("ObjectDetection", back_populates="clip", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DetectionClip(id={self.id}, classification={self.classification}, confidence={self.confidence})>"


class ObjectDetection(Base):
    """Represents individual object detections within a clip"""
    __tablename__ = 'object_detections'
    
    id = Column(Integer, primary_key=True)
    clip_id = Column(Integer, ForeignKey('detection_clips.id'), nullable=False)
    
    frame_number = Column(Integer, nullable=False)
    centroid_x = Column(Float)
    centroid_y = Column(Float)
    bbox_x = Column(Integer)
    bbox_y = Column(Integer)
    bbox_width = Column(Integer)
    bbox_height = Column(Integer)
    area = Column(Float)
    aspect_ratio = Column(Float)
    
    # Relationships
    clip = relationship("DetectionClip", back_populates="detections")
    
    def __repr__(self):
        return f"<ObjectDetection(id={self.id}, frame={self.frame_number}, pos=({self.centroid_x}, {self.centroid_y}))>"


# Database setup functions
def get_database_url():
    """Get database URL from environment variables"""
    return os.environ.get('DATABASE_URL')


def init_database():
    """Initialize database connection and create tables"""
    database_url = get_database_url()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(database_url, poolclass=NullPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine


def get_session():
    """Get a database session"""
    database_url = get_database_url()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(database_url, poolclass=NullPool)
    Session = sessionmaker(bind=engine)
    return Session()
