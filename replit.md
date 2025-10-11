# SkySeer AI

## Overview

SkySeer is an advanced computer vision and machine learning system for detecting and classifying sky objects in night sky video footage. It focuses on identifying very obvious movements like satellite passes and meteor events, minimizing false positives. The system processes raw video into structured data through motion detection, numerical "flight signature" extraction, and K-Means clustering for categorization (satellites, meteors, planes). It aggressively filters results to achieve less than 10 detections per typical video, prioritizing precision over recall. This project is ideal for amateur astronomy and low-light camera setups.

## Recent Changes

**October 11, 2025:**

**502 Timeout Fix for Long Videos:**
- Fixed critical WebSocket timeout issue that occurred with videos longer than 5 minutes
- Implemented progress callback system that fires every (30 × frame_skip) processed frames
- Status text updates with unique content (callback count + frame number) to prevent Streamlit deduplication
- WebSocket now receives fresh data every 3-6 seconds, preventing 60-second timeout
- Works with all video lengths and handles unknown frame counts gracefully

**UI Improvements:**
- Corrected Maximum Clip Duration slider range from 5-120s to 5-30s to align with typical recommendation values (~15s)

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

The application uses Streamlit as a single-page web application with a sidebar configuration panel. It provides a web-based interface for video upload (up to 5GB), real-time processing feedback, interactive data visualization (Plotly), advanced trajectory visualization, and downloadable results. The interface consists of a main content area for video upload, processing controls, and results, and a sidebar for configuration parameters.

### Backend Architecture

The system employs a multi-stage sequential processing pipeline:

1.  **Video Ingestion**: Uses OpenCV for frame extraction, MOG2 background subtraction for motion detection, and generates motion clips with metadata.
2.  **Feature Extraction**: Transforms visual detections into numerical "flight signatures" by calculating kinematic metrics (speed, acceleration, trajectory linearity) and consistency scores.
3.  **ML Classification**: Applies StandardScaler for feature normalization and K-Means clustering (3 clusters) to categorize objects as Satellite, Meteor, Plane, or Junk, with confidence scores.
4.  **Utility Functions**: Handles video metadata, file operations (ZIP creation), and formatting.
5.  **Trajectory Visualization**: Creates interactive path visualizations, speed heatmaps, polar plots for direction, and timeline analyses.
6.  **Database Service**: Manages PostgreSQL connections for persisting analysis sessions, detection clips, and object detections, enabling multi-night analysis and historical tracking.

The pipeline architecture separates concerns for maintainability and extensibility. It is robust to visual noise, uses unsupervised learning, and statistical feature analysis.

### Data Storage Solutions

A hybrid storage architecture combines file-based storage and a PostgreSQL database. File-based storage handles input video files and output classified motion clips (organized into Satellite/, Meteor/, Plane/, Junk/ directories). Temporary files are used during OpenCV processing. The PostgreSQL database stores `AnalysisSession`, `DetectionClip`, and `ObjectDetection` data to enable multi-night analysis, historical tracking, and query-based analysis.

### Machine Learning Model Architecture

The system utilizes an unsupervised learning approach with K-Means Clustering (3 clusters) for automatic categorization of motion patterns. It operates on a 10-dimensional feature space derived from kinematic "flight signatures" rather than visual features, which are more reliable in low-light conditions. This approach requires no labeled training data and adapts to new data patterns without retraining.

## External Dependencies

### Computer Vision Libraries
-   **OpenCV (cv2)**: Video processing, frame extraction, MOG2 background subtraction.
-   **NumPy**: Numerical operations.

### Machine Learning Libraries
-   **scikit-learn**: K-Means clustering, feature scaling.

### Web Framework & UI
-   **Streamlit**: Web application framework, UI components.
-   **Plotly**: Interactive data visualization.

### Data Processing
-   **Pandas**: DataFrame operations.

### Utility Libraries
-   **Python Standard Library**: File/directory operations (`os`, `shutil`), archive creation (`zipfile`), in-memory file handling (`io.BytesIO`), timestamp generation (`datetime`), efficient data structures (`collections`), CSV handling (`csv`), encoding (`base64`), temporary files (`tempfile`).