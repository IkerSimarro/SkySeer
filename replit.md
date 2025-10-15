# SkySeer AI

## Overview

SkySeer is an advanced computer vision and machine learning system for detecting and classifying sky objects in night sky video footage. It focuses on identifying very obvious movements like satellite passes and meteor events, minimizing false positives. The system processes raw video into structured data through motion detection, numerical "flight signature" extraction, and K-Means clustering for categorization (satellites, meteors, planes). It aggressively filters results to achieve less than 10 detections per typical video, prioritizing precision over recall. This project is ideal for amateur astronomy and low-light camera setups.

## Recent Changes

**October 15, 2025 - Latest:**

**Critical Accuracy Improvements:**
- **Enhanced Small Satellite Detection:** Lowered minimum object size threshold from 40 to ~15 pixels to capture distant satellites
- **Improved Background Subtraction:** Reduced varThreshold from 60 to 45 for better detection of dim, small objects
- **Speed-Based Classification Filter:** Added critical speed validation to satellite scoring
  - Objects moving <0.8 px/frame receive 90% score penalty (eliminates false positives from slow-moving noise)
  - Objects moving 0.8-1.5 px/frame receive 50% penalty
  - Optimal satellite speed range: 1.5-35 px/frame
  - Very fast objects (>35 px/frame) penalized to favor meteor classification
- **Balanced Satellite vs Plane Scoring:**
  - Satellites peak at 3-15s duration (1.4x boost), remain strong for longer passes (1.0x up to 25s)
  - Planes require EITHER blinking lights OR long duration (15+s) to score high
  - Planes without blinking now get same speed penalties as satellites
  - Reduced plane blinking bonus from 1.8x to 1.5x for better balance
- **Result:** Significantly reduced false positives from stationary/very slow objects while improving detection of genuine small satellites and proper satellite/plane discrimination

**October 15, 2025:**

**Enhanced Plane Detection:**
- Implemented sophisticated blinking light pattern detection for improved aircraft identification
- Added periodic blinking analysis that detects on/off brightness patterns typical of navigation lights
- Increased blinking bonus from 0.2x to 0.8x for stronger plane vs satellite discrimination
- New blinking_score feature combines brightness variance with periodic flash detection
- Safeguarded against division by zero in brightness calculations

**ML Classification Optimization:**
- Added blinking_score to 11-dimensional feature space for better accuracy
- Enhanced feature set now includes: speed, consistency, duration, linearity, direction changes, size consistency, acceleration, blinking patterns, and object-specific scores

**ZIP Download Enhancements:**
- Added SUMMARY.txt file with detection counts and detailed object listings
- Enhanced README.txt with improved organization and file structure explanation
- Summary includes: classification, confidence %, duration, and average speed for each detected object

**Comprehensive 502 Timeout Fix (Updated October 15, 2025):**
- Fixed critical WebSocket timeout issue affecting ALL processing stages
- **Stage 1 (Motion Detection):** Progress callbacks fire every (30 × frame_skip) frames, sending updates every 3-6 seconds
- **Stage 2 (Feature Extraction):** Added multi-step progress updates with unique status text
- **Stage 3 (ML Classification):** Implemented intermediate progress callbacks during AI analysis
- **Stage 4 (Rectangle Drawing):** Added per-clip progress updates to prevent timeout on large result sets
- All status updates use unique content (callback counts, frame numbers, percentages) to prevent Streamlit deduplication
- WebSocket now receives fresh data throughout entire pipeline, preventing 60-second timeout
- Fixed ML classifier defensive programming: uses .get() methods with fallbacks for all field access
- Works with all video lengths and processing scenarios

**UI Improvements:**
- Removed "About Me" section for cleaner, more professional interface
- "How It Works" section now displays in full width
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

The system utilizes an unsupervised learning approach with K-Means Clustering (3 clusters) for automatic categorization of motion patterns. It operates on an 11-dimensional feature space derived from kinematic "flight signatures" and brightness patterns rather than visual features, which are more reliable in low-light conditions. The feature space includes: average speed, speed consistency, duration, linearity, direction changes, size consistency, acceleration, blinking patterns, and object-specific scores. This approach requires no labeled training data and adapts to new data patterns without retraining.

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