# SkySeer AI Pipeline

## Overview

SkySeer is an advanced sky object detection and classification system that uses computer vision and machine learning to analyze night sky video footage. The system is optimized to detect **very obvious movement only** - focusing on clear satellite passes and meteor events while minimizing false positives from noise and atmospheric artifacts.

The pipeline transforms raw video into structured data by:
1. Detecting motion events using conservative computer vision techniques
2. Extracting numerical "flight signatures" (speed, trajectory, consistency metrics)
3. Applying K-Means clustering to automatically categorize detections (satellites, meteors, planes)
4. Filtering aggressively to achieve <10 detections per typical video

This system is designed for amateur astronomy and sky observation projects, particularly suited for low-light camera equipment like Raspberry Pi NoIR modules. The focus is on precision over recall - it's better to miss an edge case than to flood results with false positives.

## Recent Changes

**October 11, 2025** (User-Requested Improvements):

**Meteor Detection Enhancement:**
- **Improved meteor scoring algorithm** - Now better detects fast, bright, short-duration objects:
  - Speed factor: avg_speed/30 capped at 3x for very fast meteors
  - Duration factor: Heavy penalty for objects >3s (meteors are brief)
  - Brightness factor: Up to 1.5x boost for bright objects (meteors often flash brightly)
  - New formula: speed_factor * linearity * duration_factor * brightness_factor

**UI/UX Improvements:**
- **Removed "Enable Maximum Duration Filter" checkbox** - Max duration slider now always active
- **Increased max duration limit to 120 seconds** (up from 30s) - Stars won't move out of frame even at 120s
- **Simplified configuration** - One less control to worry about

**Smart Video Analysis:**
- **Deep content analysis** - System now samples 10 frames across video to analyze:
  - Average brightness (detects very dark night sky vs twilight)
  - Noise levels (detects grainy/noisy footage)
  - Contrast levels (overall video quality)
- **Personalized recommendations** - Settings now tailored to actual video content:
  - Very dark & noisy videos: Lower sensitivity to reduce false positives
  - Clean dark videos: Higher sensitivity for better detection
  - Bright videos: Lower sensitivity to avoid over-detection
  - High noise videos: Suggests longer min duration for reliability
  - Long videos (>10min): Higher frame skip and adaptive max duration

**October 10, 2025** (Major Overhaul):

**Detection System Improvements:**
- **Removed UAP/anomaly detection** - Eliminated Isolation Forest and all anomaly classification (excessive false positives)
- **Made detection much more conservative** - System now targets <10 satellites and 1 meteor per video:
  - Increased background subtractor variance threshold: 50 → 100 (only obvious motion)
  - Increased minimum contour area: 3x larger (55 pixels for sensitivity=5)
  - Fixed critical duration filtering bug that didn't account for frame skipping
  - Increased default minimum duration: 0.5s → 1.5s
  - Enabled maximum duration filter by default (15s) to filter out stationary stars
- **Added automatic settings recommendations** - System analyzes uploaded video and suggests optimal parameters based on:
  - Video duration (adjusts frame skip)
  - Resolution (adjusts sensitivity for noise)
  - FPS (adjusts duration thresholds)
  - Recommendations appear in sidebar below configuration with explanations

**Previous Changes:**
- Fixed 502 timeout error for large video uploads (900MB+)
- Added advanced trajectory visualization system
- Enhanced database persistence with metadata storage

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack**: Streamlit web application framework

The application uses Streamlit for the user interface, providing:
- Web-based video upload and processing interface (supports files up to 5GB)
- Real-time processing feedback with progress indicators
- Interactive data visualization using Plotly (charts and graphs)
- Advanced trajectory visualization (path tracking, heatmaps, direction analysis, timeline)
- Session state management for maintaining processing results
- Downloadable results packages (ZIP files with classified clips)

**Design Pattern**: Single-page application with sidebar configuration panel

The interface is organized into:
- Main content area for video upload, processing controls, and results display
- Sidebar for configuration parameters (sensitivity, duration thresholds)
- Session state persistence to maintain results between interactions
- Tabbed trajectory analysis section with interactive visualizations

### Backend Architecture

**Processing Pipeline**: Multi-stage video analysis workflow

The system follows a sequential processing model:

1. **Video Ingestion** (`video_processor.py`)
   - Uses OpenCV for video frame extraction and analysis
   - Implements MOG2 background subtraction for motion detection
   - Generates motion clips with temporal buffering (pre/post motion frames)
   - Outputs metadata for each detected motion event

2. **Feature Extraction** (`feature_extractor.py`)
   - Transforms visual detections into numerical features
   - Calculates kinematic metrics: speed, acceleration, trajectory linearity
   - Computes consistency scores: speed variance, direction changes, size stability
   - Generates specialized scores for object type classification (satellite/meteor/plane patterns)

3. **ML Classification** (`ml_classifier.py`)
   - Applies StandardScaler for feature normalization
   - Uses K-Means clustering (3 clusters) to group similar motion patterns
   - Classifies objects as: Satellite, Meteor, Plane, or Junk
   - Assigns confidence scores based on trajectory quality and consistency

4. **Utility Functions** (`utils.py`)
   - Handles video metadata extraction
   - Manages file operations (ZIP creation, temporary files)
   - Provides formatting utilities for duration and file sizes

5. **Trajectory Visualization** (`trajectory_visualizer.py`)
   - Creates interactive path visualizations showing object trajectories
   - Generates speed heatmaps showing object movement across the frame
   - Produces polar plots for direction distribution analysis
   - Builds timeline visualizations for temporal analysis

6. **Database Service** (`db_service.py`)
   - Manages PostgreSQL database connections and operations
   - Persists analysis sessions, detection clips, and object detections
   - Enables multi-night analysis and historical tracking
   - Handles data serialization and NaN value conversion

**Design Rationale**: The pipeline architecture separates concerns between video processing, feature engineering, and machine learning. This modular approach allows each stage to be optimized independently and makes the system maintainable and extensible.

**Alternative Considered**: Real-time visual classification using heuristic rules was attempted but abandoned due to noise amplification in low-light conditions and excessive false positives from star field movement.

**Pros of Current Approach**:
- Robust to visual noise and lighting conditions
- Unsupervised learning requires no labeled training data
- Statistical feature analysis is more reliable than pixel-based rules
- Scalable to large video archives

**Cons of Current Approach**:
- Requires multi-stage processing (not real-time)
- Feature engineering requires domain knowledge
- Unsupervised clusters may need manual interpretation

### Data Storage Solutions

**Hybrid Storage Architecture**: File-based + PostgreSQL database

**File-Based Storage**:
- **Input**: Video files uploaded through Streamlit interface (temporary storage, up to 5GB)
- **Output**: Organized directory structure with classified motion clips
  - `Satellite/`: Stable, linear orbital trajectories
  - `Meteor/`: High-speed, short-duration events
  - `Plane/`: Aircraft with predictable flight paths
  - `Junk/`: Noise, artifacts, and low-confidence detections
  - `Star/`: Stationary star field movement (if detected)
- **Temporary Files**: OpenCV processing requires temporary file creation for video analysis

**PostgreSQL Database** (`db_models.py`, `db_service.py`):
- **AnalysisSession**: Stores video metadata, processing parameters, and session info
- **DetectionClip**: Stores clip metadata, file paths, and processing results
- **ObjectDetection**: Stores individual object detections with features and classifications
- **Purpose**: Enables multi-night analysis, historical tracking, and data aggregation
- **Features**: Automatic NaN handling, JSON serialization, relationship management

**Design Rationale**: File-based storage handles video clips efficiently while the database enables:
1. Multi-night analysis and trend detection
2. Historical comparison of detections
3. Query-based filtering and analysis
4. Data persistence across sessions

### Authentication and Authorization

**Authentication**: Not implemented - single-user desktop/research tool

The application is designed as a standalone analysis tool without user authentication requirements. It's intended for use by individual researchers or small teams in controlled environments.

**Design Rationale**: Adding authentication would be premature optimization for the current use case (portfolio demonstration and research tool). If deployed as a multi-user service, authentication would be added using Streamlit's built-in auth mechanisms or OAuth integration.

### Machine Learning Model Architecture

**Unsupervised Learning Approach**: K-Means Clustering (simplified, conservative)

**K-Means Clustering**:
- Purpose: Automatic categorization of motion patterns
- Configuration: 3 clusters representing distinct object types (Satellite, Meteor, Plane)
- Features: 10-dimensional feature space (speed, consistency, trajectory metrics)
- Output: Cluster assignments with confidence scores based on trajectory quality

**Design Rationale**: Unsupervised learning was chosen because:
1. No labeled dataset exists for supervised training
2. Common objects (satellites, meteors, planes) have consistent kinematic signatures
3. System focuses on obvious detections rather than edge cases
4. Unsupervised approach adapts to new data patterns without retraining

**Feature Engineering Strategy**: The system extracts kinematic "flight signatures" rather than visual features because:
- Visual analysis is unreliable in low-light/noisy conditions
- Motion physics are more consistent than appearance
- Numerical features enable statistical analysis and ML processing
- Conservative thresholds minimize false positives

**Alternatives Considered**:
- CNN-based visual classification: Rejected due to lack of labeled data and noise sensitivity
- Rule-based heuristics: Rejected due to poor performance with real-world noise
- Isolation Forest anomaly detection: Removed due to excessive false positives

## External Dependencies

### Computer Vision Libraries
- **OpenCV (cv2)**: Core video processing, frame extraction, background subtraction (MOG2 algorithm)
- **NumPy**: Numerical operations, array processing, mathematical computations

### Machine Learning Libraries
- **scikit-learn**: 
  - K-Means clustering (`sklearn.cluster.KMeans`)
  - Feature scaling (`sklearn.preprocessing.StandardScaler`)
  - Dimensionality reduction (`sklearn.decomposition.PCA`)

### Web Framework & UI
- **Streamlit**: Web application framework, UI components, session state management
- **Plotly**: Interactive data visualization (charts, graphs, scatter plots)

### Data Processing
- **Pandas**: DataFrame operations, CSV handling, feature table management

### Utility Libraries
- **Python Standard Library**:
  - `os`, `shutil`: File and directory operations
  - `zipfile`: Archive creation for download packages
  - `io.BytesIO`: In-memory file handling
  - `datetime`: Timestamp generation
  - `collections.deque`, `collections.defaultdict`: Efficient data structures
  - `csv`: Metadata logging
  - `base64`: File encoding for downloads
  - `tempfile`: Temporary file management

### Hardware Considerations
- **Target Hardware**: Designed for Raspberry Pi Camera Module NoIR (low-light camera)
- **Processing Requirements**: CPU-based OpenCV and scikit-learn (no GPU required)
- **Storage**: Local filesystem for video input/output and temporary processing files

### Notable Design Decisions
- No database system required (file-based storage sufficient for batch processing)
- No external API integrations (self-contained analysis pipeline)
- No cloud services (designed for local/edge deployment)
- Future consideration: Integration with civilian aerospace research platforms for data sharing