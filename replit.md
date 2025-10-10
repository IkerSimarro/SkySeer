# SkySeer AI Pipeline

## Overview

SkySeer is an advanced sky anomaly detection and UAP (Unidentified Aerial Phenomena) classification system that uses unsupervised machine learning to analyze long-duration astronomical video footage. The system processes video files to detect motion events, extract kinematic features, and automatically classify detected objects into categories (satellites, meteors, planes, or anomalies) without requiring labeled training data.

The pipeline transforms raw video into structured data by:
1. Detecting motion events using computer vision techniques
2. Extracting numerical "flight signatures" (speed, trajectory, consistency metrics)
3. Applying unsupervised ML (K-Means clustering and Isolation Forest) to automatically categorize detections
4. Flagging statistically rare events for human review

This system is designed for aerospace research and civilian anomaly detection projects, particularly suited for low-light camera equipment like Raspberry Pi NoIR modules.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack**: Streamlit web application framework

The application uses Streamlit for the user interface, providing:
- Web-based video upload and processing interface
- Real-time processing feedback with progress indicators
- Interactive data visualization using Plotly (charts and graphs)
- Session state management for maintaining processing results
- Downloadable results packages (ZIP files with classified clips)

**Design Pattern**: Single-page application with sidebar configuration panel

The interface is organized into:
- Main content area for video upload, processing controls, and results display
- Sidebar for configuration parameters (sensitivity, duration thresholds)
- Session state persistence to maintain results between interactions

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
   - Uses K-Means clustering (4 clusters) to group similar motion patterns
   - Implements Isolation Forest for anomaly detection
   - Assigns confidence scores and classifications to each detection

4. **Utility Functions** (`utils.py`)
   - Handles video metadata extraction
   - Manages file operations (ZIP creation, temporary files)
   - Provides formatting utilities for duration and file sizes

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

**File-Based Storage**: The system uses local filesystem storage

- **Input**: Video files uploaded through Streamlit interface (temporary storage)
- **Output**: Organized directory structure with classified motion clips
  - `0_ANOMALY_UAP_REVIEW/`: Statistically rare detections
  - `1_METEOR_EVENT/`: High-speed, short-duration events
  - `2_SATELLITE_ORBIT/`: Stable, linear trajectories
  - `3_PLANE_OR_JUNK/`: Common or low-priority detections
- **Metadata**: CSV files containing detection features and classifications
- **Temporary Files**: OpenCV processing requires temporary file creation for video analysis

**Design Rationale**: File-based storage is appropriate for this batch processing system. Video processing is compute-intensive rather than data-query intensive, so a database would add unnecessary complexity without performance benefits.

### Authentication and Authorization

**Authentication**: Not implemented - single-user desktop/research tool

The application is designed as a standalone analysis tool without user authentication requirements. It's intended for use by individual researchers or small teams in controlled environments.

**Design Rationale**: Adding authentication would be premature optimization for the current use case (portfolio demonstration and research tool). If deployed as a multi-user service, authentication would be added using Streamlit's built-in auth mechanisms or OAuth integration.

### Machine Learning Model Architecture

**Unsupervised Learning Approach**: K-Means + Isolation Forest ensemble

**K-Means Clustering**:
- Purpose: Automatic categorization of motion patterns
- Configuration: 4 clusters representing distinct object types
- Features: 10-dimensional feature space (speed, consistency, trajectory metrics)
- Output: Cluster assignments representing satellite/meteor/plane/other categories

**Isolation Forest**:
- Purpose: Anomaly detection for rare/unusual events
- Configuration: 10% contamination rate (expects ~10% anomalies)
- Features: Same 10-dimensional space as K-Means
- Output: Binary anomaly flag and anomaly score

**Design Rationale**: Unsupervised learning was chosen because:
1. No labeled UAP dataset exists for supervised training
2. Common objects (satellites, meteors) have consistent kinematic signatures
3. Anomalies are defined by statistical rarity rather than visual features
4. System adapts to new data patterns without retraining

**Feature Engineering Strategy**: The system extracts kinematic "flight signatures" rather than visual features because:
- Visual analysis is unreliable in low-light/noisy conditions
- Motion physics are more consistent than appearance
- Numerical features enable statistical analysis and ML processing

**Alternatives Considered**:
- CNN-based visual classification: Rejected due to lack of labeled data and noise sensitivity
- Rule-based heuristics: Rejected due to poor performance with real-world noise
- Deep learning time-series models (LSTM): Considered future enhancement but unnecessary complexity for current feature set

## External Dependencies

### Computer Vision Libraries
- **OpenCV (cv2)**: Core video processing, frame extraction, background subtraction (MOG2 algorithm)
- **NumPy**: Numerical operations, array processing, mathematical computations

### Machine Learning Libraries
- **scikit-learn**: 
  - K-Means clustering (`sklearn.cluster.KMeans`)
  - Isolation Forest anomaly detection (`sklearn.ensemble.IsolationForest`)
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