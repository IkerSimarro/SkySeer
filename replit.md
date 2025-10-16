# SkySeer AI

## Overview

SkySeer is an advanced computer vision and machine learning system for detecting and classifying sky objects in night sky video footage. It focuses on identifying very obvious movements like satellite passes and meteor events, minimizing false positives. The system processes raw video into structured data through motion detection, numerical "flight signature" extraction, and K-Means clustering for categorization (satellites, meteors). It aggressively filters results to achieve less than 10 detections per typical video, prioritizing precision over recall. This project is ideal for amateur astronomy and low-light camera setups.

## Recent Changes

**October 16, 2025 - Latest:**

**Category Download Filtering & Classification Fixes:**
- **Category-Specific Video Filtering:** Fixed downloads to show ONLY objects of that classification
  - Satellite downloads now show only satellites (green boxes)
  - Meteor downloads now show only meteors (red boxes)
  - Junk downloads now show only junk (gray boxes)
  - Each category ZIP includes: filtered video + filtered CSV + filtered summary
  - Clean video preserved before annotation for proper filtering
- **Improved Satellite Detection:** Reduced misclassification of slow-moving satellites as Junk
  - Lowered forced-to-Junk threshold from 0.3 to 0.15 px/frame (allows distant satellites)
  - Reduced speed penalties: < 0.15 px/frame gets 30% penalty (was 20% at < 0.3)
  - Widened normal satellite range from 0.6-35 to 0.4-35 px/frame
  - More permissive classification reduces false Junk assignments

**Critical Video Processing Fixes:**
- **Precise 10x Speed Control:** Fixed output video duration to be exactly 1/10th of input duration
  - Formula: output_fps = 10 * fps / frame_skip ensures consistent speedup
  - 10-minute input now correctly produces 1-minute output
  - Works correctly regardless of frame_skip setting
- **Object ID Synchronization:** Fixed ID mismatch between video annotations and results table
  - Removed duplicate rectangle drawing during initial processing (was causing frame misalignment)
  - Metadata stores BOTH input frame numbers (for speed calculations) and output frame numbers (for video overlay)
  - Output frame numbers are 0-based to match video frame indices exactly
  - Video overlay now uses synchronized frame numbering
  - IDs in video now precisely match IDs in Available Objects table and Clip Extractor
- **Video Display Optimization:** Color-coded rectangles for all objects
  - GREEN boxes for Satellites with "ID:{number} Satellite" labels
  - RED boxes for Meteors with "ID:{number} Meteor" labels
  - GRAY boxes for Junk with "ID:{number} Junk" labels
  - All objects visible so users can identify which IDs to download
  - IDs match Available Objects table and Clip Extractor

**Improved Meteor Detection & Simplified Classification:**
- **Enhanced Meteor Scoring:** Optimized meteor detection with realistic thresholds
  - Speed-focused: >10 px/frame for meteor classification (lowered from 15 for better detection)
  - Extended duration: 1-4 seconds considered typical meteor range (up to 4s)
  - Reduced brightness bias: Meteors can be faint like satellites (max 1.5x bonus instead of 2x)
  - Linearity squared to heavily favor straight paths
  - Speed is the primary discriminator between satellites and meteors
- **Removed Plane Classification:** Simplified system to focus on satellites and meteors only
  - Plane detection was unreliable and rarely used
  - Updated ML classifier from 3 clusters to 2 clusters (Satellite/Meteor)
  - Removed all plane-related scoring and UI elements
  - Cleaner, more focused classification system

**Category-Specific Download Enhancement:**
- Fixed download functionality - now includes full sped-up video (10x speed) in all category downloads
- Each category download (Satellite, Meteor, Junk) contains:
  * Complete 1-minute video (for 10-minute input) showing ALL detections with color-coded boxes
  * CSV report filtered to show only that classification's objects
  * Category-specific SUMMARY.txt with filtered statistics
- Video shows all objects (user can see context): Green=Satellites, Red=Meteors, Gray=Junk
- Labels format: "ID:{number} {classification}" matching Available Objects table
- Improved recommended settings panel with clean metric cards and collapsible explanations

**October 15, 2025:**

**Critical Accuracy Improvements:**
- **Enhanced Small Satellite Detection:** Lowered minimum object size threshold from 40 to ~15 pixels to capture distant satellites
- **Improved Background Subtraction:** Reduced varThreshold from 60 to 45 for better detection of dim, small objects
- **Speed-Based Classification Filter:** Added speed validation to reduce false positives
  - Objects moving <0.3 px/frame are forced to Junk classification (eliminates stationary noise)
  - Objects moving 0.3-0.6 px/frame receive moderate penalties (40-60%)
  - Optimal satellite speed range: 0.6-35 px/frame with full scoring
  - Very fast objects (>35 px/frame) penalized to favor meteor classification
- **Balanced Satellite vs Plane Scoring:**
  - Satellites peak at 3-15s duration (1.4x boost), remain strong for longer passes (1.0x up to 25s)
  - Planes require EITHER blinking lights OR long duration (15+s) to score high
  - Planes without blinking get same speed penalties as satellites
  - Blinking bonus capped for slow objects to prevent tower light false positives
- **Result:** Balanced detection of slow-moving distant satellites while filtering stationary objects and properly discriminating satellites from planes

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