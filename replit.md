# SkySeer AI

## Overview

SkySeer AI is an advanced computer vision and machine learning system designed for detecting and classifying sky objects such as satellites and meteors in night sky video footage. It processes raw video to identify movements, extracts numerical "flight signatures," and uses K-Means clustering for categorization, prioritizing precision to minimize false positives. The system includes trajectory prediction analysis for aerospace applications, comprehensive technical documentation of its ML pipeline, and professional PDF mission report generation. It aims to provide a robust solution for amateur astronomy and low-light camera setups.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

The application uses Streamlit to provide a web-based interface. It features a sidebar for configuration, allowing video upload (up to 5GB), real-time processing feedback, and interactive data visualizations (Plotly). The interface supports advanced trajectory visualization and enables the download of processed results.

### Backend Architecture

The system operates through a multi-stage sequential processing pipeline:

1.  **Video Ingestion**: Employs OpenCV for frame extraction and MOG2 background subtraction to detect motion and generate clips.
2.  **Feature Extraction**: Converts visual detections into 11-dimensional numerical "flight signatures" using kinematic metrics (speed, acceleration, trajectory linearity) and consistency scores.
3.  **ML Classification**: Utilizes StandardScaler for feature normalization and K-Means clustering (3 clusters) to categorize objects as Satellite, Meteor, or Junk, with confidence scores.
4.  **Utility Functions**: Manages video metadata, file operations (ZIP creation), and data formatting.
5.  **Trajectory Visualization**: Generates interactive path visualizations, speed heatmaps, polar plots, and timeline analyses, including predictive modeling for object trajectories.

The pipeline is designed for maintainability and extensibility, focusing on robustness to visual noise, unsupervised learning, and statistical feature analysis.

### Data Storage Solutions

A hybrid storage approach is used, combining file-based storage for input videos and classified output clips, and a PostgreSQL database for `AnalysisSession`, `DetectionClip`, and `ObjectDetection` data. This setup supports multi-night analysis and historical tracking.

### Machine Learning Model Architecture

The system employs an unsupervised K-Means Clustering model (3 clusters) for automatic categorization. It operates on an 11-dimensional feature space derived from kinematic and brightness patterns, which are more reliable in low-light conditions. This approach eliminates the need for labeled training data and adapts to new patterns automatically.

## External Dependencies

### Computer Vision Libraries

-   **OpenCV (cv2)**: For video processing, frame extraction, and background subtraction.
-   **NumPy**: For numerical operations.

### Machine Learning Libraries

-   **scikit-learn**: For K-Means clustering and feature scaling.

### Web Framework & UI

-   **Streamlit**: For the web application framework and UI components.
-   **Plotly**: For interactive data visualization.

### Data Processing

-   **Pandas**: For DataFrame operations.

### Utility Libraries

-   **ReportLab**: Professional PDF generation for mission reports.
-   **Python Standard Library**: For file/directory operations, archive creation, in-memory file handling, timestamp generation, efficient data structures, CSV handling, encoding, and temporary files.