# =======================================================================
# SKYSEER AI PIPELINE: LOCAL PC EXECUTION
# Executes all stages: Video Harvester, Feature Pre-processing, and K-Means Clustering.
# Optimized for local CPU (like i7-7700) performance.
# =======================================================================

import cv2
import os
import shutil
import numpy as np
import pandas as pd
import csv
from collections import deque
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest # Added for Stage 5

# -----------------------------------------------------------------------
# --- CONFIGURATION (EDIT THESE LINES) ----------------------------------
# -----------------------------------------------------------------------

# 🚨 REQUIRED: Set the exact path/name of your video file here.
# NOTE: Using the long video name from your error message for now.
VIDEO_PATH = "long_night_sky.mp4" 

# Set the number of groups the AI should find (3 groups: Meteor, Satellite, Outlier)
NUMBER_OF_CLUSTERS = 3

# --- GLOBAL VARIABLES --------------------------------------------------

CLIPS_FOLDER = "motion_clips"
FRAMES_FOLDER = "motion_frames"
LOG_FILENAME = "motion_metadata.csv"
SUMMARY_DATA_FILE = "ai_summary_features.csv"
FINAL_OUTPUT_FILE = "ai_classified_clips.csv"
DATA_FILES = [LOG_FILENAME, SUMMARY_DATA_FILE, FINAL_OUTPUT_FILE]

# -----------------------------------------------------------------------
# --- STAGE 1: FOLDER RESET & SETUP -------------------------------------
# -----------------------------------------------------------------------

def reset_project_folders():
    """Clears old clips, folders, and data files for a clean run."""
    print("--- 🗑️ STAGE 1: Starting Project Reset ---")

    # Delete existing folders
    for folder in [CLIPS_FOLDER, FRAMES_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"✅ Deleted existing folder: '{folder}'")

    # Delete existing data files
    for filename in DATA_FILES:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"✅ Deleted old data file: '{filename}'")
    
    # Recreate empty clips folder
    os.makedirs(CLIPS_FOLDER, exist_ok=True)
    print(f"➕ Created new empty folder: '{CLIPS_FOLDER}'")

    print("--- Project Reset Complete. ---")

# -----------------------------------------------------------------------
# --- STAGE 2: CLIP HARVESTER & RAW DATA LOGGER -------------------------
# -----------------------------------------------------------------------

def run_harvester():
    """Processes video, saves clips, and logs raw frame-by-frame data."""
    print("\n--- 🎬 STAGE 2: Running Harvester & Data Logger ---")

    # 🚨 FIX 1: Video Capture MUST be inside the function.
    # 🚨 FIX 2: Use r"" and cv2.CAP_FFMPEG for robust reading on Windows/Anaconda.
    cap = cv2.VideoCapture(r"{}".format(VIDEO_PATH), cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise ValueError(f"ERROR: Video file '{VIDEO_PATH}' could not be opened. Check path or try a different video codec.")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps == 0:
        print("ERROR: Could not read video FPS. Check file integrity.")
        return

    # MOG2 parameters for timelapse/noise
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=64, detectShadows=False)

    # --- Initialization ---
    frame_count = 0
    clip_index = 0
    
    FRAME_SKIP_RATE = 5      # Process 1 frame out of every 5 (80% speed boost)
    min_frames_for_clip = int(fps * 1.0) # Minimum 1.0s clip length
    pre_buffer_frames = fps
    post_buffer_frames = fps
    STAR_NOISE_FRAMES = 5    # Motion must be persistent for 5 frames to be logged
    
    pre_motion_buffer = deque(maxlen=pre_buffer_frames)
    post_motion_buffer = deque()
    
    clip_frames_count = 0
    motion_active = False
    clip_writer = None
    false_motion_count = 0

    # AI Logging Setup
    metadata_file = open(LOG_FILENAME, 'w', newline='')
    csv_writer = csv.writer(metadata_file)
    csv_writer.writerow(["CLIP_ID", "FRAME_NUM", "CLIP_FRAME_COUNT", "MAX_AREA", 
                         "CENTROID_X", "CENTROID_Y", "ASPECT_RATIO"])
    current_clip_frames_data = [] 

    # Define Contour Filters
    MIN_CONTOUR_AREA = 10
    FRAME_AREA = frame_width * frame_height
    MAX_CONTOUR_AREA = int(FRAME_AREA * 0.005) if FRAME_AREA > 0 else 10000 

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            # Handle final clip cleanup on video end
            if motion_active and clip_writer is not None:
                if clip_frames_count >= min_frames_for_clip:
                    for f in post_motion_buffer: clip_writer.write(f)
                    csv_writer.writerows(current_clip_frames_data)
                
                # Release writer regardless of clip length
                clip_writer.release()
                if clip_frames_count < min_frames_for_clip:
                     os.remove(clip_filename)
            break

        frame_count += 1
        
        # Frame skipping optimization
        if frame_count % FRAME_SKIP_RATE != 0:
            pre_motion_buffer.append(frame.copy())
            if false_motion_count > 0:
                false_motion_count = max(0, false_motion_count - 1)
            continue

        # Processing steps (MOG2, Blur, Dilation)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
        fgmask = fgbg.apply(gray_blur)
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1,1),np.uint8) # FIX: Reduced kernel size
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Contour Detection and Filtering
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_motion_found = False 
        max_area_in_frame, best_centroid_x, best_centroid_y, best_aspect_ratio = 0, -1, -1, 0
        valid_contours = []
        
        for c in contours:
            area = cv2.contourArea(c)
            if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                raw_motion_found = True
                valid_contours.append(c)
                
                if area > max_area_in_frame:
                    max_area_in_frame = area
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        best_centroid_x = int(M["m10"] / M["m00"])
                        best_centroid_y = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(c)
                    best_aspect_ratio = w / h if h > 0 else 0
        
        # Star Filter Logic
        motion_detected = False
        if raw_motion_found:
            false_motion_count += 1
            if false_motion_count >= STAR_NOISE_FRAMES:
                motion_detected = True
        elif not motion_active:
            false_motion_count = 0 

        # Log Data and Draw Rectangles
        if motion_detected:
            # Draw visuals
            for c in valid_contours:
                x, y, w, h = cv2.boundingRect(c)
                pad = 8 
                cv2.rectangle(frame, (max(0, x-pad), max(0, y-pad)),
                              (min(frame.shape[1]-1, x+w+pad), min(frame.shape[0]-1, y+h+pad)),
                              (0, 0, 255), 2)
            
            # Log raw data for the frame
            current_clip_frames_data.append([
                -1, frame_count, clip_frames_count + 1, max_area_in_frame,
                best_centroid_x, best_centroid_y, best_aspect_ratio
            ])
            clip_frames_count += 1

        # Clip Writing Logic
        if motion_detected:
            if not motion_active:
                clip_index += 1
                clip_filename = os.path.join(CLIPS_FOLDER, f"clip_{clip_index:04d}.mp4")
                
                for row in current_clip_frames_data: row[0] = clip_index 

                clip_writer = cv2.VideoWriter(clip_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                motion_active = True
                for f in pre_motion_buffer: clip_writer.write(f)

            clip_writer.write(frame)
            post_motion_count = 0
            post_motion_buffer.clear()

        else:
            # Motion ended or not persistent
            if motion_active:
                post_motion_buffer.append(frame.copy())
                post_motion_count += 1
                
                if post_motion_count >= post_buffer_frames:
                    # Clip is ending, log data regardless of length
                    csv_writer.writerows(current_clip_frames_data)
                    
                    if clip_frames_count < min_frames_for_clip:
                        clip_writer.release()
                        os.remove(clip_filename)
                        clip_index -= 1
                    else:
                        for f in post_motion_buffer: clip_writer.write(f)
                        clip_writer.release()

                    clip_writer = None
                    motion_active = False
                    clip_frames_count = 0
                    post_motion_buffer.clear()
                    current_clip_frames_data.clear()
            else:
                 current_clip_frames_data.clear()

        pre_motion_buffer.append(frame.copy())


    cap.release()
    metadata_file.close() 
    print("--- Harvester Complete! ---")
    print(f"Total clips generated: {clip_index}")


# -----------------------------------------------------------------------
# --- STAGE 3: AI FEATURE PRE-PROCESSING --------------------------------
# -----------------------------------------------------------------------

def run_pre_processor():
    """Calculates Avg Speed, Consistency, and Duration per clip from raw data."""
    print("\n--- 🧠 STAGE 3: Running AI Feature Pre-processing ---")

    if not os.path.exists(LOG_FILENAME):
        print(f"ERROR: Raw data file '{LOG_FILENAME}' not found. Check Stage 2 output.")
        return

    df = pd.read_csv(LOG_FILENAME)
    if df.empty or len(df[df['CLIP_ID'] != -1]) == 0:
         print("ERROR: Loaded 0 valid data points. Cannot proceed.")
         return
         
    df = df[df['CENTROID_X'] != -1].copy()
    
    # Calculate Frame-to-Frame Metrics (Speed)
    df['PREV_X'] = df.groupby('CLIP_ID')['CENTROID_X'].shift(1)
    df['PREV_Y'] = df.groupby('CLIP_ID')['CENTROID_Y'].shift(1)
    df['PREV_X'] = df['PREV_X'].fillna(df['CENTROID_X'])
    df['PREV_Y'] = df['PREV_Y'].fillna(df['CENTROID_Y'])
    
    df['SPEED'] = np.sqrt(
        (df['CENTROID_X'] - df['PREV_X'])**2 + 
        (df['CENTROID_Y'] - df['PREV_Y'])**2
    )
    
    # Aggregate Features Per Clip
    summary_df = df.groupby('CLIP_ID').agg(
        Total_Duration_Frames=('CLIP_FRAME_COUNT', 'max'),
        Avg_Speed_Pixel_Per_Frame=('SPEED', 'mean'),
        Speed_Consistency_STD=('SPEED', 'std'),
        Max_Area_Overall=('MAX_AREA', 'max'),
        Avg_Aspect_Ratio=('ASPECT_RATIO', 'mean'), 
        Max_Aspect_Ratio=('ASPECT_RATIO', 'max'),
    ).reset_index()

    summary_df['Speed_Consistency_STD'] = summary_df['Speed_Consistency_STD'].fillna(0)
    
    summary_df.to_csv(SUMMARY_DATA_FILE, index=False)
    
    print(f"--- Pre-processing Complete! Generated {len(summary_df)} summary rows. ---")


# -----------------------------------------------------------------------
# --- STAGE 4: K-MEANS CLUSTERING ---------------------------------------
# -----------------------------------------------------------------------

def run_kmeans_clustering():
    """Uses K-Means to automatically group clips into primary motion categories."""
    print("\n--- 🤖 STAGE 4: Running K-Means Clustering ---")

    if not os.path.exists(SUMMARY_DATA_FILE):
        print(f"ERROR: Summary data file '{SUMMARY_DATA_FILE}' not found. Check Stage 3 output.")
        return

    df = pd.read_csv(SUMMARY_DATA_FILE)
    if df.empty:
        print("ERROR: Summary file is empty. Cannot run clustering.")
        return

    # Select Features
    feature_columns = [
        'Avg_Speed_Pixel_Per_Frame',
        'Total_Duration_Frames',
        'Speed_Consistency_STD',
        'Max_Area_Overall',
        'Max_Aspect_Ratio'
    ]
    
    X = df[feature_columns].values 

    # Data Scaling (Crucial)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Implementation
    kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init=10)
    df['CLUSTER_ID'] = kmeans.fit_predict(X_scaled)
    
    # Analyze Cluster Centers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_columns)
    cluster_centers['CLUSTER_ID'] = range(NUMBER_OF_CLUSTERS)
    
    avg_speed_global = df['Avg_Speed_Pixel_Per_Frame'].mean()
    avg_duration_global = df['Total_Duration_Frames'].mean()
    
    cluster_map = {}
    for index, row in cluster_centers.iterrows():
        if row['Avg_Speed_Pixel_Per_Frame'] > 1.5 * avg_speed_global and row['Total_Duration_Frames'] < avg_duration_global:
            cluster_map[row['CLUSTER_ID']] = '1_METEOR_EVENT'
        elif row['Total_Duration_Frames'] > 1.5 * avg_duration_global and row['Max_Area_Overall'] > 500: # Area heuristic for planes
            cluster_map[row['CLUSTER_ID']] = '3_PLANE_OR_JUNK'
        else:
             cluster_map[row['CLUSTER_ID']] = '2_SATELLITE_ORBIT'

    df['CLASSIFICATION'] = df['CLUSTER_ID'].map(cluster_map)
    df.to_csv(FINAL_OUTPUT_FILE, index=False)
    
    print("--- K-Means Clustering Complete! ---")
    print(f"File saved to: {FINAL_OUTPUT_FILE}")
    print("\nReview the cluster centers table for initial grouping insights:")
    print(cluster_centers)

# -----------------------------------------------------------------------
# --- STAGE 5: ISOLATION FOREST ANOMALY DETECTION -----------------------
# -----------------------------------------------------------------------

def run_anomaly_detection():
    """
    Loads classified clip features and uses Isolation Forest to identify clips 
    that are statistical outliers (potential UAPs or rare events).
    """
    print("\n--- 🤖 STAGE 5: Running Isolation Forest Anomaly Detection ---")

    if not os.path.exists(FINAL_OUTPUT_FILE):
        print(f"ERROR: Classified data file '{FINAL_OUTPUT_FILE}' not found. Check Stage 4 output.")
        return

    # 1. Load Classified Data (from K-Means output)
    df = pd.read_csv(FINAL_OUTPUT_FILE)
    if df.empty:
        print("ERROR: Classified file is empty. Cannot run anomaly detection.")
        return

    # 2. Select Features for Anomaly Detection (same features used for clustering)
    feature_columns = [
        'Avg_Speed_Pixel_Per_Frame', 
        'Total_Duration_Frames',     
        'Speed_Consistency_STD',     
        'Max_Area_Overall',          
        'Max_Aspect_Ratio'
    ]
    
    X = df[feature_columns].values 

    # 3. Data Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Implement Isolation Forest
    # contamination=0.01 expects 1% of the data to be outliers (UAPs/rare events).
    model = IsolationForest(contamination=0.01, random_state=42) 
    
    # Fit the model and predict the anomalies
    # -1 = Anomaly (Outlier), 1 = Inlier (Normal/Common)
    df['IS_ANOMALY'] = model.fit_predict(X_scaled)
    
    # 5. Add Final Classification Flag
    # If it's an anomaly, it overrides the K-Means classification for human review.
    df['FINAL_CLASSIFICATION'] = np.where(
        df['IS_ANOMALY'] == -1, '0_ANOMALY_UAP_REVIEW', df['CLASSIFICATION']
    )

    # 6. Save Final Data
    df.to_csv(FINAL_OUTPUT_FILE, index=False)
    
    anomaly_count = len(df[df['IS_ANOMALY'] == -1])
    
    print("--- Isolation Forest Analysis Complete! ---")
    print(f"Total ANOMALIES flagged for review: {anomaly_count}")
    print(f"Final classified data saved to: {FINAL_OUTPUT_FILE}")

# -----------------------------------------------------------------------
# --- STAGE 6: AUTOMATED CLIP ORGANIZATION ------------------------------
# -----------------------------------------------------------------------

def organize_clips():
    """Moves clips into folders based on the final AI classification."""
    print("\n--- 📂 STAGE 6: Organizing Clips into Folders ---")

    if not os.path.exists(FINAL_OUTPUT_FILE):
        print(f"ERROR: Final data file '{FINAL_OUTPUT_FILE}' not found.")
        return

    df = pd.read_csv(FINAL_OUTPUT_FILE)
    base_clip_folder = CLIPS_FOLDER # 'motion_clips'

    if not os.path.exists(base_clip_folder):
         print(f"ERROR: Base clips folder '{base_clip_folder}' not found. Did Stage 2 run?")
         return

    # Process each clip in the final dataset
    for index, row in df.iterrows():
        clip_id = row['CLIP_ID']
        classification = row['FINAL_CLASSIFICATION']
        
        # 1. Determine the source and destination paths
        source_filename = f"clip_{clip_id:04d}.mp4"
        # NOTE: os.path.join handles differences between Windows (\\) and Linux (/)
        source_path = os.path.join(base_clip_folder, source_filename)
        
        destination_folder = os.path.join(base_clip_folder, classification)
        destination_path = os.path.join(destination_folder, source_filename)
        
        # 2. Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)

        # 3. Move the file
        if os.path.exists(source_path):
            try:
                shutil.move(source_path, destination_path)
            except Exception as e:
                print(f"Warning: Could not move {source_filename}. Error: {e}")
        
    print("--- File Organization Complete! ---")
    print(f"Clips are now sorted into subfolders under: '{base_clip_folder}'")


# =======================================================================
# --- EXECUTION FLOW ----------------------------------------------------
# =======================================================================

if __name__ == "__main__":
    try:
        # Check if the mandatory video path is set
        if VIDEO_PATH == "your_video_file_name_here.mp4":
            print("\n🚨 ERROR: Please update the 'VIDEO_PATH' variable at the top of the script.")
        else:
            reset_project_folders()
            run_harvester()
            run_pre_processor()
            run_kmeans_clustering()
            run_anomaly_detection() 
            organize_clips()
            print("\n🎉 AI Pipeline complete! Your classified results are in 'ai_classified_clips.csv'.")
    except Exception as e:
        print(f"\n❌ A fatal error occurred during execution: {e}")
        print("Please check the video path, file existence, and ensure all required libraries (OpenCV, pandas, scikit-learn) are installed via Anaconda/pip.")
