import streamlit as st
import os
import cv2
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import zipfile
import shutil
from datetime import datetime
import base64

# Import custom modules
from video_processor import VideoProcessor
from feature_extractor import FeatureExtractor
from ml_classifier import MLClassifier
from utils import create_download_zip, format_duration, get_video_info, recommend_settings
from db_service import DatabaseService
from trajectory_visualizer import TrajectoryVisualizer
from trajectory_analysis import (
    analyze_all_trajectories, 
    create_trajectory_comparison_plot,
    create_trajectory_error_plot,
    get_trajectory_summary_stats
)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Configure page
st.set_page_config(
    page_title="SkySeer AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'processed_clips' not in st.session_state:
    st.session_state.processed_clips = []
if 'db_service' not in st.session_state:
    st.session_state.db_service = DatabaseService()
if 'metadata' not in st.session_state:
    st.session_state.metadata = []
if 'video_info' not in st.session_state:
    st.session_state.video_info = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'uploaded_video_path' not in st.session_state:
    st.session_state.uploaded_video_path = None

def main():
    st.title("🌌 SkySeer AI")
    st.markdown("**Advanced Sky Object Detection & Classification System**")
    
    # How It Works section
    st.success("""
    **🔍 How Does It Work?**
    
    SkySeer uses a three-step process to find satellites and meteors in your videos:
    
    1. **Motion Detection** - Computer vision scans your video frame-by-frame to spot 
       objects moving across the sky, filtering out background stars and noise.
    
    2. **Feature Extraction** - Each detected object is analyzed to measure its speed, 
       trajectory, brightness, and consistency, creating a unique "flight signature."
    
    3. **Smart Classification** - Machine learning algorithms categorize objects as 
       satellites, meteors, or noise based on their flight patterns.
    
    The system is designed to only catch **very obvious movement**, minimizing false 
    positives and giving you clean, reliable results!
    """)
    
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Motion detection sensitivity
        sensitivity = st.slider(
            "Motion Detection Sensitivity",
            min_value=1,
            max_value=10,
            value=5,
            help="Controls how small of an object can be detected. Higher = detects smaller/fainter satellites. Lower = only bright, obvious objects. Increase if missing small satellites."
        )
        
        # Minimum clip duration
        min_duration = st.slider(
            "Minimum Clip Duration (seconds)",
            min_value=0.3,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Filters out quick flashes and noise. Objects must be visible for at least this long to be saved. Increase to reduce false positives from camera noise or birds."
        )
        
        # Maximum clip duration (always enabled)
        max_duration = st.slider(
            "Maximum Clip Duration (seconds)",
            min_value=5.0,
            max_value=120.0,
            value=15.0,
            step=1.0,
            help="Filters out stationary objects like stars (which appear to move due to Earth's rotation). Objects visible longer than this are discarded. Most satellites cross in 5-30 seconds."
        )
        
        # Frame skip rate for performance
        frame_skip = st.slider(
            "Frame Skip Rate",
            min_value=1,
            max_value=10,
            value=3,
            help="Processes every Nth frame to speed up analysis. Higher = faster processing but may miss very fast meteors. Use 1 for high accuracy, 3-5 for balanced, 7+ for long videos."
        )
        
        st.markdown("---")
        
        # Settings recommendations (always visible, updates after video upload)
        st.markdown("### 💡 Recommended Settings")
        
        # Use stored recommendations or defaults
        if st.session_state.recommendations:
            rec = st.session_state.recommendations
            # Format values for display
            sensitivity_val = rec['sensitivity']
            min_dur_val = f"{rec['min_duration']}s"
            max_dur_val = f"{rec['max_duration']}s"
            frame_skip_val = rec['frame_skip']
        else:
            # Default recommendations before video upload - show dashes
            rec = {
                'explanations': ['Upload a video to see customized recommendations']
            }
            sensitivity_val = "-"
            min_dur_val = "-"
            max_dur_val = "-"
            frame_skip_val = "-"
        
        # Create a clean grid layout for recommended values
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.metric(label="Sensitivity", value=sensitivity_val)
            st.metric(label="Min Duration", value=min_dur_val)
        
        with rec_col2:
            st.metric(label="Max Duration", value=max_dur_val)
            st.metric(label="Frame Skip", value=frame_skip_val)
        
        # Show explanations in an expander for a cleaner look
        if rec['explanations']:
            with st.expander("ℹ️ Why these settings?"):
                for explanation in rec['explanations']:
                    st.markdown(f"• {explanation}")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Video Upload")
        
        # File uploader with drag and drop
        uploaded_file = st.file_uploader(
            "Drop your night sky video here or click to browse",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV. Best results with stable, tripod-mounted footage."
        )
        
        st.warning("⚠️ **Important for videos >3 minutes:** Use the recommended Frame Skip settings shown in the sidebar after uploading. Higher frame skip (5-6) prevents connection timeouts during processing.")
        
        if uploaded_file is not None:
            # Display video info and generate recommendations
            with st.expander("📊 Video Information", expanded=True):
                video_info = get_video_info(uploaded_file)
                
                # Generate recommendations based on video properties and content analysis
                if 'error' not in video_info:
                    # Check if recommendations need to be generated
                    previous_recommendations = st.session_state.recommendations
                    st.session_state.recommendations = recommend_settings(video_info, uploaded_file)
                    
                    # Trigger rerun if recommendations just changed from None to actual values
                    if previous_recommendations is None and st.session_state.recommendations is not None:
                        st.rerun()
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Duration", video_info.get('duration', 'Unknown'))
                    st.metric("FPS", video_info.get('fps', 'Unknown'))
                
                with col_b:
                    st.metric("Resolution", video_info.get('resolution', 'Unknown'))
                    st.metric("File Size", video_info.get('file_size', 'Unknown'))
                
                with col_c:
                    st.metric("Format", video_info.get('format', 'Unknown'))
                    st.metric("Bitrate", video_info.get('bitrate', 'Unknown'))
            
            # Process video button
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                process_video(uploaded_file, sensitivity, min_duration, max_duration, frame_skip)
    
    with col2:
        st.header("📈 Processing Status")
        
        if not st.session_state.processing_complete:
            st.info("Upload a video to begin analysis")
            
            # Processing pipeline visualization
            st.markdown("**Analysis Pipeline:**")
            st.markdown("1. 🎥 Motion Detection")
            st.markdown("2. 📊 Feature Extraction") 
            st.markdown("3. 🧠 ML Classification")
            st.markdown("4. 📋 Results Generation")
        else:
            st.success("✅ Processing Complete!")
            
            if st.button("🔄 Process New Video", use_container_width=True):
                reset_session()
                st.rerun()
    
    # FAQ / Troubleshooting Section - Under video upload
    st.markdown("---")
    st.info("**❓ Quick Troubleshooting Tips**")
    
    col_faq1, col_faq2 = st.columns(2)
    
    with col_faq1:
        with st.expander("🔍 Too many false detections?"):
            st.markdown("""
            - Lower sensitivity to 2-4
            - Increase minimum duration to 2-3s
            - Check footage quality
            """)
        
        with st.expander("⏱️ Processing too slow?"):
            st.markdown("""
            - Increase frame skip to 5-6
            - Use recommended settings
            """)
    
    with col_faq2:
        with st.expander("😕 Missing satellites?"):
            st.markdown("""
            **If satellites are not being detected:**
            - **Increase sensitivity to 6-7** (detects smaller, dimmer objects)
            - **Decrease min duration to 1.0-1.5s** (catches brief passes)
            - **Lower frame skip to 2-3** (processes more frames)
            - **Increase max duration to 25-30s** (allows slower-moving satellites)
            
            **Note:** The system is optimized to detect obvious satellite passes. Very faint or very slow satellites may still be missed to avoid false positives.
            """)
        
        with st.expander("🎨 Color codes?"):
            st.markdown("""
            - 🔴 **RED** = Satellites
            - 🟡 **YELLOW** = Meteors
            """)

    # Results section
    if st.session_state.processing_complete and st.session_state.results_data is not None:
        display_results()

def process_video(uploaded_file, sensitivity, min_duration, max_duration, frame_skip):
    """Process the uploaded video through the complete pipeline"""
    
    # Create temporary file
    temp_path = f"temp_uploads/{uploaded_file.name}"
    os.makedirs("temp_uploads", exist_ok=True)
    os.makedirs("processed_clips", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Get video info for database
    video_info = get_video_info(uploaded_file)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Stage 1: Motion Detection
        status_text.text("🎥 Stage 1/4: Detecting motion in video...")
        progress_bar.progress(10)
        
        processor = VideoProcessor(
            sensitivity=sensitivity,
            min_duration=min_duration,
            max_duration=max_duration,
            frame_skip=frame_skip
        )
        
        # Progress callback to keep connection alive during long processing
        callback_count = [0]
        
        def video_progress_callback(current_frame, total_frames):
            callback_count[0] += 1
            
            # Calculate actual progress percentage
            frame_percent = (current_frame / max(total_frames, 1)) * 100
            stage_progress = 10 + min(int(frame_percent * 0.2), 20)  # 10-30%
            
            # Update progress bar
            progress_bar.progress(stage_progress)
            
            # CRITICAL: Update status text with unique content to prevent timeout
            # This keeps WebSocket alive even if progress bar value repeats
            status_text.text(f"🎥 Stage 1/4: Detecting motion... ({callback_count[0]} updates, frame {current_frame})")
        
        motion_clips, metadata = processor.process_video(temp_path, progress_callback=video_progress_callback)
        progress_bar.progress(30)
        
        if not motion_clips:
            st.error("No motion detected in video. Try lowering sensitivity or check video quality.")
            return
        
        if not metadata:
            max_filter_msg = f"\n- Objects appear for more than {max_duration} seconds (maximum duration filter enabled)" if max_duration else ""
            max_solution_msg = "\n- Disable or increase the maximum duration filter" if max_duration else ""
            
            st.error(f"""⚠️ Motion detected but all objects were filtered out!
            
**Possible causes:**
- Objects appear for less than {min_duration} seconds (current minimum duration){max_filter_msg}
- Objects are intermittent or flickering

**Solutions:**
- Lower the minimum duration threshold (try 0.3 or 0.5 seconds)
- Increase sensitivity to capture more motion{max_solution_msg}
- Check if video has stable objects that move continuously
            """)
            return
        
        st.success(f"✅ Detected {len(motion_clips)} motion events with {len(set([m['clip_id'] for m in metadata]))} tracked objects")
        
        # Stage 2: Feature Extraction
        status_text.text("📊 Stage 2/4: Extracting movement features... (0/3 steps)")
        progress_bar.progress(40)
        
        extractor = FeatureExtractor()
        
        # Keep connection alive during feature extraction
        status_text.text("📊 Stage 2/4: Extracting movement features... (1/3 steps)")
        progress_bar.progress(50)
        
        features_df = extractor.extract_features(metadata)
        
        status_text.text("📊 Stage 2/4: Extracting movement features... (2/3 steps)")
        progress_bar.progress(60)
        
        # Extra keepalive
        status_text.text("📊 Stage 2/4: Extracting movement features... (3/3 steps)")
        progress_bar.progress(70)
        
        # Stage 3: ML Classification
        status_text.text("🧠 Stage 3/4: Classifying objects with AI... (processing)")
        progress_bar.progress(75)
        
        classifier = MLClassifier()
        
        # Keep connection alive during classification
        status_text.text("🧠 Stage 3/4: Classifying objects with AI... (analyzing patterns)")
        progress_bar.progress(82)
        
        results_df = classifier.classify_objects(features_df)
        
        status_text.text("🧠 Stage 3/4: Classifying objects with AI... (finalizing)")
        progress_bar.progress(90)
        
        # Stage 4: Add color-coded rectangles and generate results
        status_text.text("📋 Stage 4/4: Adding color-coded rectangles... (starting)")
        progress_bar.progress(92)
        
        # Add colored rectangles based on classification
        from utils import add_colored_rectangles_to_clips
        
        # Progress callback for rectangle drawing to prevent timeout
        rectangle_callback_count = [0]
        
        def rectangle_progress_callback(current_clip, total_clips):
            rectangle_callback_count[0] += 1
            percent = int((current_clip / max(total_clips, 1)) * 100)
            
            # Update progress bar (92-97% range for this stage)
            stage_progress = 92 + min(int(percent * 0.05), 5)
            progress_bar.progress(stage_progress)
            
            # CRITICAL: Update status with unique text to prevent WebSocket timeout
            status_text.text(f"📋 Stage 4/4: Adding rectangles... (clip {current_clip}/{total_clips}, {percent}%)")
        
        motion_clips = add_colored_rectangles_to_clips(
            motion_clips, metadata, results_df, 
            progress_callback=rectangle_progress_callback
        )
        
        status_text.text("📋 Stage 4/4: Rectangles complete, finalizing...")
        progress_bar.progress(97)
        
        status_text.text("📋 Stage 4/4: Finalizing results...")
        
        # Store results in session state
        st.session_state.results_data = results_df
        st.session_state.processed_clips = motion_clips
        st.session_state.metadata = metadata
        st.session_state.video_info = video_info
        st.session_state.processing_complete = True
        
        # Save to database
        try:
            session_id = st.session_state.db_service.save_analysis_session(
                video_info, results_df, motion_clips
            )
            if session_id:
                st.session_state.current_session_id = session_id
        except Exception as e:
            st.warning(f"Could not save to database: {e}")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")
        
        # Store video path for clip extraction (will be cleaned up on reset)
        st.session_state.uploaded_video_path = temp_path
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def display_results():
    """Display the analysis results with interactive dashboard"""
    
    st.header("🎯 Detection Results")
    
    results_df = st.session_state.results_data
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_detections = len(results_df)
        st.metric("Total Detections", total_detections)
    
    with col2:
        satellites = len(results_df[results_df['classification'] == 'Satellite'])
        st.metric("🛰️ Satellites", satellites)
    
    with col3:
        meteors = len(results_df[results_df['classification'] == 'Meteor'])
        st.metric("☄️ Meteors", meteors)
    
    # Classification distribution chart
    st.subheader("📊 Classification Distribution")
    
    classification_counts = results_df['classification'].value_counts()
    
    fig_pie = px.pie(
        values=classification_counts.values,
        names=classification_counts.index,
        title="Object Classification Breakdown",
        color_discrete_map={
            'Star': '#ffd700',
            'Satellite': '#1f77b4',
            'Meteor': '#ff7f0e', 
            'Junk': '#d62728'
        }
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Feature analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏃 Speed Analysis")
        fig_speed = px.histogram(
            results_df,
            x='avg_speed',
            color='classification',
            title="Speed Distribution by Classification",
            labels={'avg_speed': 'Average Speed (pixels/frame)'}
        )
        st.plotly_chart(fig_speed, use_container_width=True)
    
    with col2:
        st.subheader("📏 Duration Analysis") 
        fig_duration = px.histogram(
            results_df,
            x='duration',
            color='classification',
            title="Duration Distribution by Classification",
            labels={'duration': 'Duration (seconds)'}
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    # Trajectory Visualization
    if st.session_state.metadata:
        st.subheader("🎯 Trajectory Analysis")
        
        # Create visualizer
        video_info = st.session_state.video_info
        width = video_info.get('width', 1920)
        height = video_info.get('height', 1080)
        fps = video_info.get('fps', 30)
        
        visualizer = TrajectoryVisualizer(frame_width=width, frame_height=height)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📍 Trajectories", "🌡️ Speed Heatmap", "🧭 Directions", "⏱️ Timeline"])
        
        with tab1:
            traj_fig = visualizer.create_trajectory_plot(st.session_state.metadata)
            st.plotly_chart(traj_fig, use_container_width=True)
            st.caption("Interactive trajectory plot showing object paths. Green stars = start, Red X = end")
        
        with tab2:
            heatmap_fig = visualizer.create_speed_heatmap(st.session_state.metadata)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.caption("Heatmap showing average object speed across different regions of the frame")
        
        with tab3:
            direction_fig = visualizer.create_direction_plot(st.session_state.metadata)
            st.plotly_chart(direction_fig, use_container_width=True)
            st.caption("Polar plot showing the distribution of object movement directions")
        
        with tab4:
            timeline_fig = visualizer.create_timeline_plot(st.session_state.metadata, fps=fps)
            st.plotly_chart(timeline_fig, use_container_width=True)
            st.caption("Timeline showing when objects were detected throughout the video")
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_classifications = st.multiselect(
            "Filter by Classification",
            options=results_df['classification'].unique(),
            default=results_df['classification'].unique()
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=['confidence', 'avg_speed', 'duration'],
            index=0
        )
    
    # Apply filters
    filtered_df = results_df[
        (results_df['classification'].isin(selected_classifications)) &
        (results_df['confidence'] >= min_confidence)
    ].sort_values(sort_by, ascending=False)
    
    # Display filtered table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            'clip_id': 'Clip ID',
            'classification': 'Classification',
            'confidence': st.column_config.ProgressColumn(
                'Confidence',
                min_value=0,
                max_value=1,
                format='%.2f'
            ),
            'avg_speed': 'Avg Speed',
            'speed_consistency': 'Speed Consistency', 
            'duration': 'Duration (s)'
        }
    )
    
    # Download section
    st.subheader("📥 Download Results")
    
    # CSV download (always available)
    csv_buffer = BytesIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    st.download_button(
        label="📊 Download CSV Report",
        data=csv_buffer,
        file_name=f"skyseer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Classification-specific clip downloads
    st.markdown("**Download Video Clips by Category:**")
    
    # Get unique classifications with counts (only show Satellite and Meteor)
    classification_counts = results_df['classification'].value_counts()
    # Filter to only show Satellite and Meteor in downloads
    classification_counts = classification_counts[classification_counts.index.isin(['Satellite', 'Meteor'])]
    
    # Create emoji mapping (only Satellite and Meteor)
    emoji_map = {
        'Satellite': '🛰️',
        'Meteor': '☄️'
    }
    
    # Initialize session state for prepared downloads
    if 'prepared_zips' not in st.session_state:
        st.session_state.prepared_zips = {}
    
    # Create download buttons for Satellite and Meteor only
    cols = st.columns(min(len(classification_counts), 2)) if len(classification_counts) > 0 else st.columns(1)
    
    for idx, (classification, count) in enumerate(classification_counts.items()):
        with cols[idx % len(cols)]:
            emoji = emoji_map.get(classification, '📦')
            
            # Check if this classification's ZIP is already prepared
            if classification in st.session_state.prepared_zips:
                # Show download button
                st.download_button(
                    label=f"📦 Download {classification} ZIP",
                    data=st.session_state.prepared_zips[classification],
                    file_name=f"skyseer_{classification.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key=f"download_btn_{classification}"
                )
            else:
                # Show prepare button
                if st.button(
                    f"{emoji} {classification} ({count})",
                    key=f"prepare_{classification}",
                    use_container_width=True
                ):
                    # Prepare the ZIP with simple loading
                    with st.spinner(""):
                        zip_buffer = create_download_zip(
                            st.session_state.processed_clips, 
                            results_df,
                            classification_filter=classification,
                            metadata=st.session_state.metadata
                        )
                        # Convert BytesIO to bytes for session state storage
                        st.session_state.prepared_zips[classification] = zip_buffer.getvalue()
                        st.rerun()
    
    # Object Clip Extractor Section
    st.markdown("---")
    st.subheader("🎬 Object Clip Extractor")
    st.markdown("Extract normal-speed clips of specific detected objects from the original video.")
    
    # Check if we have the uploaded video available
    if st.session_state.uploaded_video_path and os.path.exists(st.session_state.uploaded_video_path):
        # Quick reference table
        st.markdown("**Available Objects:**")
        
        # Create reference table with key info
        ref_data = []
        for _, row in results_df.iterrows():
            ref_data.append({
                'ID': int(row['clip_id']),
                'Classification': row['classification'],
                'Confidence': f"{row['confidence']*100:.0f}%",
                'Duration': f"{row['duration']:.1f}s",
                'Avg Speed': f"{row['avg_speed']:.1f} px/f"
            })
        
        ref_df = pd.DataFrame(ref_data)
        st.dataframe(ref_df, use_container_width=True, hide_index=True)
        
        # Input field for object IDs
        st.markdown("**Enter Object ID(s):**")
        col_input, col_button = st.columns([3, 1])
        
        with col_input:
            object_input = st.text_input(
                "Enter single ID (e.g., '3') or multiple IDs separated by commas (e.g., '3, 7, 12')",
                key="object_id_input",
                label_visibility="collapsed"
            )
        
        with col_button:
            extract_button = st.button("Extract Clip", use_container_width=True, type="primary")
        
        # Process extraction when button is clicked
        if extract_button and object_input:
            try:
                # Parse input IDs
                object_ids = [int(x.strip()) for x in object_input.split(',')]
                
                # Validate IDs
                valid_ids = results_df['clip_id'].unique().tolist()
                invalid_ids = [oid for oid in object_ids if oid not in valid_ids]
                
                if invalid_ids:
                    st.error(f"❌ Invalid Object ID(s): {', '.join(map(str, invalid_ids))}. Valid range: {min(valid_ids)}-{max(valid_ids)}")
                else:
                    with st.spinner(f"Extracting clip for Object(s): {', '.join(map(str, object_ids))}..."):
                        # Extract the clip
                        clip_path = extract_object_clip(
                            object_ids,
                            st.session_state.uploaded_video_path,
                            st.session_state.metadata,
                            results_df
                        )
                        
                        if clip_path and os.path.exists(clip_path):
                            # Read the clip file
                            with open(clip_path, 'rb') as f:
                                clip_data = f.read()
                            
                            # Generate filename
                            if len(object_ids) == 1:
                                filename = f"Object_{object_ids[0]}_clip.mp4"
                            else:
                                filename = f"Objects_{'_'.join(map(str, object_ids))}_clip.mp4"
                            
                            # Display success message
                            st.success(f"✅ Clip extracted successfully for {len(object_ids)} object(s)!")
                            
                            # Show object details
                            for obj_id in object_ids:
                                obj_row = results_df[results_df['clip_id'] == obj_id].iloc[0]
                                st.info(
                                    f"**Object {obj_id}:** {obj_row['classification']} | "
                                    f"Confidence: {obj_row['confidence']*100:.0f}% | "
                                    f"Duration: {obj_row['duration']:.1f}s | "
                                    f"Speed: {obj_row['avg_speed']:.1f} px/frame"
                                )
                            
                            # Download button
                            st.download_button(
                                label=f"📥 Download {filename}",
                                data=clip_data,
                                file_name=filename,
                                mime="video/mp4",
                                use_container_width=True
                            )
                            
                            # Clean up temp file
                            try:
                                os.remove(clip_path)
                            except:
                                pass
                        else:
                            st.error("❌ Failed to extract clip. Please try again.")
                            
            except ValueError:
                st.error("❌ Invalid input format. Please enter numeric IDs separated by commas (e.g., '3, 7, 12')")
            except Exception as e:
                st.error(f"❌ Error extracting clip: {str(e)}")
    else:
        st.warning("⚠️ Original video file not available. Please process a new video to use this feature.")
    
    # Trajectory Prediction Analysis Section
    st.markdown("---")
    st.subheader("🎯 Trajectory Prediction Analysis")
    st.markdown("Demonstrates predictive modeling capabilities - comparing actual object paths vs. predicted trajectories using linear regression.")
    
    # Initialize trajectory_results outside the expander
    trajectory_results = None
    
    with st.expander("📊 View Trajectory Predictions", expanded=False):
        # Prepare detection data for trajectory analysis
        detections_list = []
        for meta in st.session_state.metadata:
            detections_list.append({
                'object_id': meta['clip_id'],
                'frame_number': meta['frame_number'],
                'center_x': meta['centroid_x'],
                'center_y': meta['centroid_y']
            })
        
        if detections_list:
            detections_df = pd.DataFrame(detections_list)
            
            # Analyze all trajectories
            trajectory_results = analyze_all_trajectories(detections_df, method='linear')
            
            if trajectory_results:
                # Show summary statistics
                summary = get_trajectory_summary_stats(trajectory_results)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Objects Analyzed", summary['total_objects'])
                with col2:
                    st.metric("Avg Prediction Error", f"{summary['avg_mean_error']:.2f} px")
                with col3:
                    st.metric("Avg R² Score", f"{summary['avg_r2_x']:.3f}")
                with col4:
                    st.metric("Highly Predictable", f"{summary['highly_predictable']}/{summary['total_objects']}")
                
                st.caption("**Interpretation:** R² > 0.95 indicates highly linear motion (typical for satellites). Lower R² suggests curved/irregular paths (meteors, aircraft maneuvers).")
                
                # Show error plot
                error_fig = create_trajectory_error_plot(trajectory_results)
                if error_fig:
                    st.plotly_chart(error_fig, use_container_width=True)
                
                # Allow user to select specific objects for detailed view
                st.markdown("**Detailed Trajectory Comparison:**")
                selected_obj = st.selectbox(
                    "Select object to view prediction details",
                    options=[r['object_id'] for r in trajectory_results],
                    format_func=lambda x: f"Object #{x}"
                )
                
                if selected_obj:
                    selected_result = next(r for r in trajectory_results if r['object_id'] == selected_obj)
                    
                    # Get classification for this object
                    obj_classification = results_df[results_df['clip_id'] == selected_obj]['classification'].iloc[0]
                    
                    # Show detailed comparison plot
                    comparison_fig = create_trajectory_comparison_plot(selected_result, obj_classification)
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Show prediction metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Error", f"{selected_result['mean_error']:.2f} px")
                    with col2:
                        st.metric("Max Error", f"{selected_result['max_error']:.2f} px")
                    with col3:
                        st.metric("RMSE", f"{selected_result['rmse_total']:.2f} px")
                    
                    st.info(f"✨ **Technical Insight:** This object's trajectory has an R² score of {selected_result['r2_x']:.3f}, "
                           f"indicating {'highly' if selected_result['r2_x'] > 0.95 else 'moderately'} predictable linear motion. "
                           f"{'This is characteristic of satellite passes.' if selected_result['r2_x'] > 0.95 else 'This suggests non-linear or irregular movement patterns.'}")
        else:
            st.info("No trajectory data available for analysis.")
    
    # Technical Details Section
    st.markdown("---")
    st.subheader("📚 Technical Details")
    st.markdown("Deep dive into the machine learning pipeline and computer vision techniques used in SkySeer.")
    
    with st.expander("🔬 View ML Pipeline Architecture", expanded=False):
        st.markdown("""
        ### **System Architecture Overview**
        
        SkySeer employs a sophisticated multi-stage pipeline combining computer vision and unsupervised machine learning:
        
        ---
        
        #### **Stage 1: Motion Detection**
        - **Algorithm:** MOG2 (Mixture of Gaussians) Background Subtraction
        - **Purpose:** Identify moving objects against the static night sky
        - **Parameters:** Adaptive variance threshold (45), history frames (500)
        - **Output:** Motion blobs with bounding boxes and frame-by-frame tracking
        
        #### **Stage 2: Feature Extraction (11-Dimensional Feature Space)**
        
        Each detected object is transformed into a numerical "flight signature" consisting of:
        
        1. **avg_speed** - Average velocity in pixels/frame
        2. **speed_consistency** - Standard deviation of speed (0-1, lower = more consistent)
        3. **duration** - Total observation time in seconds
        4. **linearity** - How straight the path is (0-1, higher = straighter)
        5. **direction_changes** - Number of significant heading changes
        6. **size_consistency** - Variation in object size across frames
        7. **acceleration** - Rate of speed change
        8. **blinking_score** - Detection of periodic brightness changes (aircraft lights)
        9. **satellite_score** - Weighted score favoring satellite characteristics
        10. **meteor_score** - Weighted score favoring meteor characteristics
        11. **avg_brightness** - Average pixel intensity
        
        **Why not visual features?** In low-light conditions, kinematic patterns are more reliable than pixel-level features.
        
        ---
        
        #### **Stage 3: ML Classification (K-Means Clustering)**
        
        - **Algorithm:** K-Means Unsupervised Clustering
        - **Number of Clusters:** 2 (Satellite, Meteor)
        - **Preprocessing:** StandardScaler normalization on all 11 features
        - **Classification Logic:**
          - Cluster centers are analyzed for characteristic patterns
          - **Satellites:** Moderate speed (0.6-35 px/frame), high linearity, long duration (3-25s)
          - **Meteors:** High speed (>10 px/frame), short duration (1-4s), high linearity
          - Objects not matching either pattern are classified as Junk
        
        **Why K-Means?**
        - No labeled training data required
        - Adapts automatically to new motion patterns
        - Computationally efficient for real-time analysis
        
        ---
        
        #### **Stage 4: Aggressive Filtering**
        
        - **Speed Validation:** Objects <0.3 px/frame → Junk (stationary noise)
        - **Target:** <10 detections per video (precision over recall)
        - **Confidence Scoring:** Distance from cluster center determines confidence (0-100%)
        
        ---
        
        ### **Performance Characteristics**
        
        **Strengths:**
        - Excellent at detecting obvious satellite passes and meteor streaks
        - Minimal false positives due to aggressive filtering
        - No training data dependency
        
        **Limitations:**
        - May miss very faint or very slow-moving objects
        - Optimized for tripod-mounted, stable footage
        - Best results with clear, dark skies
        
        ---
        
        ### **Video Processing**
        
        - **Frame Skip:** Adaptive (3-6) based on video length and FPS
        - **Output Speed:** 10x speedup for efficient review
        - **Color Coding:** RED = Satellites, YELLOW = Meteors
        
        ---
        
        ### **Technical Stack**
        
        - **Computer Vision:** OpenCV (MOG2, contour detection)
        - **Machine Learning:** scikit-learn (K-Means, StandardScaler)
        - **Visualization:** Plotly (interactive charts), Streamlit (web interface)
        - **Data Processing:** NumPy, Pandas
        - **Database:** PostgreSQL (multi-night tracking capability)
        """)
        
        # Show actual feature distribution
        st.markdown("### **Feature Distribution in Current Dataset**")
        st.markdown("These box plots show how different features separate Satellites from Meteors in your video:")
        
        feature_info = {
            'avg_speed': {
                'title': 'Average Speed (pixels/frame)',
                'explanation': '**What to look for:** Meteors typically move much faster (>10 px/frame) than satellites (0.6-35 px/frame). The higher the box, the faster the objects moved.'
            },
            'speed_consistency': {
                'title': 'Speed Consistency (0-1, lower = more consistent)',
                'explanation': '**What to look for:** Both satellites and meteors should have LOW values (boxes near 0), meaning constant speed. Higher values indicate erratic movement (likely junk).'
            },
            'duration': {
                'title': 'Duration (seconds)',
                'explanation': '**What to look for:** Satellites typically appear for 3-25 seconds (longer boxes), while meteors are brief flashes lasting 1-4 seconds (boxes near bottom).'
            },
            'linearity': {
                'title': 'Path Linearity (0-1, higher = straighter)',
                'explanation': '**What to look for:** Both satellites and meteors should have HIGH values (boxes near 1), meaning straight paths. Curved or zigzag paths suggest planes or noise.'
            }
        }
        
        available_features = [col for col in feature_info.keys() if col in results_df.columns]
        
        if available_features:
            for feature in available_features:
                info = feature_info[feature]
                
                fig = px.box(
                    results_df,
                    x='classification',
                    y=feature,
                    color='classification',
                    title=info['title'],
                    labels={feature: info['title']}
                )
                fig.update_layout(
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(info['explanation'])
    
    # PDF Mission Report Download
    st.markdown("---")
    st.subheader("📄 Export Mission Report")
    st.markdown("Generate a professional PDF report documenting all detections, technical parameters, and analysis results.")
    
    if st.button("📥 Generate PDF Mission Report", use_container_width=True):
        with st.spinner("Generating professional mission report..."):
            pdf_buffer = generate_mission_report_pdf(
                results_df,
                st.session_state.metadata,
                st.session_state.video_info,
                trajectory_results
            )
            
            if pdf_buffer:
                st.success("✅ Mission report generated successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"SkySeer_Mission_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("❌ Failed to generate PDF report.")

def generate_mission_report_pdf(results_df, metadata, video_info, trajectory_results=None):
    """
    Generate a professional PDF mission report
    
    Args:
        results_df: DataFrame with detection results
        metadata: List of detection metadata
        video_info: Dict with video information
        trajectory_results: Optional trajectory prediction results
        
    Returns:
        BytesIO buffer with PDF content
    """
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        title = Paragraph("SkySeer AI - Mission Report", title_style)
        story.append(title)
        
        timestamp = Paragraph(f"<para align=center>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</para>", 
                            styles["Normal"])
        story.append(timestamp)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Video Information", heading_style))
        total_frames = video_info.get('total_frames', 0)
        video_data = [
            ['Parameter', 'Value'],
            ['Filename', str(video_info.get('filename', 'N/A'))],
            ['Duration', video_info.get('duration', 'N/A')],
            ['Resolution', video_info.get('resolution', 'N/A')],
            ['FPS', str(video_info.get('fps', 'N/A'))],
            ['Total Frames', f"{total_frames:,}" if isinstance(total_frames, (int, float)) else str(total_frames)]
        ]
        
        video_table = Table(video_data, colWidths=[2*inch, 3.5*inch])
        video_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(video_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Detection Summary", heading_style))
        total_detections = len(results_df)
        satellites = len(results_df[results_df['classification'] == 'Satellite'])
        meteors = len(results_df[results_df['classification'] == 'Meteor'])
        
        summary_data = [
            ['Metric', 'Count'],
            ['Total Detections', str(total_detections)],
            ['Satellites', str(satellites)],
            ['Meteors', str(meteors)],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        if trajectory_results:
            story.append(Paragraph("Trajectory Analysis Summary", heading_style))
            summary = get_trajectory_summary_stats(trajectory_results)
            
            traj_data = [
                ['Metric', 'Value'],
                ['Objects Analyzed', str(summary['total_objects'])],
                ['Avg Prediction Error', f"{summary['avg_mean_error']:.2f} pixels"],
                ['Avg R² Score (X-axis)', f"{summary['avg_r2_x']:.3f}"],
                ['Highly Predictable Objects', f"{summary['highly_predictable']}/{summary['total_objects']}"],
            ]
            
            traj_table = Table(traj_data, colWidths=[3*inch, 2*inch])
            traj_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(traj_table)
            story.append(Spacer(1, 0.3*inch))
        
        story.append(PageBreak())
        story.append(Paragraph("Detailed Detection Results", heading_style))
        
        detection_data = [['ID', 'Classification', 'Confidence', 'Speed', 'Duration']]
        for _, row in results_df.iterrows():
            detection_data.append([
                str(int(row['clip_id'])),
                row['classification'],
                f"{row['confidence']*100:.0f}%",
                f"{row['avg_speed']:.1f} px/f",
                f"{row['duration']:.1f}s"
            ])
        
        det_table = Table(detection_data, colWidths=[0.6*inch, 1.3*inch, 1.1*inch, 1.1*inch, 1.1*inch])
        det_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(det_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(PageBreak())
        story.append(Paragraph("Technical Documentation", heading_style))
        
        tech_text = """
        <b>Detection Pipeline:</b><br/>
        1. MOG2 Background Subtraction for motion detection<br/>
        2. 11-dimensional feature extraction (speed, linearity, duration, etc.)<br/>
        3. K-Means clustering for unsupervised classification<br/>
        4. Aggressive filtering to minimize false positives<br/>
        <br/>
        <b>Classification Criteria:</b><br/>
        • Satellites: Moderate speed (0.6-35 px/frame), high linearity, 3-25s duration<br/>
        • Meteors: High speed (>10 px/frame), short duration (1-4s), high linearity<br/>
        • Junk: Objects not matching satellite/meteor patterns or speed <0.3 px/frame<br/>
        <br/>
        <b>System Characteristics:</b><br/>
        • Target: <10 detections per video (precision over recall)<br/>
        • Optimized for stable, tripod-mounted footage<br/>
        • Best results with clear, dark skies<br/>
        """
        
        tech_para = Paragraph(tech_text, styles['Normal'])
        story.append(tech_para)
        
        footer_text = f"""
        <para align=center>
        <br/><br/>
        ---<br/>
        <i>Generated by SkySeer AI - Advanced Sky Object Detection System</i><br/>
        <i>Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        </para>
        """
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(footer_text, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

def extract_object_clip(object_ids, video_path, metadata, results_df):
    """
    Extract clips for specified object IDs with overlays and bounding boxes.
    
    Args:
        object_ids: List of object IDs to extract
        video_path: Path to the original video file
        metadata: List of detection metadata
        results_df: DataFrame with classification results
    
    Returns:
        Path to the extracted clip video file
    """
    import tempfile
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate buffer frames
    buffer_before = int(2 * fps)  # 2 seconds before
    buffer_after = int(1 * fps)   # 1 second after
    
    # Collect all frame ranges for requested objects
    frame_segments = []
    object_info = {}
    
    for obj_id in object_ids:
        # Get metadata for this object
        obj_detections = [m for m in metadata if m.get('clip_id') == obj_id]
        if not obj_detections:
            continue
        
        # Get frame range
        frame_numbers = [d['frame_number'] for d in obj_detections]
        start_frame = min(frame_numbers)
        end_frame = max(frame_numbers)
        
        # Add buffers (respect video boundaries)
        buffered_start = max(0, start_frame - buffer_before)
        buffered_end = min(total_frames - 1, end_frame + buffer_after)
        
        frame_segments.append((buffered_start, buffered_end, obj_id))
        
        # Store object info for overlay
        obj_row = results_df[results_df['clip_id'] == obj_id].iloc[0]
        object_info[obj_id] = {
            'classification': obj_row['classification'],
            'confidence': obj_row['confidence'],
            'avg_speed': obj_row['avg_speed'],
            'detections': obj_detections
        }
    
    if not frame_segments:
        return None
    
    # Sort segments by start frame
    frame_segments.sort(key=lambda x: x[0])
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each segment
    for start_frame, end_frame, obj_id in frame_segments:
        obj = object_info[obj_id]
        
        # Set video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find if there's a detection at this frame for this object
            detection_at_frame = None
            for det in obj['detections']:
                if det['frame_number'] == frame_idx:
                    detection_at_frame = det
                    break
            
            # Draw bounding box if detection exists at this frame
            if detection_at_frame:
                x = detection_at_frame['bbox_x']
                y = detection_at_frame['bbox_y']
                w = detection_at_frame['bbox_width']
                h = detection_at_frame['bbox_height']
                
                # Draw rectangle with padding
                pad = 8
                cv2.rectangle(frame,
                            (max(0, x - pad), max(0, y - pad)),
                            (min(width - 1, x + w + pad), min(height - 1, y + h + pad)),
                            (0, 255, 0), 2)
            
            # Add text overlay in corner
            overlay_text = f"Object {obj_id} | {obj['classification']} | {obj['confidence']*100:.0f}% | {obj['avg_speed']:.1f} px/frame"
            
            # Create background for text
            text_size = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (10, 10), (20 + text_size[0], 40 + text_size[1]), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, overlay_text, (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            out.write(frame)
    
    cap.release()
    out.release()
    
    return output_path

def reset_session():
    """Reset session state for new analysis"""
    st.session_state.processing_complete = False
    st.session_state.results_data = None
    st.session_state.processed_clips = []
    st.session_state.metadata = []
    st.session_state.video_info = {}
    st.session_state.recommendations = None
    st.session_state.uploaded_video_path = None
    st.session_state.prepared_zips = {}
    
    # Clean up directories
    for directory in ['temp_uploads', 'processed_clips', 'results']:
        if os.path.exists(directory):
            shutil.rmtree(directory)

if __name__ == "__main__":
    main()
