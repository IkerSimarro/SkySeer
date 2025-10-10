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

# Configure page
st.set_page_config(
    page_title="SkySeer AI Pipeline",
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

def main():
    st.title("🌌 SkySeer AI Pipeline")
    st.markdown("**Advanced Sky Object Detection & Classification System**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Motion detection sensitivity
        sensitivity = st.slider(
            "Motion Detection Sensitivity",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher values detect smaller objects but may increase false positives"
        )
        
        # Minimum clip duration
        min_duration = st.slider(
            "Minimum Clip Duration (seconds)",
            min_value=0.3,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Minimum duration for a valid detection clip. Higher values reduce false positives from noise."
        )
        
        # Maximum clip duration
        max_duration_enabled = st.checkbox(
            "Enable Maximum Duration Filter",
            value=True,
            help="Filter out objects that stay on screen too long (like stationary stars)"
        )
        
        max_duration = None
        if max_duration_enabled:
            max_duration = st.slider(
                "Maximum Clip Duration (seconds)",
                min_value=1.0,
                max_value=30.0,
                value=15.0,
                step=1.0,
                help="Objects visible longer than this will be filtered out (e.g., to exclude stars)"
            )
        
        # Frame skip rate for performance
        frame_skip = st.slider(
            "Frame Skip Rate",
            min_value=1,
            max_value=10,
            value=3,
            help="Process every Nth frame (higher = faster but less accurate)"
        )
        
        st.markdown("---")
        
        # Settings recommendations (shown after video upload)
        if st.session_state.recommendations:
            st.info("💡 **Recommended Settings for Your Video:**")
            rec = st.session_state.recommendations
            for explanation in rec['explanations']:
                st.markdown(f"• {explanation}")
            
            # Display recommended values
            st.markdown(f"**Suggested values:** Sensitivity={rec['sensitivity']}, " +
                       f"Min Duration={rec['min_duration']}s, " +
                       f"Max Duration={'Enabled' if rec['max_duration_enabled'] else 'Disabled'} " +
                       f"({rec['max_duration']}s), Frame Skip={rec['frame_skip']}")
        
        st.markdown("---")
        st.markdown("**Classification Categories:**")
        st.markdown("🛰️ **Satellite** - Steady orbital motion")
        st.markdown("☄️ **Meteor** - Fast, straight trajectory")
        st.markdown("✈️ **Plane** - Predictable flight path")
        st.markdown("🗑️ **Junk** - Noise/artifacts")

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
        
        st.info("💡 **Tip for large videos:** For videos over 500MB, consider compressing or splitting them into smaller clips for faster processing.")
        
        if uploaded_file is not None:
            # Display video info and generate recommendations
            with st.expander("📊 Video Information", expanded=True):
                video_info = get_video_info(uploaded_file)
                
                # Generate recommendations based on video properties
                if 'error' not in video_info:
                    st.session_state.recommendations = recommend_settings(video_info)
                
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
        
        motion_clips, metadata = processor.process_video(temp_path)
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
        status_text.text("📊 Stage 2/4: Extracting movement features...")
        progress_bar.progress(50)
        
        extractor = FeatureExtractor()
        features_df = extractor.extract_features(metadata)
        progress_bar.progress(70)
        
        # Stage 3: ML Classification
        status_text.text("🧠 Stage 3/4: Classifying objects with AI...")
        
        classifier = MLClassifier()
        results_df = classifier.classify_objects(features_df)
        progress_bar.progress(90)
        
        # Stage 4: Generate Results
        status_text.text("📋 Stage 4/4: Generating results...")
        
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
        
        # Clean up temp file
        os.remove(temp_path)
        
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
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_detections = len(results_df)
        st.metric("Total Detections", total_detections)
    
    with col2:
        stars = len(results_df[results_df['classification'] == 'Star'])
        st.metric("⭐ Stars", stars)
    
    with col3:
        satellites = len(results_df[results_df['classification'] == 'Satellite'])
        st.metric("🛰️ Satellites", satellites)
    
    with col4:
        meteors = len(results_df[results_df['classification'] == 'Meteor'])
        st.metric("☄️ Meteors", meteors)
    
    with col5:
        planes = len(results_df[results_df['classification'] == 'Plane'])
        st.metric("✈️ Planes", planes)
    
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
            'Plane': '#2ca02c',
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
    
    # Get unique classifications with counts
    classification_counts = results_df['classification'].value_counts()
    
    # Create emoji mapping
    emoji_map = {
        'Satellite': '🛰️',
        'Meteor': '☄️',
        'Plane': '✈️',
        'Star': '⭐',
        'Junk': '🗑️'
    }
    
    # Create download buttons for each classification
    cols = st.columns(min(len(classification_counts), 5))
    
    for idx, (classification, count) in enumerate(classification_counts.items()):
        with cols[idx % len(cols)]:
            emoji = emoji_map.get(classification, '📦')
            
            if st.button(
                f"{emoji} {classification} ({count})",
                key=f"download_{classification}",
                use_container_width=True
            ):
                with st.spinner(f"Preparing {classification} clips..."):
                    zip_buffer = create_download_zip(
                        st.session_state.processed_clips, 
                        results_df,
                        classification_filter=classification
                    )
                    
                    st.download_button(
                        label=f"📦 Download {classification} ZIP",
                        data=zip_buffer,
                        file_name=f"skyseer_{classification.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key=f"download_btn_{classification}"
                    )

def reset_session():
    """Reset session state for new analysis"""
    st.session_state.processing_complete = False
    st.session_state.results_data = None
    st.session_state.processed_clips = []
    st.session_state.metadata = []
    st.session_state.video_info = {}
    st.session_state.recommendations = None
    
    # Clean up directories
    for directory in ['temp_uploads', 'processed_clips', 'results']:
        if os.path.exists(directory):
            shutil.rmtree(directory)

if __name__ == "__main__":
    main()
