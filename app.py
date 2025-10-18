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
    page_title="SkySeer AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean loading spinner
st.markdown("""
<style>
    /* Smooth rotation animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide only the default Streamlit spinner SVG */
    div[data-testid="stSpinner"] svg {
        display: none !important;
    }
    
    /* Custom spinner container */
    div[data-testid="stSpinner"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 2rem !important;
        position: relative !important;
    }
    
    /* Create custom spinner using pseudo-element */
    div[data-testid="stSpinner"]::before {
        content: "" !important;
        display: block !important;
        width: 50px !important;
        height: 50px !important;
        border: 4px solid rgba(31, 119, 180, 0.15) !important;
        border-top: 4px solid #1f77b4 !important;
        border-radius: 50% !important;
        animation: spin 0.8s linear infinite !important;
        margin-bottom: 1rem !important;
    }
    
    /* Style spinner text - target div and span elements */
    div[data-testid="stSpinner"] div,
    div[data-testid="stSpinner"] span {
        color: #1f77b4 !important;
        font-weight: 500 !important;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

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
            max_value=30.0,
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
        else:
            # Default recommendations before video upload
            rec = {
                'sensitivity': 5,
                'min_duration': 1.5,
                'max_duration': 15,
                'frame_skip': 3,
                'explanations': ['Upload a video to see customized recommendations']
            }
        
        # Create a clean grid layout for recommended values
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.metric(label="Sensitivity", value=rec['sensitivity'])
            st.metric(label="Min Duration", value=f"{rec['min_duration']}s")
        
        with rec_col2:
            st.metric(label="Max Duration", value=f"{rec['max_duration']}s")
            st.metric(label="Frame Skip", value=rec['frame_skip'])
        
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
                    st.session_state.recommendations = recommend_settings(video_info, uploaded_file)
                
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
    
    # Create download buttons for Satellite and Meteor only
    cols = st.columns(min(len(classification_counts), 2)) if len(classification_counts) > 0 else st.columns(1)
    
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
                        classification_filter=classification,
                        metadata=st.session_state.metadata
                    )
                    
                    st.download_button(
                        label=f"📦 Download {classification} ZIP",
                        data=zip_buffer,
                        file_name=f"skyseer_{classification.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key=f"download_btn_{classification}"
                    )
    
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
    
    # Small FAQ / Troubleshooting Section
    st.markdown("---")
    st.subheader("❓ Troubleshooting")
    
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
            - Trim video to <10 minutes
            - Use recommended settings
            """)
    
    with col_faq2:
        with st.expander("😕 Missing objects?"):
            st.markdown("""
            - Increase sensitivity to 6-7
            - Decrease minimum duration to 1.0s
            - Lower frame skip to 2-3
            """)
        
        with st.expander("🎨 Color codes?"):
            st.markdown("""
            - 🔴 **RED** = Satellites
            - 🟡 **YELLOW** = Meteors
            """)

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
    
    # Clean up directories
    for directory in ['temp_uploads', 'processed_clips', 'results']:
        if os.path.exists(directory):
            shutil.rmtree(directory)

if __name__ == "__main__":
    main()
