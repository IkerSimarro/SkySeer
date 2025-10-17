# 🌌 SkySeer AI - Sky Object Detection System

**Advanced computer vision and machine learning for automatic satellite and meteor detection in night sky videos.**

---

## 📖 What is SkySeer?

SkySeer AI is an intelligent video analysis tool that automatically detects and classifies moving objects in night sky footage. Perfect for amateur astronomers and space enthusiasts, it transforms hours of raw video into precise, actionable data—identifying satellites and meteors while filtering out noise and false positives.

**What you get:**
- ✅ Sped-up annotated videos (10x speed) with color-coded detections
- ✅ Detailed CSV reports with speeds, trajectories, and confidence scores
- ✅ Category-specific downloads (separate Satellite and Meteor packages)
- ✅ Smart filtering: typically <10 high-quality detections per video

---

## 🚀 Quick Start

### System Requirements
- **Operating System:** Windows 10/11 (64-bit)
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space for processing
- **Display:** 1920x1080 or higher recommended

### How to Run
1. **Double-click** `SkySeer.exe` to launch the application
2. Your default web browser will open automatically to the SkySeer interface
3. If the browser doesn't open, manually navigate to: `http://localhost:5000`

### First-Time Setup
- No installation required - it's a standalone executable
- No internet connection needed after initial launch
- All processing happens locally on your computer

---

## 📹 How to Use

### Step 1: Upload Your Video
1. Click **"Browse files"** or drag-and-drop your night sky video
2. Supported formats: **MP4, AVI, MOV, MKV**
3. Best results: stable, tripod-mounted footage

### Step 2: Configure Settings (Optional)
The system auto-recommends optimal settings, but you can adjust:

**Motion Detection Sensitivity (1-10)**
- Low (1-3): Fewer detections, reduces noise
- Medium (4-6): Balanced (recommended for most videos)
- High (7-10): More sensitive, may increase false positives

**Minimum Clip Duration (0.5-5s)**
- Default: 1.5s
- Lower for brief meteors, higher to filter noise

**Maximum Clip Duration (5-30s)**
- Default: 15s
- Increase for slow-moving satellites

**Frame Skip Rate (1-6)**
- Higher = faster processing (recommended for long videos)
- Default auto-adjusts based on video duration

### Step 3: Process Video
1. Click **"🚀 Start Detection"**
2. Watch real-time progress updates
3. Processing time: ~1-5 minutes per 10-minute video (varies by settings)

### Step 4: Review Results
- **Main Video:** Sped-up 10x with all detections marked
- **Results Table:** Detailed metrics for each object
- **Downloads:** Category-specific packages (see below)

---

## 🎨 Color Coding

**RED Boxes** 🔴 = **Satellites**
- Slow, steady movement across the sky
- Typical duration: 5-20 seconds
- Examples: ISS, Starlink, orbital debris

**YELLOW Boxes** 🟡 = **Meteors**
- Fast, brief streaks
- Typical duration: <4 seconds
- Examples: shooting stars, meteor showers

---

## 📦 Understanding Your Downloads

### 1. CSV Report (`analysis_report.csv`)
Contains detailed metrics for every detection:
- **clip_id:** Unique identifier (matches video labels)
- **classification:** Satellite, Meteor, or Junk
- **confidence:** Classification confidence (0-1)
- **avg_speed:** Average speed in pixels/frame
- **speed_consistency:** Movement stability (0-1)
- **duration:** Detection duration in seconds
- **linearity:** Path straightness (0-1, higher = straighter)
- **direction_changes:** Number of trajectory shifts
- **avg_brightness:** Relative brightness
- **satellite_score / meteor_score:** AI scoring metrics

### 2. Category Downloads (Satellite / Meteor)
Each category ZIP includes:
- **`{category}_detections.mp4`:** Sped-up video showing ONLY that classification
- **`analysis_report.csv`:** Filtered data for that category only
- **`SUMMARY.txt`:** Quick overview with detection counts

**Note:** Only **Satellite** and **Meteor** downloads are available. Junk/noise detections are automatically filtered out.

### 3. Video Outputs
**Main Output Video:**
- 10x speed (10-minute input → 1-minute output)
- All detections visible with color-coded boxes
- Labels show: "ID:{number} {Classification}"

**Category Videos:**
- Same 10x speed format
- Shows ONLY that specific classification
- Clean filtering: Satellite downloads = only RED boxes, Meteor downloads = only YELLOW boxes

---

## 🎯 Best Practices

### For Optimal Results:
✅ **Use tripod-mounted cameras** - reduces false positives from camera shake
✅ **Record in dark conditions** - best for satellites and meteors
✅ **Avoid cloudy footage** - clouds create motion artifacts
✅ **Use higher sensitivity (5-6)** for clean, dark night sky videos
✅ **Use lower sensitivity (2-4)** for noisy or brighter videos

### Common Issues:
❌ **Too many false detections?** → Lower sensitivity or increase min duration
❌ **Missing obvious objects?** → Increase sensitivity or lower min duration
❌ **Processing timeout?** → Increase frame skip rate (especially for long videos)

---

## 🔧 Troubleshooting

### Application Won't Start
- **Check antivirus:** Some security software may block the .exe
- **Run as Administrator:** Right-click → "Run as administrator"
- **Port conflict:** Ensure port 5000 isn't used by another application

### Browser Doesn't Open
- Manually open your browser and go to: `http://localhost:5000`
- Try a different browser (Chrome, Firefox, Edge)

### Processing Errors
- **Video too large:** Try videos under 2GB for best performance
- **Unsupported format:** Convert to MP4 using free tools like HandBrake
- **Out of memory:** Close other applications, restart SkySeer

### Poor Detection Quality
- **Too sensitive:** Reduce sensitivity setting to 3-4
- **Not sensitive enough:** Increase sensitivity to 6-7
- **Video too shaky:** Use tripod footage or stabilize video first
- **Too much noise:** Use longer minimum duration (2-3s)

---

## 📊 Technical Details

### How It Works
SkySeer uses a **4-stage intelligent pipeline:**

1. **Motion Detection:** Advanced background subtraction (MOG2 algorithm)
2. **Feature Extraction:** Calculates numerical "flight signatures" (speed, trajectory, brightness patterns)
3. **ML Classification:** Unsupervised K-Means clustering categorizes objects
4. **Output Generation:** Creates annotated videos and detailed reports

### Technology Stack
- **Computer Vision:** OpenCV with MOG2 background subtraction
- **Machine Learning:** scikit-learn K-Means clustering (3 clusters)
- **Frontend:** Streamlit web interface
- **Data Processing:** Pandas, NumPy for numerical operations
- **Visualization:** Plotly for interactive charts

### Privacy & Security
- ✅ **100% offline processing** - no internet required after launch
- ✅ **No data uploads** - all videos stay on your computer
- ✅ **No accounts needed** - completely standalone

---

## 📄 License & Credits

**SkySeer AI** - Developed as a portfolio project showcasing computer vision and machine learning expertise.

**Created by:** A passionate CS student combining AI and space exploration

**Use Cases:**
- Amateur astronomy and satellite tracking
- Meteor shower observation and documentation
- Time-lapse photography event extraction
- Citizen science data collection

---

## 🆘 Support & Contact

For questions, bug reports, or feature suggestions:
- Review this README's troubleshooting section
- Check your video quality and settings
- Ensure your system meets minimum requirements

---

## 🌟 Tips for Great Results

1. **Start with recommended settings** - the auto-configuration is optimized for most videos
2. **Test different sensitivities** - every camera and location is different
3. **Use the CSV data** - detailed metrics help understand detection quality
4. **Compare categories** - review both Satellite and Meteor downloads to see differences
5. **Save your settings** - note what works best for your specific camera setup

---

**SkySeer: Transforming night sky footage into structured astronomical data.** 🌌🛰️☄️

*Happy stargazing and detection!*
