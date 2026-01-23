<div align="center">

# 🌌 SkySeer: Autonomous Night Sky Surveillance
### AI-Powered Detection for Satellites, Meteors, and UAP

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

<br>
<img src="Screenshots/1.png" alt="SkySeer Banner" width="100%">
<br>

<p align="center">
  <b>SkySeer</b> is a computer vision pipeline designed to automate the analysis of night-sky footage.<br>
  It solves the "Empty Sky Problem" by filtering 99% of static frames and classifying the remaining 1% of motion into <b>Satellites</b>, <b>Meteors</b>, or <b>Noise</b> using unsupervised learning.
</p>

[View Demo](#-detection-results) • [Read the Logic](#-the-engineering-pipeline) • [Install](#-installation--usage)

</div>

---

## 🚀 The Engineering Problem
Amateur astronomers and UAP researchers record terabytes of footage, but manual analysis is impossible. Standard motion detectors (like security cameras) fail in astrophotography because they cannot distinguish between:
* **Satellites:** Linear trajectory, constant velocity, sustained duration.
* **Meteors:** Transient, high-velocity bursts, variable brightness.
* **Sensor Noise:** Wind shake, cloud drift, or grain.

**SkySeer** solves this by extracting **Kinematic Signatures** (Velocity, Linearity, Duration) from raw pixels and clustering them in a high-dimensional feature space.

---

## 📸 Detection Results

The system autonomously classifies objects based on their flight behavior.

| **Satellite Detection (Class 0)** | **Meteor Detection (Class 1)** |
|:---:|:---:|
| <img src="Screenshots/23.png" width="100%"> | <img src="Screenshots/24.png" width="100%"> |
| **Characteristics:** High linearity ($R^2 > 0.95$), Constant Velocity, Long Duration. | **Characteristics:** High velocity spike, Short duration, Brightness flare. |

---

## 🛠️ The Engineering Pipeline

SkySeer processes video in three distinct mathematical stages:

### 1. Motion Isolation (Background Modeling)
We utilize a **Mixture of Gaussians (MOG2)** background subtractor to model the static star field.
* **Star Removal:** The system learns the "static" background over $N$ frames.
* **Noise Filtering:** Morphological operations (Erosion/Dilation) remove single-pixel sensor noise.

### 2. Kinematic Feature Extraction
Once a contour is tracked, we extract physics-based features to define its identity. For a set of centroids $C = \{(x_1, y_1), ..., (x_n, y_n)\}$, we calculate:

**A. Velocity Vector:**
$$v = \frac{\sqrt{(x_i - x_{i-1})^2 + (y_i - y_{i-1})^2}}{\Delta t}$$

**B. Trajectory Linearity ($R^2$):**
We fit a linear regression model to the path. Satellites approach $R^2 \approx 1.0$, while insects/birds/noise show lower linearity.

### 3. Unsupervised Classification (K-Means)
Instead of relying on a labeled dataset (which is scarce for night sky objects), SkySeer uses **K-Means Clustering ($k=3$)** to group objects dynamically.

* **Cluster 0 (Satellites):** Low variance in velocity, high linearity.
* **Cluster 1 (Meteors):** High variance in velocity, short duration.
* **Cluster 2 (Noise):** Discarded based on erratic trajectory.

---

## 💻 Software Interface
The engine is wrapped in a **Streamlit** dashboard, allowing for drag-and-drop video processing, real-time parameter tuning, and data visualization.

<div align="center">
  <img src="Screenshots/1.png" alt="SkySeer Interface" width="85%">
  <p><i>The SkySeer Dashboard: Upload, Process, and Analyze.</i></p>
</div>

---

## 📊 Data Output
Beyond visual bounding boxes, SkySeer generates a scientific manifest (`analysis_report.csv`) containing:
* **Timestamp:** Exact frame of entry/exit.
* **Object ID:** Unique tracking identifier.
* **Classification:** Satellite vs. Meteor.
* **Confidence Score:** Distance from the cluster centroid.
* **Velocity Profile:** Avg speed (px/frame).

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* OpenCV (`opencv-python`)

### 1. Clone the Repository
```bash
git clone [https://github.com/IkerSimarro/SkySeer.git](https://github.com/IkerSimarro/SkySeer.git)
cd SkySeer

📂 Project Structure
SkySeer/
├── src/
│   ├── motion_engine.py    # Core MOG2 Logic
│   ├── kinematics.py       # Velocity & Linearity Math
│   ├── classifier.py       # Scikit-Learn K-Means Logic
│   └── app.py              # Streamlit Entry Point
├── Screenshots/            # Documentation Images
├── requirements.txt        # Dependencies
└── README.md               # Documentation

👨‍💻 Author
Iker Simarro Cuevas

Focus: Computer Vision, Signal Processing, Scientific ML

https://www.linkedin.com/in/iker-simarro-546169227/
