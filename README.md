<div align="center">

# 🌌 SkySeer: Autonomous Night Sky Surveillance
### AI-Powered Detection for Satellites, Meteors, and UAP

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

<br>
<img src="assets/1.png" alt="SkySeer Banner" width="100%">
<br>

<p align="center">
  <b>SkySeer</b> uses Computer Vision to turn hours of raw night-sky footage into actionable data.<br>
  It autonomously isolates movement, calculates flight kinematics, and classifies objects without human supervision.
</p>

</div>

---

## 🚀 The Engineering Challenge
Amateur astronomers record hours of footage, but manual analysis is tedious. Standard motion detectors fail because they cannot distinguish between:
* **Satellites:** Linear trajectory, constant velocity.
* **Meteors:** Transient, high-velocity bursts.
* **Noise:** Clouds, sensor grain, or wind shake.

**SkySeer solves this** by implementing a 3-stage pipeline that extracts the "Flight Signature" of every moving pixel.

---

## 📸 Detection Results
The system uses **Unsupervised Clustering (K-Means)** to separate objects based on their kinematic profile.

| **Satellite Detection (Red)** | **Meteor Detection (Yellow)** |
|:---:|:---:|
| <img src="assets/23.png" width="100%"> | <img src="assets/24.png" width="100%"> |
| *High linearity ($R^2 > 0.95$), Constant Velocity* | *High speed, Short duration, Brightness spike* |

---

## 🛠️ How It Works (The Pipeline)

### 1. Motion Isolation (MOG2)
We use a **Mixture of Gaussians** background subtractor to learn the static star field. This allows the system to "ignore" the rotation of the earth and focus only on independent movement.

### 2. Feature Extraction (Kinematics)
For every contour detected, we calculate a velocity vector $(\Delta x, \Delta y)$ and a linearity score.
```math
\text{Velocity} = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}
