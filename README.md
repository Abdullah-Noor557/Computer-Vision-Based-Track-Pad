# Sci-Fi Touch Interface - Hand Gesture Scroll Control

A Python-based interface that enables touchless scrolling using webcam input and machine learning. Combines YOLOv5 classification with optical flow tracking for intuitive hand gesture control.


![Visualization](https://github.com/user-attachments/assets/cae90d31-5afe-4274-9337-4946cee9d2ba)


## Features

- 🖐️ Real-time hand state classification using YOLOv5
- 🌀 Optical flow-based scroll detection
- 📊 Sci-fi inspired HUD overlay with:
  - Classification confidence bar
  - Scroll speed visualization
  - Real-time FPS counter
  - Directional flow arrows
- ⚡ Performance optimizations for smooth operation

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install opencv-python torch torchvision pyautogui numpy
```
##Usage
Download YOLOv5 classification model (.pt file)

Update configuration in script:
```bash
MODEL_PATH = r'/absolute/path/to/your_model.pt'  # Update this path
CAMERA_SOURCE = 0  # Change camera index if needed
```
