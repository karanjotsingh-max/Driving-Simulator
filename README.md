# Real-Time Face Recognition and Eye Tracking: Driving Simulator

This project implements a real-time drowsiness detection system for driving simulators using face and eye tracking. It uses YOLO v8 for face detection and Haar Cascade for eye detection. If the driver shows signs of drowsiness, an alarm is triggered to alert them.

## Project Overview

- **Objective**: Detect drowsiness based on eye closures using computer vision.
- **Technologies**:
  - YOLO v8 for face detection
  - Haar Cascade for eye detection
  - Pygame for sound alerts
  - OpenCV for computer vision tasks

## Features

- Real-time face detection using YOLO v8.
- Eye detection using Haar Cascade in the upper half of the detected face.
- Alarm triggers if eyes are detected closed for a certain period (indicating drowsiness).
- Local contrast enhancement (CLAHE) for better eye detection performance.

## Requirements

To run the project, you will need to install the required dependencies. This can be done using `pip`.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/karanjotsingh-max/Driving-Simulator.git
    cd Driving-Simulator
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv8 model weights and the Haar Cascade XML files (for face and eye detection). Ensure that the following files are placed correctly:
    - `model/yolov8n_100e.pt` (YOLOv8 model for face detection)
    - `haarcascade_eye.xml` (Haar Cascade XML file for eye detection)
    - `alarm.mp3` (sound file for drowsiness alert)

4. Run the application:

    ```bash
    python drowsiness_detection.py
    ```

    The webcam will open, and the system will start detecting faces and eyes in real time.

## Usage

- The system will display the webcam feed with rectangles around detected faces and eyes.
- If the eyes remain closed for a threshold duration, a "DROWSINESS ALERT!" message will be displayed, and an alarm sound will play.
- To stop the program, press the 'q' key.

## Acknowledgments

- YOLO v8 for face detection.
- OpenCV for computer vision tools.
- Pygame for handling sound playback.
