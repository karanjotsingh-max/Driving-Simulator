import cv2
import numpy as np
from ultralytics import YOLO
from pygame import mixer
import time

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Threshold for considering the eyes closed
CLOSED_EYE_FRAMES_THRESHOLD = 30
closed_eye_frames = 0

# Initialize sound mixer and load alarm sound
mixer.init()
alarm_sound = mixer.Sound('alarm.mp3')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load YOLO model for face detection
model = YOLO('model/yolov8n_100e.pt')

try:
    # Optional: Set up for FPS measurement
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame_count += 1

        # YOLO Face Detection
        results = model(frame)
        faces_detected = results[0].boxes

        if faces_detected is not None and len(faces_detected) > 0:
            detections = results[0].boxes.xyxy.cpu().numpy()

            # Sort detections by size (area) and only use the largest one
            detections = sorted(detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
            largest_face = detections[0]

            x1, y1, x2, y2 = map(int, largest_face[:4])

            # Extract face from frame
            face = frame[y1:y2, x1:x2]

            # Convert face to grayscale for eye detection
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_face = clahe.apply(gray_face)

            # Focus on the upper half of the face for eye detection
            half_height = gray_face.shape[0] // 2
            upper_face = gray_face[:half_height, :]

            # Detect eyes in the upper half of the face
            eyes = eye_cascade.detectMultiScale(upper_face, 1.1, 5)

            if len(eyes) == 2:  # Check if two eyes are detected
                (ex1, ey1, ew1, eh1) = eyes[0]
                (ex2, ey2, ew2, eh2) = eyes[1]

                # Draw rectangles around the eyes on the 'face' region
                cv2.rectangle(face, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (255, 0, 0), 2)
                cv2.rectangle(face, (ex2, ey2), (ex2 + ew2, ey2 + eh2), (0, 255, 0), 2)

                # Place the processed face back into the frame
                frame[y1:y2, x1:x2] = face

                # Reset the closed eye counter
                closed_eye_frames = 0
            else:
                closed_eye_frames += 1
                # Trigger drowsiness alert if eyes are not detected for continuous frames
                if closed_eye_frames >= CLOSED_EYE_FRAMES_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if not mixer.get_busy():  # Play alarm if not already playing
                        alarm_sound.play()

        # (Optional) Display approximate FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = 30 / elapsed
            print(f"FPS: {fps:.2f}")
            start_time = time.time()

        # Display Key Metrics
        cv2.putText(frame, f"Closed Eye Frames: {closed_eye_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
