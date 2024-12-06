import cv2
from ultralytics import YOLO

# Load YOLO model for face detection
model = YOLO('model/yolov8n_100e.pt')  # Replace with your YOLOv8 model

# Load Haar cascade for eye detection (pre-trained for glasses)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Thresholds
CLOSED_EYE_FRAMES = 10  # Number of consecutive frames for drowsiness detection
frame_counter = 0

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # YOLO Face Detection
        results = model(frame)
        if results[0].boxes is not None:
            detections = results[0].boxes.xyxy.cpu().numpy()
            
            for detection in detections:
                # Extract face bounding box coordinates
                x1, y1, x2, y2 = map(int, detection[:4])
                face = frame[y1:y2, x1:x2]  # Crop face region

                # Convert to grayscale for Haar cascade
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Detect eyes
                eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)

                # If no eyes are detected, increment frame counter
                if len(eyes) == 0:
                    frame_counter += 1
                else:
                    frame_counter = 0  # Reset counter if eyes are detected

                # Trigger drowsiness alert if counter exceeds threshold
                if frame_counter >= CLOSED_EYE_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw rectangles around eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Overlay the face region back onto the frame
                frame[y1:y2, x1:x2] = face

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
