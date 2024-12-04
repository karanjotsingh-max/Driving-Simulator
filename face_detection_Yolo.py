import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('model/yolov8n_100e.pt')  # Replace with your YOLOv8 model

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
        results = model.predict(source=frame, conf=0.5, show=False)
        detections = results[0].boxes.xyxy

        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Real-Time Face Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
