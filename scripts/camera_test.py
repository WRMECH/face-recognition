import cv2
import sys

print("Attempting to open webcam...")
cap = cv2.VideoCapture(0) # Try to open the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam. This might trigger the macOS camera permission prompt.")
    print("Please check System Settings > Privacy & Security > Camera.")
    sys.exit(1)

print("Webcam opened successfully! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow('Webcam Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam test finished.")