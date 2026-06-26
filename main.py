import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("traffic.mp4")

vehicle_count = 0
emergency_mode = False   # (simulation variable)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    boxes = results[0].boxes

    count = 0

    for box in boxes:
        cls = int(box.cls[0])

        # Vehicle classes
        if cls in [2, 3, 5, 7]:
            count += 1

    vehicle_count = count

    #  STEP 2: Signal Logic with Emergency Simulation
    if emergency_mode:
        density = "EMERGENCY"
        signal_time = 5
    else:
        if vehicle_count < 10:
            density = "LOW"
            signal_time = 10
        elif vehicle_count < 20:
            density = "MEDIUM"
            signal_time = 20
        else:
            density = "HIGH"
            signal_time = 30

    annotated_frame = results[0].plot()

    # Display Vehicle Count
    cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3)

    # Display Density
    cv2.putText(annotated_frame, f"Density: {density}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 0, 0), 3)

    # Display Signal Time
    cv2.putText(annotated_frame, f"Signal: {signal_time}s",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    # Traffic Status
    if density == "LOW":
        status = "Smooth Traffic"
    elif density == "MEDIUM":
        status = "Moderate Traffic"
    elif density == "HIGH":
        status = "Heavy Traffic"
    else:
        status = "Emergency Priority"

    cv2.putText(annotated_frame, status,
                (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2)

    # Emergency display
    if emergency_mode:
        cv2.putText(annotated_frame, "EMERGENCY MODE ACTIVATED!",
                    (20, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    cv2.imshow("Traffic Detection", annotated_frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('e'):   # Press E → Emergency ON
        emergency_mode = True

    elif key == ord('n'): # Press N → Back to normal
        emergency_mode = False

    elif key == ord('q'): # Quit
        break

cap.release()
cv2.destroyAllWindows()
