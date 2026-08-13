import cv2
from backend.detectors.pushup_detector import PushupDetector


detector = PushupDetector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    result = detector.process_frame(frame)

    # The detector returns the processed frame
    display_frame = result["frame"]

    # Display information
    cv2.putText(
        display_frame,
        f"Exercise: {result['exercise']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Reps: {result['reps']}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3
    )

    cv2.putText(
        display_frame,
        f"Form: {result['form']}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        result["feedback"],
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Push-Up Detector Test",
        display_frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()

print("Total Push-ups:", detector.rep_count)