import cv2

from fitness_engine import FitnessEngine


engine = FitnessEngine()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit


# Start workout
engine.start_workout()

print("Workout started.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Send camera frame to FitnessEngine
    result = engine.process_frame(frame)

    # Get processed frame from detector
    display_frame = result.get("frame", frame)

    # Display backend result
    cv2.putText(
        display_frame,
        f"Exercise: {result.get('exercise', 'NO EXERCISE')}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Reps: {result.get('reps', 0)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3
    )

    cv2.putText(
        display_frame,
        f"Form: {result.get('form', 'Waiting')}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        result.get('feedback', ''),
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Burn-Ex Backend Webcam Test",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Stop
engine.stop_workout()

cap.release()
cv2.destroyAllWindows()

engine.close()

print("Workout stopped.")
print("Final result:")
print(engine.get_status())