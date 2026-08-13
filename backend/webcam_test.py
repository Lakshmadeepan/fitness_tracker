import cv2

from fitness_engine import FitnessEngine


engine = FitnessEngine()

engine.start_workout()

print()
print("====================================")
print("BURN-EX WEBCAM TEST")
print("====================================")
print("1 = Squat")
print("2 = Push-up")
print("3 = Jumping Jack")
print("4 = Crunch")
print("Q = Quit")
print("====================================")


cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not open.")
    raise SystemExit


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -------------------------------------------------
    # Keyboard selection
    # -------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("1"):

        engine.set_exercise("SQUAT")

    elif key == ord("2"):

        engine.set_exercise("PUSH-UP")

    elif key == ord("3"):

        engine.set_exercise("JUMPING JACK")

    elif key == ord("4"):

        engine.set_exercise("CRUNCH")

    elif key == ord("q"):

        break

    # -------------------------------------------------
    # Process frame
    # -------------------------------------------------

    result = engine.process_frame(frame)

    display = result.get(
        "frame",
        frame
    )

    # -------------------------------------------------
    # UI
    # -------------------------------------------------

    cv2.putText(
        display,
        f"Exercise: {result['exercise']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Reps: {result['reps']}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3
    )

    cv2.putText(
        display,
        f"Duration: {result['duration']:.1f}s",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Form: {result['form']}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        result["feedback"],
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        "1:SQUAT 2:PUSHUP 3:JJ 4:CRUNCH",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Burn-Ex",
        display
    )


engine.stop_workout()

cap.release()
cv2.destroyAllWindows()

print()
print("FINAL WORKOUT RESULT")
print(engine.get_status())

engine.close()