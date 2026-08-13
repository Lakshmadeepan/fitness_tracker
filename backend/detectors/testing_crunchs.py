import cv2
import mediapipe as mp
import numpy as np
import time


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


def calculate_angle(a, b, c):

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(
        np.arccos(cosine)
    )


cap = cv2.VideoCapture(0)

rep_count = 0
state = "DOWN"

angle_history = []
last_rep_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb)

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark

        shoulder = lm[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        hip = lm[
            mp_pose.PoseLandmark.LEFT_HIP
        ]

        knee = lm[
            mp_pose.PoseLandmark.LEFT_KNEE
        ]

        visible = all(
            x.visibility >= 0.60
            for x in [shoulder, hip, knee]
        )

        if visible:

            body_angle = calculate_angle(
                [shoulder.x, shoulder.y],
                [hip.x, hip.y],
                [knee.x, knee.y]
            )

            if body_angle is not None:

                angle_history.append(body_angle)

                if len(angle_history) > 7:
                    angle_history.pop(0)

                smooth_angle = np.mean(
                    angle_history
                )

                # DOWN → CRUNCH UP

                if state == "DOWN":

                    if smooth_angle < 125:

                        state = "UP"

                # CRUNCH UP → DOWN = 1 REP

                elif state == "UP":

                    if smooth_angle > 155:

                        now = time.time()

                        if now - last_rep_time > 0.6:

                            rep_count += 1
                            last_rep_time = now

                        state = "DOWN"

                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                cv2.putText(
                    frame,
                    f"Body Angle: {smooth_angle:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"State: {state}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"CRUNCHES: {rep_count}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    3
                )

    else:

        cv2.putText(
            frame,
            "NO PERSON DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    cv2.imshow(
        "Crunch Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
pose.close()

print("Total crunches:", rep_count)