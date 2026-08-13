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


def angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    d = np.linalg.norm(ba) * np.linalg.norm(bc)
    if d == 0:
        return None

    cosine = np.clip(np.dot(ba, bc) / d, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


cap = cv2.VideoCapture(0)

rep_count = 0
state = "UP"

angle_history = []
hip_history = []

standing_angles = []
standing_hips = []

calibrated = False
calibration_start = time.time()

baseline_angle = 0
baseline_hip = 0

last_rep_time = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark

        lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
        lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]

        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

        visible = all(
            x.visibility >= 0.60
            for x in [lh, lk, la, rh, rk, ra]
        )

        if visible:

            left = angle(
                [lh.x, lh.y],
                [lk.x, lk.y],
                [la.x, la.y]
            )

            right = angle(
                [rh.x, rh.y],
                [rk.x, rk.y],
                [ra.x, ra.y]
            )

            knee = (left + right) / 2
            hip_y = (lh.y + rh.y) / 2

            angle_history.append(knee)
            hip_history.append(hip_y)

            if len(angle_history) > 10:
                angle_history.pop(0)

            if len(hip_history) > 10:
                hip_history.pop(0)

            smooth_angle = np.mean(angle_history)
            smooth_hip = np.mean(hip_history)

            # ---------------- CALIBRATION ----------------

            if not calibrated:

                standing_angles.append(smooth_angle)
                standing_hips.append(smooth_hip)

                remaining = 3 - (
                    time.time() - calibration_start
                )

                if remaining > 0:

                    feedback = "Stand normally..."

                else:

                    baseline_angle = np.median(
                        standing_angles
                    )

                    baseline_hip = np.median(
                        standing_hips
                    )

                    calibrated = True

                    feedback = "Calibration complete"

            # ---------------- DETECTION ----------------

            else:

                down_threshold = baseline_angle - 45
                up_threshold = baseline_angle - 15

                hip_drop = smooth_hip - baseline_hip

                if state == "UP":

                    if (
                        smooth_angle < down_threshold
                        and
                        hip_drop > 0.025
                    ):

                        state = "DOWN"

                        feedback = "Squat down"

                elif state == "DOWN":

                    if smooth_angle > up_threshold:

                        now = time.time()

                        if now - last_rep_time > 0.6:

                            rep_count += 1
                            last_rep_time = now

                        state = "UP"
                        feedback = "Rep completed"

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            cv2.putText(
                frame,
                f"Knee: {smooth_angle:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,0),
                2
            )

            cv2.putText(
                frame,
                f"State: {state}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

            cv2.putText(
                frame,
                f"SQUATS: {rep_count}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0,255,0),
                3
            )

            cv2.putText(
                frame,
                feedback if calibrated else f"{max(0, remaining):.1f}s",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
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

    cv2.imshow("Squat Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
pose.close()

print("Total squats:", rep_count)