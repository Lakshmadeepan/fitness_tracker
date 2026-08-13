import cv2
import mediapipe as mp
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

cap = cv2.VideoCapture(0)

rep_count = 0
state = "CLOSED"
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

        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

        visible = all(
            x.visibility >= 0.60
            for x in [ls, rs, lw, rw, la, ra]
        )

        if visible:

            shoulder_width = abs(ls.x - rs.x)
            ankle_width = abs(la.x - ra.x)

            if shoulder_width > 0.01:

                leg_ratio = ankle_width / shoulder_width

                shoulder_y = (ls.y + rs.y) / 2
                wrist_y = (lw.y + rw.y) / 2

                hands_up = (
                    wrist_y < shoulder_y - 0.08
                )

                legs_open = (
                    leg_ratio > 1.35
                )

                open_position = (
                    hands_up and legs_open
                )

                if state == "CLOSED":

                    if open_position:

                        state = "OPEN"

                elif state == "OPEN":

                    if not hands_up and not legs_open:

                        now = time.time()

                        if now - last_rep_time > 0.5:

                            rep_count += 1
                            last_rep_time = now

                        state = "CLOSED"

                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                cv2.putText(
                    frame,
                    f"Leg Ratio: {leg_ratio:.2f}",
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
                    f"JUMPING JACKS: {rep_count}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    3
                )

        else:

            cv2.putText(
                frame,
                "Keep full body visible",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

    cv2.imshow(
        "Jumping Jack Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

print("Total jumping jacks:", rep_count)