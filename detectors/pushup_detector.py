import cv2
import mediapipe as mp
import numpy as np
import time


class PushupDetector:

    def __init__(self):

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.SMOOTHING_FRAMES = 7
        self.VISIBILITY_THRESHOLD = 0.60
        self.COOLDOWN_TIME = 0.6

        self.DOWN_ANGLE = 100
        self.UP_ANGLE = 155

        self.rep_count = 0
        self.state = "UP"

        self.left_history = []
        self.right_history = []

        self.last_rep_time = 0

        self.feedback = "Get into push-up position"

    def calculate_angle(self, a, b, c):

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

        return np.degrees(np.arccos(cosine))

    def point(self, landmark):

        return [
            landmark.x,
            landmark.y
        ]

    def process_frame(self, frame):

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(rgb)

        if not results.pose_landmarks:

            return {
                "exercise": "NO EXERCISE",
                "reps": self.rep_count,
                "form": "Waiting",
                "feedback": "No person detected",
                "frame": frame
            }

        lm = results.pose_landmarks.landmark

        left_shoulder = lm[
            self.mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        right_shoulder = lm[
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        left_elbow = lm[
            self.mp_pose.PoseLandmark.LEFT_ELBOW
        ]

        right_elbow = lm[
            self.mp_pose.PoseLandmark.RIGHT_ELBOW
        ]

        left_wrist = lm[
            self.mp_pose.PoseLandmark.LEFT_WRIST
        ]

        right_wrist = lm[
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]

        required = [
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
            left_wrist,
            right_wrist
        ]

        if not all(
            x.visibility >= self.VISIBILITY_THRESHOLD
            for x in required
        ):

            return {
                "exercise": "PUSH-UP",
                "reps": self.rep_count,
                "form": "POOR",
                "feedback": "Keep your arms visible",
                "frame": frame
            }

        left_angle = self.calculate_angle(
            self.point(left_shoulder),
            self.point(left_elbow),
            self.point(left_wrist)
        )

        right_angle = self.calculate_angle(
            self.point(right_shoulder),
            self.point(right_elbow),
            self.point(right_wrist)
        )

        if left_angle is None or right_angle is None:
            return {
                "exercise": "PUSH-UP",
                "reps": self.rep_count,
                "form": "CHECK",
                "feedback": "Adjust your position",
                "frame": frame
            }

        self.left_history.append(left_angle)
        self.right_history.append(right_angle)

        if len(self.left_history) > self.SMOOTHING_FRAMES:
            self.left_history.pop(0)

        if len(self.right_history) > self.SMOOTHING_FRAMES:
            self.right_history.pop(0)

        left_smooth = np.mean(self.left_history)
        right_smooth = np.mean(self.right_history)

        elbow_angle = (
            left_smooth +
            right_smooth
        ) / 2

        # DOWN
        if self.state == "UP":

            if elbow_angle < self.DOWN_ANGLE:

                self.state = "DOWN"

                self.feedback = "Push back up"

        # UP
        elif self.state == "DOWN":

            if elbow_angle > self.UP_ANGLE:

                now = time.time()

                if now - self.last_rep_time > self.COOLDOWN_TIME:

                    self.rep_count += 1
                    self.last_rep_time = now

                self.state = "UP"

                self.feedback = "Good! Rep completed"

        # FORM
        if self.state == "DOWN":

            if elbow_angle > 120:
                form = "CHECK"
                self.feedback = "Go lower"
            else:
                form = "GOOD"
        else:
            form = "GOOD"

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        return {
            "exercise": "PUSH-UP",
            "reps": self.rep_count,
            "form": form,
            "feedback": self.feedback,
            "state": self.state,
            "elbow_angle": round(
                float(elbow_angle), 2
            ),
            "frame": frame
        }

    def reset(self):

        self.rep_count = 0
        self.state = "UP"

        self.left_history.clear()
        self.right_history.clear()

        self.last_rep_time = 0

        self.feedback = "Get into push-up position"

    def close(self):

        self.pose.close()