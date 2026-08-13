import cv2
import mediapipe as mp
import numpy as np
import time


class CrunchDetector:

    def __init__(self):

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.VISIBILITY_THRESHOLD = 0.60
        self.SMOOTHING_FRAMES = 7
        self.COOLDOWN_TIME = 0.6

        self.DOWN_ANGLE = 125
        self.UP_ANGLE = 155

        self.rep_count = 0

        # DOWN = lying/extended
        # UP   = crunch position
        self.state = "DOWN"

        self.angle_history = []

        self.last_rep_time = 0

        self.feedback = "Lie down and prepare"

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

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return np.degrees(
            np.arccos(cosine)
        )

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

        # --------------------------------------------------------
        # LEFT SIDE LANDMARKS
        # --------------------------------------------------------

        shoulder = lm[
            self.mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        hip = lm[
            self.mp_pose.PoseLandmark.LEFT_HIP
        ]

        knee = lm[
            self.mp_pose.PoseLandmark.LEFT_KNEE
        ]

        required = [
            shoulder,
            hip,
            knee
        ]

        if not all(
            x.visibility >= self.VISIBILITY_THRESHOLD
            for x in required
        ):

            return {
                "exercise": "CRUNCH",
                "reps": self.rep_count,
                "form": "POOR",
                "feedback": "Keep your upper body visible",
                "frame": frame
            }

        # --------------------------------------------------------
        # TORSO ANGLE
        # --------------------------------------------------------

        body_angle = self.calculate_angle(
            self.point(shoulder),
            self.point(hip),
            self.point(knee)
        )

        if body_angle is None:

            return {
                "exercise": "CRUNCH",
                "reps": self.rep_count,
                "form": "CHECK",
                "feedback": "Adjust your position",
                "frame": frame
            }

        self.angle_history.append(
            body_angle
        )

        if len(self.angle_history) > self.SMOOTHING_FRAMES:

            self.angle_history.pop(0)

        smooth_angle = np.mean(
            self.angle_history
        )

        # --------------------------------------------------------
        # DOWN → UP
        # --------------------------------------------------------

        if self.state == "DOWN":

            if smooth_angle < self.DOWN_ANGLE:

                self.state = "UP"

                self.feedback = (
                    "Good! Return slowly"
                )

        # --------------------------------------------------------
        # UP → DOWN = REP
        # --------------------------------------------------------

        elif self.state == "UP":

            if smooth_angle > self.UP_ANGLE:

                now = time.time()

                if (
                    now -
                    self.last_rep_time
                    >
                    self.COOLDOWN_TIME
                ):

                    self.rep_count += 1

                    self.last_rep_time = now

                self.state = "DOWN"

                self.feedback = (
                    "Good! Crunch completed"
                )

        # --------------------------------------------------------
        # FORM
        # --------------------------------------------------------

        if self.state == "UP":

            if smooth_angle > 110:

                form = "CHECK"

                self.feedback = (
                    "Curl your upper body more"
                )

            else:

                form = "GOOD"

        else:

            form = "GOOD"

        # --------------------------------------------------------
        # DRAW
        # --------------------------------------------------------

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        return {
            "exercise": "CRUNCH",
            "reps": self.rep_count,
            "form": form,
            "feedback": self.feedback,
            "state": self.state,
            "body_angle": round(
                float(smooth_angle),
                2
            ),
            "frame": frame
        }

    def reset(self):

        self.rep_count = 0

        self.state = "DOWN"

        self.angle_history.clear()

        self.last_rep_time = 0

        self.feedback = (
            "Lie down and prepare"
        )

    def close(self):

        self.pose.close()