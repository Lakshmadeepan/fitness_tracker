import cv2
import mediapipe as mp
import numpy as np
import time


class JumpingJackDetector:

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

        self.LEG_OPEN_RATIO = 1.35

        self.rep_count = 0
        self.state = "CLOSED"

        self.ankle_history = []
        self.wrist_history = []

        self.last_rep_time = 0

        self.feedback = "Stand normally"

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

        left_wrist = lm[
            self.mp_pose.PoseLandmark.LEFT_WRIST
        ]

        right_wrist = lm[
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]

        left_ankle = lm[
            self.mp_pose.PoseLandmark.LEFT_ANKLE
        ]

        right_ankle = lm[
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

        required = [
            left_shoulder,
            right_shoulder,
            left_wrist,
            right_wrist,
            left_ankle,
            right_ankle
        ]

        if not all(
            x.visibility >= self.VISIBILITY_THRESHOLD
            for x in required
        ):

            return {
                "exercise": "JUMPING JACK",
                "reps": self.rep_count,
                "form": "POOR",
                "feedback": "Keep your full body visible",
                "frame": frame
            }

        shoulder_width = abs(
            left_shoulder.x -
            right_shoulder.x
        )

        ankle_width = abs(
            left_ankle.x -
            right_ankle.x
        )

        if shoulder_width < 0.01:

            return {
                "exercise": "JUMPING JACK",
                "reps": self.rep_count,
                "form": "CHECK",
                "feedback": "Move farther from camera",
                "frame": frame
            }

        leg_ratio = (
            ankle_width /
            shoulder_width
        )

        shoulder_y = (
            left_shoulder.y +
            right_shoulder.y
        ) / 2

        wrist_y = (
            left_wrist.y +
            right_wrist.y
        ) / 2

        hands_up = (
            wrist_y <
            shoulder_y - 0.08
        )

        legs_open = (
            leg_ratio >
            self.LEG_OPEN_RATIO
        )

        open_position = (
            hands_up and
            legs_open
        )

        # CLOSED → OPEN
        if self.state == "CLOSED":

            if open_position:

                self.state = "OPEN"

                self.feedback = (
                    "Good! Close your arms and legs"
                )

        # OPEN → CLOSED = ONE REP
        elif self.state == "OPEN":

            if not hands_up and not legs_open:

                now = time.time()

                if now - self.last_rep_time > self.COOLDOWN_TIME:

                    self.rep_count += 1
                    self.last_rep_time = now

                self.state = "CLOSED"

                self.feedback = (
                    "Good! Rep completed"
                )

        form = "GOOD"

        if self.state == "OPEN":

            if not hands_up:
                form = "CHECK"
                self.feedback = "Raise your arms"

            elif not legs_open:
                form = "CHECK"
                self.feedback = "Spread your legs"

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        return {
            "exercise": "JUMPING JACK",
            "reps": self.rep_count,
            "form": form,
            "feedback": self.feedback,
            "state": self.state,
            "frame": frame
        }

    def reset(self):

        self.rep_count = 0
        self.state = "CLOSED"

        self.ankle_history.clear()
        self.wrist_history.clear()

        self.last_rep_time = 0

        self.feedback = "Stand normally"

    def close(self):

        self.pose.close()