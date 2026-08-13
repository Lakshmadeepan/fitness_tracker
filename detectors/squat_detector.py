import cv2
import mediapipe as mp
import numpy as np
import time


class SquatDetector:

    def __init__(self):

        # ========================================================
        # MEDIAPIPE
        # ========================================================

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # ========================================================
        # SETTINGS
        # ========================================================

        self.CALIBRATION_TIME = 3.0
        self.SMOOTHING_FRAMES = 10

        self.DOWN_ANGLE_DROP = 45
        self.UP_ANGLE_DROP = 15

        self.HIP_DROP_MIN = 0.025

        self.VISIBILITY_THRESHOLD = 0.60

        self.COOLDOWN_TIME = 0.6

        # ========================================================
        # STATE
        # ========================================================

        self.rep_count = 0

        self.state = "UP"

        self.angle_history = []
        self.hip_history = []

        self.standing_angles = []
        self.standing_hip_positions = []

        self.calibrated = False

        self.calibration_start = time.time()

        self.last_rep_time = 0

        self.feedback = "Stand still for calibration"

        self.baseline_angle = None
        self.baseline_hip = None

    # ============================================================
    # ANGLE
    # ============================================================

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

        cosine_angle = np.dot(ba, bc) / denominator

        cosine_angle = np.clip(
            cosine_angle,
            -1.0,
            1.0
        )

        return np.degrees(
            np.arccos(cosine_angle)
        )

    # ============================================================
    # POINT
    # ============================================================

    def get_point(self, landmark):

        return [
            landmark.x,
            landmark.y
        ]

    # ============================================================
    # PROCESS ONE FRAME
    # ============================================================

    def process_frame(self, frame):

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(rgb)

        # ========================================================
        # NO PERSON
        # ========================================================

        if not results.pose_landmarks:

            return {
                "exercise": "NO EXERCISE",
                "reps": self.rep_count,
                "form": "Waiting",
                "feedback": "NO PERSON DETECTED",
                "calibrated": self.calibrated,
                "frame": frame
            }

        landmarks = results.pose_landmarks.landmark

        # ========================================================
        # LANDMARKS
        # ========================================================

        left_hip = landmarks[
            self.mp_pose.PoseLandmark.LEFT_HIP
        ]

        right_hip = landmarks[
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]

        left_knee = landmarks[
            self.mp_pose.PoseLandmark.LEFT_KNEE
        ]

        right_knee = landmarks[
            self.mp_pose.PoseLandmark.RIGHT_KNEE
        ]

        left_ankle = landmarks[
            self.mp_pose.PoseLandmark.LEFT_ANKLE
        ]

        right_ankle = landmarks[
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

        # ========================================================
        # VISIBILITY
        # ========================================================

        important_landmarks = [
            left_hip,
            right_hip,
            left_knee,
            right_knee,
            left_ankle,
            right_ankle
        ]

        visibility_ok = all(
            lm.visibility >=
            self.VISIBILITY_THRESHOLD
            for lm in important_landmarks
        )

        if not visibility_ok:

            return {
                "exercise": "NO EXERCISE",
                "reps": self.rep_count,
                "form": "Poor visibility",
                "feedback": "Keep your full body visible",
                "calibrated": self.calibrated,
                "frame": frame
            }

        # ========================================================
        # KNEE ANGLES
        # ========================================================

        left_angle = self.calculate_angle(
            self.get_point(left_hip),
            self.get_point(left_knee),
            self.get_point(left_ankle)
        )

        right_angle = self.calculate_angle(
            self.get_point(right_hip),
            self.get_point(right_knee),
            self.get_point(right_ankle)
        )

        if left_angle is None or right_angle is None:

            return {
                "exercise": "NO EXERCISE",
                "reps": self.rep_count,
                "form": "Waiting",
                "feedback": "Adjust your position",
                "calibrated": self.calibrated,
                "frame": frame
            }

        average_knee_angle = (
            left_angle +
            right_angle
        ) / 2

        # ========================================================
        # HIP
        # ========================================================

        average_hip_y = (
            left_hip.y +
            right_hip.y
        ) / 2

        # ========================================================
        # SMOOTH KNEE
        # ========================================================

        self.angle_history.append(
            average_knee_angle
        )

        if len(self.angle_history) > self.SMOOTHING_FRAMES:
            self.angle_history.pop(0)

        smooth_angle = np.mean(
            self.angle_history
        )

        # ========================================================
        # SMOOTH HIP
        # ========================================================

        self.hip_history.append(
            average_hip_y
        )

        if len(self.hip_history) > self.SMOOTHING_FRAMES:
            self.hip_history.pop(0)

        smooth_hip = np.mean(
            self.hip_history
        )

        # ========================================================
        # CALIBRATION
        # ========================================================

        if not self.calibrated:

            elapsed = (
                time.time() -
                self.calibration_start
            )

            self.standing_angles.append(
                smooth_angle
            )

            self.standing_hip_positions.append(
                smooth_hip
            )

            if elapsed < self.CALIBRATION_TIME:

                self.feedback = (
                    "Stand normally..."
                )

            else:

                if len(
                    self.standing_angles
                ) > 10:

                    self.baseline_angle = np.median(
                        self.standing_angles
                    )

                    self.baseline_hip = np.median(
                        self.standing_hip_positions
                    )

                    self.calibrated = True

                    self.feedback = (
                        "Calibration completed"
                    )

        # ========================================================
        # DETECTION
        # ========================================================

        else:

            down_threshold = (
                self.baseline_angle -
                self.DOWN_ANGLE_DROP
            )

            up_threshold = (
                self.baseline_angle -
                self.UP_ANGLE_DROP
            )

            hip_drop = (
                smooth_hip -
                self.baseline_hip
            )

            # ====================================================
            # UP → DOWN
            # ====================================================

            if self.state == "UP":

                if (
                    smooth_angle <
                    down_threshold
                    and
                    hip_drop >
                    self.HIP_DROP_MIN
                ):

                    self.state = "DOWN"

                    self.feedback = (
                        "Squat detected - go up"
                    )

            # ====================================================
            # DOWN → UP
            # ====================================================

            elif self.state == "DOWN":

                if smooth_angle > up_threshold:

                    current_time = time.time()

                    if (
                        current_time -
                        self.last_rep_time
                        >
                        self.COOLDOWN_TIME
                    ):

                        self.rep_count += 1

                        self.last_rep_time = (
                            current_time
                        )

                    self.state = "UP"

                    self.feedback = (
                        "Good! Rep completed"
                    )

            # ====================================================
            # FORM
            # ====================================================

            if self.state == "DOWN":

                if smooth_angle > 125:

                    self.feedback = (
                        "Go a little lower"
                    )

                else:

                    self.feedback = (
                        "Good depth"
                    )

        # ========================================================
        # DRAW
        # ========================================================

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        # ========================================================
        # RETURN DATA
        # ========================================================

        return {
            "exercise": "SQUAT",
            "reps": self.rep_count,
            "form": (
                "GOOD"
                if self.state == "DOWN"
                and smooth_angle <= 125
                else "CHECK"
            ),
            "feedback": self.feedback,
            "state": self.state,
            "knee_angle": round(
                float(smooth_angle),
                2
            ),
            "calibrated": self.calibrated,
            "frame": frame
        }

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):

        self.rep_count = 0

        self.state = "UP"

        self.angle_history.clear()

        self.hip_history.clear()

        self.standing_angles.clear()

        self.standing_hip_positions.clear()

        self.calibrated = False

        self.calibration_start = time.time()

        self.last_rep_time = 0

        self.feedback = (
            "Stand still for calibration"
        )

        self.baseline_angle = None

        self.baseline_hip = None

    # ============================================================
    # CLEANUP
    # ============================================================

    def close(self):

        self.pose.close()