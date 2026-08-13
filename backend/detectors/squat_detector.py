import cv2
import mediapipe as mp
import numpy as np
import time


class SquatDetector:

    def __init__(self):

        # ========================================================
        # MEDIAPIPE SETUP
        # ========================================================

        self.mp_pose = mp.solutions.pose

        self.mp_draw = mp.solutions.drawing_utils

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

        # How much knee angle must decrease from standing
        self.DOWN_ANGLE_DROP = 45

        # How close to standing before completing rep
        self.UP_ANGLE_DROP = 15

        # Hip movement required
        self.HIP_DROP_MIN = 0.025

        # Minimum visibility of important landmarks
        self.VISIBILITY_THRESHOLD = 0.60

        # Prevent duplicate counting
        self.COOLDOWN_TIME = 0.6

        # ========================================================
        # VARIABLES
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
    # ANGLE FUNCTION
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

        angle = np.degrees(
            np.arccos(cosine_angle)
        )

        return angle

    # ============================================================
    # LANDMARK → POINT
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

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # BGR → RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe
        results = self.pose.process(rgb)

        # ========================================================
        # NO PERSON DETECTED
        # ========================================================

        if not results.pose_landmarks:

            return {
                "exercise": "NO EXERCISE",
                "reps": self.rep_count,
                "form": "Waiting",
                "feedback": "NO PERSON DETECTED",
                "state": self.state,
                "calibrated": self.calibrated,
                "frame": frame
            }

        landmarks = results.pose_landmarks.landmark

        # ========================================================
        # IMPORTANT LANDMARKS
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
        # SIDE / FRONT VIEW VISIBILITY
        # ========================================================

        left_visible = (
            left_hip.visibility >= self.VISIBILITY_THRESHOLD
            and
            left_knee.visibility >= self.VISIBILITY_THRESHOLD
            and
            left_ankle.visibility >= self.VISIBILITY_THRESHOLD
        )

        right_visible = (
            right_hip.visibility >= self.VISIBILITY_THRESHOLD
            and
            right_knee.visibility >= self.VISIBILITY_THRESHOLD
            and
            right_ankle.visibility >= self.VISIBILITY_THRESHOLD
        )

        # At least one complete side must be visible
        visibility_ok = left_visible or right_visible

        if not visibility_ok:

            return {
                "exercise": "NO EXERCISE",
                "reps": self.rep_count,
                "form": "Poor visibility",
                "feedback": "Keep your full body visible",
                "state": self.state,
                "calibrated": self.calibrated,
                "frame": frame
            }

        # ========================================================
        # KNEE ANGLES
        # ========================================================

        angles = []

        # LEFT SIDE
        if left_visible:

            left_angle = self.calculate_angle(
                self.get_point(left_hip),
                self.get_point(left_knee),
                self.get_point(left_ankle)
            )

            if left_angle is not None:
                angles.append(left_angle)

        # RIGHT SIDE
        if right_visible:

            right_angle = self.calculate_angle(
                self.get_point(right_hip),
                self.get_point(right_knee),
                self.get_point(right_ankle)
            )

            if right_angle is not None:
                angles.append(right_angle)

        # No valid angle
        if not angles:

            return {
                "exercise": "NO EXERCISE",
                "reps": self.rep_count,
                "form": "Waiting",
                "feedback": "Adjust your position",
                "state": self.state,
                "calibrated": self.calibrated,
                "frame": frame
            }

        # If both legs are visible:
        # average them.
        #
        # If one side is visible:
        # use that side.
        average_knee_angle = float(
            np.mean(angles)
        )

        # ========================================================
        # HIP POSITION
        # ========================================================

        hip_values = []

        if left_visible:
            hip_values.append(left_hip.y)

        if right_visible:
            hip_values.append(right_hip.y)

        average_hip_y = float(
            np.mean(hip_values)
        )

        # ========================================================
        # SMOOTH KNEE ANGLE
        # ========================================================

        self.angle_history.append(
            average_knee_angle
        )

        if len(self.angle_history) > self.SMOOTHING_FRAMES:

            self.angle_history.pop(0)

        smooth_angle = float(
            np.mean(self.angle_history)
        )

        # ========================================================
        # SMOOTH HIP
        # ========================================================

        self.hip_history.append(
            average_hip_y
        )

        if len(self.hip_history) > self.SMOOTHING_FRAMES:

            self.hip_history.pop(0)

        smooth_hip = float(
            np.mean(self.hip_history)
        )

        # ========================================================
        # CALIBRATION
        # ========================================================
        form = "CHECK"
        
        if not self.calibrated:

            elapsed = (
                time.time()
                - self.calibration_start
            )

            self.standing_angles.append(
                smooth_angle
            )

            self.standing_hip_positions.append(
                smooth_hip
            )

            remaining = (
                self.CALIBRATION_TIME
                - elapsed
            )

            if remaining > 0:

                self.feedback = (
                    "Stand normally..."
                )

            else:

                if len(
                    self.standing_angles
                ) > 10:

                    self.baseline_angle = float(
                        np.median(
                            self.standing_angles
                        )
                    )

                    self.baseline_hip = float(
                        np.median(
                            self.standing_hip_positions
                        )
                    )

                    self.calibrated = True

                    self.state = "UP"

                    self.feedback = (
                        "Calibration completed"
                    )

        # ========================================================
        # DETECTION
        # ========================================================

        else:

            # Adaptive thresholds
            down_threshold = (
                self.baseline_angle
                - self.DOWN_ANGLE_DROP
            )

            up_threshold = (
                self.baseline_angle
                - self.UP_ANGLE_DROP
            )

            # Hip movement
            hip_drop = (
                smooth_hip
                - self.baseline_hip
            )

            # ====================================================
            # UP → DOWN
            # ====================================================

            if self.state == "UP":

                if (
                    smooth_angle < down_threshold
                    and
                    hip_drop > self.HIP_DROP_MIN
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
                        current_time
                        - self.last_rep_time
                        > self.COOLDOWN_TIME
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
            # FORM FEEDBACK
            # ====================================================

            if self.state == "DOWN":

                if smooth_angle > 125:

                    form = "CHECK"

                    self.feedback = (
                        "Go a little lower"
                    )

                else:

                    form = "GOOD"

                    self.feedback = (
                        "Good depth"
                    )

            else:

                form = "GOOD"

        # ========================================================
        # DRAW LANDMARKS
        # ========================================================

        self.mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        # ========================================================
        # OPTIONAL DEBUG UI
        # ========================================================

        # cv2.putText(
        #     frame,
        #     f"Knee: {smooth_angle:.1f}",
        #     (20, 35),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.65,
        #     (255, 255, 0),
        #     2
        # )

        # cv2.putText(
        #     frame,
        #     f"State: {self.state}",
        #     (20, 70),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.70,
        #     (0, 255, 255),
        #     2
        # )

        # cv2.putText(
        #     frame,
        #     f"SQUATS: {self.rep_count}",
        #     (20, 110),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.90,
        #     (0, 255, 0),
        #     3
        # )

        # cv2.putText(
        #     frame,
        #     self.feedback,
        #     (20, 150),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.62,
        #     (0, 255, 255),
        #     2
        # )

        # if self.calibrated:

        #     cv2.putText(
        #         frame,
        #         "CALIBRATED",
        #         (20, 190),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.65,
        #         (0, 255, 0),
        #         2
        #     )

        # else:

        #     remaining = max(
        #         0,
        #         self.CALIBRATION_TIME
        #         - (
        #             time.time()
        #             - self.calibration_start
        #         )
        #     )

            # cv2.putText(
            #     frame,
            #     f"CALIBRATING: {remaining:.1f}s",
            #     (20, 190),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.65,
            #     (0, 165, 255),
            #     2
            # )

        # ========================================================
        # RETURN DATA
        # ========================================================

        return {
            "exercise": "SQUAT",
            "reps": self.rep_count,
            "form": form,
            "feedback": self.feedback,
            "state": self.state,
            "knee_angle": round(
                smooth_angle,
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