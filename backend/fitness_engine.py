from detectors.squat_detector import SquatDetector
from detectors.pushup_detector import PushupDetector
from detectors.jumping_jacks_detector import JumpingJackDetector
from detectors.crunch_detector import CrunchDetector


class FitnessEngine:

    def __init__(self):

        # -----------------------------------------
        # DETECTORS
        # -----------------------------------------

        self.squat_detector = SquatDetector()
        self.pushup_detector = PushupDetector()
        self.jumping_jack_detector = JumpingJackDetector()
        self.crunch_detector = CrunchDetector()

        # -----------------------------------------
        # WORKOUT STATE
        # -----------------------------------------

        self.workout_active = False

        self.current_exercise = "NO EXERCISE"

        self.last_result = {
            "exercise": "NO EXERCISE",
            "reps": 0,
            "form": "Waiting",
            "feedback": "Start your workout"
        }

    # =============================================
    # START WORKOUT
    # =============================================

    def start_workout(self):

        self.workout_active = True

        self.current_exercise = "NO EXERCISE"

        self.squat_detector.reset()
        self.pushup_detector.reset()
        self.jumping_jack_detector.reset()
        self.crunch_detector.reset()

        self.last_result = {
            "exercise": "NO EXERCISE",
            "reps": 0,
            "form": "Waiting",
            "feedback": "Workout started"
        }

    # =============================================
    # PROCESS FRAME
    # =============================================

    def process_frame(self, frame):

        if not self.workout_active:

            return self.last_result

        # -----------------------------------------
        # IMPORTANT
        # -----------------------------------------
        # For now, use the exercise selected
        # manually. This keeps the verified
        # detectors unchanged.
        #
        # We will add automatic exercise
        # classification after this stage.
        # -----------------------------------------

        if self.current_exercise == "SQUAT":

            result = self.squat_detector.process_frame(frame)

        elif self.current_exercise == "PUSH-UP":

            result = self.pushup_detector.process_frame(frame)

        elif self.current_exercise == "JUMPING JACK":

            result = self.jumping_jack_detector.process_frame(frame)

        elif self.current_exercise == "CRUNCH":

            result = self.crunch_detector.process_frame(frame)

        else:

            return {
                "exercise": "NO EXERCISE",
                "reps": 0,
                "form": "Waiting",
                "feedback": "Select an exercise"
            }

        # -----------------------------------------
        # SAVE RESULT
        # -----------------------------------------

        self.last_result = {
            key: value
            for key, value in result.items()
            if key != "frame"
        }

        return result

    # =============================================
    # SELECT EXERCISE
    # =============================================

    def set_exercise(self, exercise):

        exercise = exercise.upper().strip()

        allowed = {
            "SQUAT",
            "PUSH-UP",
            "JUMPING JACK",
            "CRUNCH"
        }

        if exercise not in allowed:

            raise ValueError(
                f"Unsupported exercise: {exercise}"
            )

        self.current_exercise = exercise

        # Reset selected detector

        if exercise == "SQUAT":
            self.squat_detector.reset()

        elif exercise == "PUSH-UP":
            self.pushup_detector.reset()

        elif exercise == "JUMPING JACK":
            self.jumping_jack_detector.reset()

        elif exercise == "CRUNCH":
            self.crunch_detector.reset()

        self.last_result = {
            "exercise": exercise,
            "reps": 0,
            "form": "Waiting",
            "feedback": f"Ready for {exercise}"
        }

    # =============================================
    # STOP WORKOUT
    # =============================================

    def stop_workout(self):

        self.workout_active = False

        return self.last_result

    # =============================================
    # STATUS
    # =============================================

    def get_status(self):

        return self.last_result

    # =============================================
    # CLEANUP
    # =============================================

    def close(self):

        self.squat_detector.close()
        self.pushup_detector.close()
        self.jumping_jack_detector.close()
        self.crunch_detector.close()