import time

from detectors.squat_detector import SquatDetector
from detectors.pushup_detector import PushupDetector
from detectors.jumping_jacks_detector import JumpingJackDetector
from detectors.crunch_detector import CrunchDetector


class FitnessEngine:

    def __init__(self):

        # ==========================================
        # DETECTORS
        # ==========================================

        self.squat_detector = SquatDetector()
        self.pushup_detector = PushupDetector()
        self.jumping_jack_detector = JumpingJackDetector()
        self.crunch_detector = CrunchDetector()

        # ==========================================
        # WORKOUT STATE
        # ==========================================

        self.workout_active = False
        self.paused = False

        self.current_exercise = "NO EXERCISE"

        # ==========================================
        # SESSION TIMER
        # ==========================================

        self.session_start_time = None
        self.session_duration = 0.0

        self.pause_start_time = None
        self.total_paused_time = 0.0

        # ==========================================
        # ACTIVE EXERCISE TIMER
        # ==========================================

        self.active_start_time = None
        self.active_duration = 0.0
        self.exercise_started = False

        # ==========================================
        # RESULT
        # ==========================================

        self.last_result = {
            "exercise": "NO EXERCISE",
            "reps": 0,
            "duration": 0.0,
            "session_duration": 0.0,
            "form": "Waiting",
            "feedback": "Start your workout",
            "paused": False
        }

    # ==================================================
    # START WORKOUT
    # ==================================================

    def start_workout(self):

        self.workout_active = True
        self.paused = False

        self.current_exercise = "NO EXERCISE"

        self.session_start_time = time.time()
        self.session_duration = 0.0

        self.pause_start_time = None
        self.total_paused_time = 0.0

        self.active_start_time = None
        self.active_duration = 0.0
        self.exercise_started = False

        self.squat_detector.reset()
        self.pushup_detector.reset()
        self.jumping_jack_detector.reset()
        self.crunch_detector.reset()

        self.last_result = {
            "exercise": "NO EXERCISE",
            "reps": 0,
            "duration": 0.0,
            "session_duration": 0.0,
            "form": "Waiting",
            "feedback": "Workout started",
            "paused": False
        }

    # ==================================================
    # SELECT EXERCISE
    # ==================================================

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

        # Stop previous active timer
        self._stop_active_timer()

        self.current_exercise = exercise

        self.active_start_time = None
        self.active_duration = 0.0
        self.exercise_started = False

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
            "duration": 0.0,
            "session_duration": self._get_session_duration(),
            "form": "Waiting",
            "feedback": f"Ready for {exercise}",
            "paused": self.paused
        }

    # ==================================================
    # PAUSE
    # ==================================================

    def pause_workout(self):

        if not self.workout_active:
            return self.get_status()

        if self.paused:
            return self.get_status()

        self.paused = True

        self.pause_start_time = time.time()

        # Freeze active exercise timer
        if self.exercise_started:
            self._stop_active_timer()

        self.last_result["paused"] = True
        self.last_result["feedback"] = "Workout paused"

        return self.get_status()

    # ==================================================
    # RESUME
    # ==================================================

    def resume_workout(self):

        if not self.workout_active:
            return self.get_status()

        if not self.paused:
            return self.get_status()

        now = time.time()

        paused_duration = (
            now - self.pause_start_time
        )

        self.total_paused_time += paused_duration

        self.pause_start_time = None

        self.paused = False

        self.last_result["paused"] = False
        self.last_result["feedback"] = "Workout resumed"

        return self.get_status()

    # ==================================================
    # ACTIVE MOVEMENT CHECK
    # ==================================================

    def _is_active_movement(self, result):

        state = result.get("state")

        if self.current_exercise == "SQUAT":
            return state == "DOWN"

        if self.current_exercise == "PUSH-UP":
            return state == "DOWN"

        if self.current_exercise == "JUMPING JACK":
            return state == "OPEN"

        if self.current_exercise == "CRUNCH":
            return state == "UP"

        return False

    # ==================================================
    # START ACTIVE TIMER
    # ==================================================

    def _start_active_timer(self):

        if not self.exercise_started and not self.paused:

            self.active_start_time = time.time()
            self.exercise_started = True

    # ==================================================
    # STOP ACTIVE TIMER
    # ==================================================

    def _stop_active_timer(self):

        if (
            self.exercise_started
            and self.active_start_time is not None
        ):

            self.active_duration += (
                time.time()
                - self.active_start_time
            )

        self.active_start_time = None
        self.exercise_started = False

    # ==================================================
    # GET ACTIVE DURATION
    # ==================================================

    def _get_active_duration(self):

        duration = self.active_duration

        if (
            self.exercise_started
            and self.active_start_time is not None
            and not self.paused
        ):

            duration += (
                time.time()
                - self.active_start_time
            )

        return duration

    # ==================================================
    # GET SESSION DURATION
    # ==================================================

    def _get_session_duration(self):

        if self.session_start_time is None:
            return self.session_duration

        if self.workout_active:

            current = time.time()

            paused_time = self.total_paused_time

            # Current pause also excluded
            if self.paused and self.pause_start_time:
                paused_time += (
                    current -
                    self.pause_start_time
                )

            return (
                current
                - self.session_start_time
                - paused_time
            )

        return self.session_duration

    # ==================================================
    # PROCESS FRAME
    # ==================================================

    def process_frame(self, frame):

        if not self.workout_active:
            return {
                **self.last_result,
                "frame": frame
            }

        # When paused, DON'T run detectors
        if self.paused:

            result = {
                **self.last_result,
                "paused": True,
                "feedback": "Workout paused",
                "frame": frame
            }

            return result

        # ----------------------------------------------
        # Exercise must be selected
        # ----------------------------------------------

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
                "duration": 0.0,
                "session_duration": round(
                    self._get_session_duration(),
                    2
                ),
                "form": "Waiting",
                "feedback": "Select an exercise",
                "paused": False,
                "frame": frame
            }

        # ----------------------------------------------
        # Active movement timer
        # ----------------------------------------------

        if self._is_active_movement(result):
            self._start_active_timer()

        active_duration = self._get_active_duration()
        session_duration = self._get_session_duration()

        # ----------------------------------------------
        # Build result
        # ----------------------------------------------

        self.last_result = {
            "exercise": result.get(
                "exercise",
                self.current_exercise
            ),

            "reps": result.get(
                "reps",
                0
            ),

            "duration": round(
                active_duration,
                2
            ),

            "session_duration": round(
                session_duration,
                2
            ),

            "form": result.get(
                "form",
                "Waiting"
            ),

            "feedback": result.get(
                "feedback",
                ""
            ),

            "state": result.get(
                "state",
                None
            ),

            "paused": False
        }

        return {
            **self.last_result,
            "frame": result.get(
                "frame",
                frame
            )
        }

    # ==================================================
    # STOP WORKOUT
    # ==================================================

    def stop_workout(self):

        if self.paused and self.pause_start_time:

            self.total_paused_time += (
                time.time()
                - self.pause_start_time
            )

            self.pause_start_time = None

        self._stop_active_timer()

        self.session_duration = (
            self._get_session_duration()
        )

        self.workout_active = False
        self.paused = False

        self.last_result["duration"] = round(
            self.active_duration,
            2
        )

        self.last_result["session_duration"] = round(
            self.session_duration,
            2
        )

        self.last_result["paused"] = False

        return self.last_result

    # ==================================================
    # STATUS
    # ==================================================

    def get_status(self):

        return {
            **self.last_result,
            "duration": round(
                self._get_active_duration(),
                2
            ),
            "session_duration": round(
                self._get_session_duration(),
                2
            ),
            "paused": self.paused
        }

    # ==================================================
    # CLEANUP
    # ==================================================

    def close(self):

        self.squat_detector.close()
        self.pushup_detector.close()
        self.jumping_jack_detector.close()
        self.crunch_detector.close()