import time


class FitnessEngine:

    def __init__(self):
        self.exercise = "NO EXERCISE"
        self.reps = 0
        self.form = "Waiting"
        self.feedback = "Start your workout"
        self.start_time = None
        self.is_active = False

    def start_workout(self):

        self.exercise = "NO EXERCISE"
        self.reps = 0
        self.form = "Waiting"
        self.feedback = "Workout started"

        self.start_time = time.time()
        self.is_active = True

    def stop_workout(self):

        self.is_active = False

        duration = 0

        if self.start_time:
            duration = time.time() - self.start_time

        return self.get_status(duration)

    def update(
        self,
        exercise=None,
        reps=None,
        form=None,
        feedback=None
    ):

        if exercise is not None:
            self.exercise = exercise

        if reps is not None:
            self.reps = reps

        if form is not None:
            self.form = form

        if feedback is not None:
            self.feedback = feedback

        duration = 0

        if self.start_time:
            duration = time.time() - self.start_time

        return self.get_status(duration)

    def get_status(self, duration=None):

        if duration is None:
            duration = 0

            if self.start_time:
                duration = time.time() - self.start_time

        return {
            "exercise": self.exercise,
            "reps": self.reps,
            "form": self.form,
            "feedback": self.feedback,
            "duration": round(duration, 2)
        }