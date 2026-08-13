class CalorieEngine:

    def __init__(self, met_model=None):
        self.met_model = met_model

    def predict_met(
        self,
        age,
        exercise_name,
        exercise_rep,
        duration
    ):
        """
        Returns MET predicted by the friend's model.

        duration is in seconds.
        """

        if self.met_model is None:
            raise ValueError("MET model is not loaded.")

        # --------------------------------------------------
        # CHANGE ONLY THIS PART IF YOUR MODEL USES A
        # DIFFERENT INPUT FORMAT.
        # --------------------------------------------------

        features = [[
            age,
            exercise_name,
            exercise_rep,
            duration
        ]]

        met = self.met_model.predict(features)[0]

        return float(met)

    def calculate_calories(
        self,
        met,
        weight_kg,
        duration_seconds
    ):
        """
        Standard MET-based calorie estimation.

        kcal = MET × 3.5 × weight(kg) / 200 × duration(minutes)
        """

        duration_minutes = duration_seconds / 60.0

        calories = (
            met
            * 3.5
            * weight_kg
            / 200.0
            * duration_minutes
        )

        return round(calories, 2)