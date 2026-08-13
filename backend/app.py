from flask import Flask, jsonify, request
from flask_cors import CORS

from fitness_engine import FitnessEngine


app = Flask(__name__)

CORS(app)

fitness = FitnessEngine()


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "message": "Burn-Ex backend is running",
        "service": "AI Fitness Assistant"
    })


# ============================================================
# START
# ============================================================

@app.route("/api/workout/start", methods=["POST"])
def start_workout():

    fitness.start_workout()

    return jsonify({
        "status": "success",
        "message": "Workout started",
        "data": fitness.get_status()
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/api/workout/status", methods=["GET"])
def workout_status():

    return jsonify({
        "status": "success",
        "data": fitness.get_status()
    })


# ============================================================
# PROCESS FRAME
# ============================================================

@app.route("/api/workout/frame", methods=["POST"])
def process_frame():

    # For this step, we'll connect the webcam
    # directly to Python first.
    #
    # So this endpoint is only a placeholder
    # for the frontend integration.

    return jsonify({
        "status": "success",
        "message": "Frame endpoint ready",
        "data": fitness.get_status()
    })


# ============================================================
# STOP
# ============================================================

@app.route("/api/workout/stop", methods=["POST"])
def stop_workout():

    result = fitness.stop_workout()

    return jsonify({
        "status": "success",
        "message": "Workout stopped",
        "data": result
    })


@app.route("/api/workout/exercise", methods=["POST"])
def set_exercise():

    data = request.get_json() or {}

    exercise = data.get("exercise")

    if not exercise:
        return jsonify({
            "status": "error",
            "message": "Exercise is required"
        }), 400

    try:

        fitness.set_exercise(exercise)

        return jsonify({
            "status": "success",
            "data": fitness.get_status()
        })

    except ValueError as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )