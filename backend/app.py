from flask import Flask, jsonify, request
from flask_cors import CORS

from fitness_engine import FitnessEngine


app = Flask(__name__)

CORS(app)

fitness = FitnessEngine()


@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "message": "Burn-Ex backend is running",
        "service": "AI Fitness Assistant"
    })


@app.route("/api/workout/start", methods=["POST"])
def start_workout():

    fitness.start_workout()

    return jsonify({
        "status": "success",
        "message": "Workout started",
        "data": fitness.get_status()
    })


@app.route("/api/workout/status", methods=["GET"])
def workout_status():

    return jsonify({
        "status": "success",
        "data": fitness.get_status()
    })


@app.route("/api/workout/update", methods=["POST"])
def update_workout():

    data = request.get_json() or {}

    result = fitness.update(
        exercise=data.get("exercise"),
        reps=data.get("reps"),
        form=data.get("form"),
        feedback=data.get("feedback")
    )

    return jsonify({
        "status": "success",
        "data": result
    })


@app.route("/api/workout/stop", methods=["POST"])
def stop_workout():

    result = fitness.stop_workout()

    return jsonify({
        "status": "success",
        "message": "Workout stopped",
        "data": result
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )