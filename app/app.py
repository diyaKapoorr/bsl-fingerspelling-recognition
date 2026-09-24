from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from pathlib import Path
import os

from inference import predict_bsl


# -------------------------
# APP SETUP
# -------------------------

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"

# Create upload folder automatically
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

app = Flask(__name__)


# -------------------------
# HOME PAGE
# -------------------------

@app.route("/")
def index():
    return render_template("index.html")


# -------------------------
# VIDEO PREDICTION
# -------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:
        return render_template(
            "index.html",
            error="No video was uploaded."
        )

    video = request.files["video"]

    if video.filename == "":
        return render_template(
            "index.html",
            error="Please select a video."
        )

    filename = secure_filename(
        video.filename
    )

    filepath = UPLOAD_DIR / filename

    # Save uploaded video temporarily
    video.save(filepath)

    try:

        predicted_label, confidence = predict_bsl(
            filepath
        )

        if predicted_label is None:
            return render_template(
                "index.html",
                error=(
                    "Not enough hand frames were "
                    "detected in the video."
                )
            )

        return render_template(
            "index.html",
            prediction=predicted_label,
            confidence=confidence
        )

    finally:

        # Remove uploaded video after prediction
        if filepath.exists():
            os.remove(filepath)


# -------------------------
# RUN APP
# -------------------------

if __name__ == "__main__":
    app.run(debug=True)