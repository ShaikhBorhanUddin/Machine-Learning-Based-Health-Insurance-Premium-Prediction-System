from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ----------------------------
# Load model and feature schema
# ----------------------------
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    FEATURE_NAMES = pickle.load(f)


# ----------------------------
# Prepare feature vector
# ----------------------------
def prepare_features(form):
    """
    Converts user form input into model-ready numpy array
    Feature order strictly follows FEATURE_NAMES
    """
    values = []

    for feature in FEATURE_NAMES:
        val = form.get(feature)

        if val is None or val == "":
            raise ValueError(f"Missing input for feature: {feature}")

        values.append(float(val))

    return np.array(values).reshape(1, -1)


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            X = prepare_features(request.form)
            pred = model.predict(X)[0]
            prediction = round(float(pred), 2)
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        feature_names=FEATURE_NAMES
    )


# ----------------------------
# Run locally
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)