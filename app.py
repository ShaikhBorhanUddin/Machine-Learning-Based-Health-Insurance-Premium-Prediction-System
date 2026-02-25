import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

model = joblib.load("xgboost_model.joblib")
feature_names = joblib.load("feature_names.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []

    for feature in feature_names:
        value = float(request.form.get(feature))
        input_data.append(value)

    X = np.array(input_data).reshape(1, -1)
    prediction = model.predict(X)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated Annual Premium: ${prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
