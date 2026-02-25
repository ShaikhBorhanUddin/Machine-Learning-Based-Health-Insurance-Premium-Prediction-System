from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("xgboost_model.joblib")
feature_names = joblib.load("feature_names.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        input_data = {}

        for feature in feature_names:
            input_data[feature] = float(request.form.get(feature, 0))

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
