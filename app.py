from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("student_model.pkl")

@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        study_hours = float(request.form["study_hours"])

        attendance = float(request.form["attendance"])

        previous_grades = float(request.form["previous_grades"])

        extracurricular = request.form["extracurricular"]

        parent_education = request.form["parent_education"]

        input_data = pd.DataFrame({

            "Study Hours per Week": [study_hours],

            "Attendance Rate": [attendance],

            "Previous Grades": [previous_grades],

            "Participation in Extracurricular Activities": [extracurricular],

            "Parent Education Level": [parent_education]

        })

        prediction = model.predict(input_data)

        prediction_value = prediction[0]

        if str(prediction_value).lower() in ["yes", "pass", "passed", "1"]:

            result = " Student Will PASS"

        elif str(prediction_value).lower() in ["no", "fail", "failed", "0"]:

            result = " Student Will FAIL"

        else:

            result = f"Prediction Result: {prediction_value}"

        return render_template(

            "index.html",

            prediction_text=result

        )

    except Exception as e:

        return render_template(

            "index.html",

            prediction_text=f"Error: {str(e)}"

        )

if __name__ == "__main__":

    app.run(

        debug=True,

        host="0.0.0.0",

        port=5000

    )