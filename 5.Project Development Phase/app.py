from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("rdf.pkl", "rb"))

scaler = pickle.load(open("scale1.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():

    try:

        Gender = float(request.form["Gender"])
        Married = float(request.form["Married"])
        Dependents = float(request.form["Dependents"])
        Education = float(request.form["Education"])
        Self_Employed = float(request.form["Self_Employed"])
        ApplicantIncome = float(request.form["ApplicantIncome"])
        CoapplicantIncome = float(request.form["CoapplicantIncome"])
        LoanAmount = float(request.form["LoanAmount"])
        Loan_Amount_Term = float(request.form["Loan_Amount_Term"])
        Credit_History = float(request.form["Credit_History"])
        Property_Area = float(request.form["Property_Area"])

        data = [[
            Gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Property_Area
        ]]

        data = scaler.transform(data)

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Rejected"

        return render_template(
            "submit.html",
            prediction=result
        )

    except Exception as e:

        return render_template(
            "submit.html",
            prediction="Error : " + str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)