import pandas as pd
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
preprocessor = pickle.load(open("preprocessor.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods = ["POST"])
def predict():
    new_data = pd.DataFrame([{
        "Pregnancies": int(request.form["pregnancies"]),
        "Glucose": int(request.form["glucose"]),
        "BloodPressure": float(request.form["bloodpressure"]),
        "SkinThickness": float(request.form["skinthickness"]),
        "Insulin": float(request.form["insulin"]),
        "BMI": float(request.form["BMI"]),
        "DPF": float(request.form["DPF"]),
        "Age": int(request.form["age"]),
    }])

    processed = preprocessor.transform(new_data)
    pred = model.predict(processed)[0]

    result = "Chance of Diabetes"if pred==1 else "Person is Low Risk"
    return render_template("index.html",prediction_text = result)

if __name__ == "__main__":
    app.run(debug=True)