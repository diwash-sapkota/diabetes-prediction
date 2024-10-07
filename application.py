import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd


application = Flask(__name__)
app = application


#importing the pickle file
nb_model = pickle.load(open('models/naive_bayes_model.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method=="POST":
        Pregnancies = int(request.form.get('Pregnancies')) 
        Glucose = int(request.form.get('Glucose')) 
        BloodPressure = int(request.form.get('BloodPressure')) 
        SkinThickness = int(request.form.get('SkinThickness')) 
        Insulin = int(request.form.get('Insulin')) 
        BMI = float(request.form.get('BMI')) 
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction')) 
        Age = int(request.form.get('Age')) 

        # Reshape the input to 2D array
        input_features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result = nb_model.predict(input_features)
        return render_template('predict.html', results=result[0])

    else:
        return render_template('predict.html')



if __name__=="__main__":
    app.run(debug=True)
