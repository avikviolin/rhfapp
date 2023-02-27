#!/usr/bin/env python3.9

from flask import Flask, render_template, request
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
model = joblib.load('model.pkl')
explainer = joblib.load('explainer.pkl')


@app.route('/', methods=['GET'])
def form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def predict():

    intermacs = float(request.form['intermacs']) if request.form['intermacs'] else 6
    meld = float(request.form['meld']) if request.form['meld'] else 6
    inotropes = float(request.form['inotropes']) if request.form['inotropes'] else 0
    hemoglobin = float(request.form['hemoglobin']) if request.form['hemoglobin'] else 14
    race = float(request.form['race']) if request.form['race'] else 1
    smoking = float(request.form['smoking']) if request.form['smoking'] else 0
    rap = float(request.form['rap']) if request.form['rap'] else 7
    ast = float(request.form['ast']) if request.form['ast'] else 25
    bun = float(request.form['bun']) if request.form['bun'] else 15
    gfr = float(request.form['gfr']) if request.form['gfr'] else 100


    input_data = np.array([[intermacs, meld, inotropes, hemoglobin, race, smoking, rap, ast, bun, gfr]])
    prediction = str((model.predict_proba(input_data)[0][1]*100).round(2)) + '%'


    shap_values = explainer(input_data)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig('static/shap_plot.png', bbox_inches='tight')


    print(prediction)
    return render_template('prediction.html', prediction=prediction, shap_plot='shap_plot.png')


if __name__ == "__main__":
    app.run()
