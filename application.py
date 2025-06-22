from flask import Flask,request, jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
application = Flask(__name__)
app=application

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            WS = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            features = [[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]]
            new_data_scaled = standard_scaler.transform(features)
            result = ridge_model.predict(new_data_scaled)
            return render_template('home.html', results=result[0])

        except (TypeError, ValueError) as e:
            return f"Input error: {e}", 400
        except Exception as e:
            return f"Something went wrong: {e}", 500

        return render_template('home.html')
    else:
        return render_template('home.html')


@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    application.run(host='0.0.0.0',debug=True)