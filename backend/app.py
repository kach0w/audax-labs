import pandas as pd
import numpy as np
import pickle
import requests
import sklearn
from sklearn.metrics import accuracy_score, r2_score,mean_squared_error
from flask import Flask, request, jsonify# from utils import preprocessing
import os
model = pickle.load(open('model.pkl', 'rb'))
from utils import preprocessing


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/upload', methods=['POST'])
def handle_data():
    if request.method == 'POST':
        input_file = request.files['file']
        df = pd.read_csv(input_file, low_memory=False)
        df, has_col, error = preprocessing(df)
        if(not error):
            if(has_col):
                X = df.iloc[:, :-1]
                Y = (df.iloc[:, -1:])
                Y.replace((-1, 1), (0, 1), inplace=True)
                pred = model.predict(X)
                df["Predictions"] = pred
                accuracy = accuracy_score(Y, pred)
                rmse = np.sqrt(mean_squared_error(Y, pred))
                r2_square = r2_score(Y, pred)
                df_dict = df.sample(25).to_dict(orient='records')
                response_data = {
                    "df": df_dict,
                    "Good/Bad": has_col,
                    "accuracy": accuracy,
                    "rmse": rmse,
                    "r2_square": r2_square
                }
                return jsonify(response_data)
            else:
                X = df
                pred = model.predict(X)
                df["Prediction"] = pred
                return jsonify({"df": df})
                
        else:
            return jsonify({"message": "Your data did not contain all of the necessary features: ['Sensor-57', 'Sensor-134', 'Sensor-76', 'Sensor-28', 'Sensor-164', 'Sensor-369', 'Sensor-108', 'Sensor-81', 'Sensor-449', 'Sensor-319']"})
    else:
        return jsonify({"message": "Not Post Method"})  
if __name__ == '__main__':
    app.run(debug=True)


# if upload is not None:
    
#     