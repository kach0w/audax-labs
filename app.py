import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.metrics import accuracy_score, r2_score,mean_squared_error

from utils import preprocessing

model = pickle.load(open('model.pkl', 'rb'))

# gonna do a file upload or enter top 10 
# have to do preprocessing on both

# top 10 sensors (on correlation)
# ['Sensor-37',
#  'Sensor-393',
#  'Sensor-577',
#  'Sensor-413',
#  'Sensor-543',
#  'Sensor-254',
#  'Sensor-126',
#  'Sensor-22',
#  'Sensor-299',
#  'Sensor-181']

st.title('Audax Labs - Water Fault Detection')
upload = st.file_uploader("Choose a file (.csv)")

if upload is not None:
    df, has_col, error = preprocessing(upload)
    if(not error):
        if(has_col):
            # st.write(df.shape)
            X = df.iloc[:, :-1]
            Y = (df.iloc[:, -1:])
            Y.replace((-1, 1), (0, 1), inplace=True)
            pred = model.predict(X)
            df["Predictions"] = pred
           
            st.write(f"### Dataframe")
            st.write(df)
            st.write(f"Contains 'Good/Bad' Column: *{has_col}*")   
            
            accuracy = accuracy_score(Y, pred)
            rmse = np.sqrt(mean_squared_error(Y, pred))
            r2_square = r2_score(Y, pred)
            
            st.write(f"Accuracy: *{accuracy}*")
            st.write(f"Mean Squared Error: *{rmse}*")
            st.write(f"R^2: *{r2_square}*")

        else:
            X = df
            pred = model.predict(X)
            df["Prediction"] = pred
            result_csv = df.to_csv().encode("utf-8")
            
            st.write(f"### Dataframe")
            st.write(df)
            st.download_button(
                label="Download prediction as CSV",
                data=result_csv,
                file_name='prediction.csv',
                mime='text/csv',
            )    
    else:
        st.write("Your data did not contain all of the necessary features: ['Sensor-57', 'Sensor-134', 'Sensor-76', 'Sensor-28', 'Sensor-164', 'Sensor-369', 'Sensor-108', 'Sensor-81', 'Sensor-449', 'Sensor-319']")
     