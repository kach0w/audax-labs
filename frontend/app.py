import streamlit as st
import requests
import pandas as pd
import json

st.write("# Audax Labs - Water Fault Detection")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("Uploaded file:", uploaded_file.name)

    backend_url = 'http://localhost:5000/api/upload'
    files = {'file': uploaded_file.getvalue()}
    response = requests.post(backend_url, files=files)
    # df = pd.read_json(response.json()["df"], orient='records')
    df_json = response.json()["df"]
    has_col = response.json()["Good/Bad"]
    accuracy = response.json()["accuracy"]
    rmse = response.json()["rmse"]
    r2_square = response.json()["r2_square"]
    
    df = pd.read_json(json.dumps(df_json), orient='records')
    result_csv = df.to_csv().encode("utf-8")
    st.write(f"### Dataframe")
    st.write(df)
    st.download_button(
        label=":green[Download prediction as CSV]",
        data=result_csv,
        file_name='prediction.csv',
        mime='text/csv',
    )   
    st.write(f"Contains 'Good/Bad' Column: *{has_col}*")   
    if accuracy:
        st.write(f"Accuracy: *{accuracy}*")
        st.write(f"Mean Squared Error: *{rmse}*")
        st.write(f"R^2: *{r2_square}*")
    
