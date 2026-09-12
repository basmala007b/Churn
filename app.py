import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
features = joblib.load('features.pkl')
target_encoder = joblib.load('target_encoder.pkl')

st.title("Customer Churn Prediction")

columns = st.columns(3)

tenure = columns[0].number_input("Tenure", min_value=0, max_value=500)

TechSupport = columns[1].selectbox("TechSupport", ['No', 'Yes', 'No internet service'])

Contract = columns[2].selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])

df = pd.DataFrame([[tenure, TechSupport, Contract]], columns=features)

if st.button('predict'):

    for encode in encoder.keys():
        df[encode] = encoder[encode].transform(df[encode])

    predict = model.predict(df[features])
    result = target_encoder.inverse_transform(predict)
    