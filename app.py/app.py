import streamlit as st
import pandas as pd
import joblib

joblib.load('KNN_heart.pkl')
joblib.load('scaler.pkl')
joblib.load('columns.pkl')

st.title("Heart Disease Prediction")
st.markdown("Provide the following details")

age=st.slider("Age", 10, 100, 40)
sex=st.selectbox("SEX", ['M', 'F'])
chest_pain_type=st.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
resting_bp=st.slider("Resting Blood Pressure", 80, 200, 120)
cholestrol =st.number_input("Cholestrol (mg/dl)", 100, 400, 200)
fasting_blood_sugar=st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['True', 'False'])
resting_ecg=st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_heart_rate=st.slider("Max Heart Rate", 60, 220,150)
excercise_angina=st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
oldpeak=st.slider("Oldpeak", 0.0, 10.0, 1.0)
st_slope=st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    raw_data = {
        'age': age,
        'ReastingBP' : resting_bp,
        'Cholesterol' : cholestrol,
        'MaxHR' : max_heart_rate,
        'Oldpeak' : oldpeak,
        'sex' + sex:1,
        'chest_pain_type' + chest_pain_type:1,
        'fasting_blood_sugar' + fasting_blood_sugar:1,
        'resting_ecg' + resting_ecg:1,
        'excercise_angina' + excercise_angina:1,
        'ST_slope' + st_slope:1
    }

    input_df = pd.DataFrame([raw_data])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    input_df_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df_scaled)[0]
    if prediction == 1:
        st.error("The model predicts that you have heart disease.")
    else:
        st.success("The model predicts that you do not have heart disease.")