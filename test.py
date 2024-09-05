import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib

# Load your logistic regression model
lr_model = joblib.load('heartdisease_logisticregression.joblib')

# Streamlit application starts here
def main():
    # Title of your web app
    st.title('Heart Disease Prediction')

    # Collect user input
    age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
    sex = st.selectbox("Select your sex:", ("Male", "Female"))
    chest_pain_type = st.selectbox("Select chest pain type:", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
    resting_bp = st.number_input("Enter resting blood pressure (mm Hg):", min_value=50, max_value=250, step=1)
    cholesterol = st.number_input("Enter cholesterol (mg/dL):", min_value=100, max_value=600, step=1)
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL:", (0, 1))
    resting_ecg = st.selectbox("Select resting ECG result:", ("Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"))
    max_hr = st.number_input("Enter maximum heart rate achieved:", min_value=50, max_value=220, step=1)
    exercise_angina = st.selectbox("Do you have exercise-induced angina?", ("Yes", "No"))
    oldpeak = st.number_input("Enter oldpeak (ST depression):", min_value=0.0, max_value=10.0, step=0.1)
    st_slope = st.selectbox("Select the slope of the peak exercise ST segment:", ("Up", "Flat", "Down"))

    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    chest_pain_type_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    chest_pain_type = chest_pain_type_mapping[chest_pain_type]
    resting_ecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    resting_ecg = resting_ecg_mapping[resting_ecg]
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope_mapping = {"Up": 0, "Flat": 1, "Down": 2}
    st_slope = st_slope_mapping[st_slope]

    # Create a pandas DataFrame from the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

   # Ensure all columns are numeric by converting to float
    input_data = input_data.astype(float)

    # Check the input data structure
    st.write("Input Data (Pandas DataFrame):")
    st.write(input_data)

  # # Check for missing values or invalid inputs
  #   if np.isnan(input_data).any():
  #       st.write("Warning: Some input fields are missing or invalid.")
  #   else:
        # When the user clicks the 'Predict' button, make the prediction
        if st.button("Predict Heart Disease"):
            try:
                prediction = lr_model.predict(input_data)
                # Show the result
                if prediction[0] == 1:
                    st.write("The model predicts that this person has heart disease.")
                else:
                    st.write("The model predicts that this person does not have heart disease.")
            except Exception as e:
                st.write(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
