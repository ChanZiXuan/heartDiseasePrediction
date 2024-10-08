import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Load the logistic regression model with preprocessing steps included
lr_model = load('heartdisease_logisticregression.joblib')
scaler = load('oldscaler.joblib')

# Streamlit application starts here
def main():
    st.title('Heart Disease Prediction')

    # Collect user input
    age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
    sex = st.selectbox("Select your sex:", ("Male", "Female"))
    chest_pain_type = st.selectbox("Select chest pain type:", ("TA", "ATA", "NAP", "ASY"))
    resting_bp = st.number_input("Enter resting blood pressure (mm Hg):", min_value=50, max_value=250, step=1)
    cholesterol = st.number_input("Enter cholesterol (mg/dL):", min_value=100, max_value=600, step=1)
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL:", (0, 1))
    resting_ecg = st.selectbox("Select resting ECG result:", ("Normal", "ST", "LVH"))
    max_hr = st.number_input("Enter maximum heart rate achieved:", min_value=50, max_value=220, step=1)
    exercise_angina = st.selectbox("Do you have exercise-induced angina?", ("Yes", "No"))
    oldpeak = st.number_input("Enter oldpeak (ST depression):", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
    st_slope = st.selectbox("Select the slope of the peak exercise ST segment:", ("Up", "Flat", "Down"))

    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    chest_pain_type_mapping = {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}
    chest_pain_type = chest_pain_type_mapping[chest_pain_type]
    resting_ecg_mapping = {"Normal": 0, "ST": 1, "LVH": 2}
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
    
    # Apply scaling to the input data
    try:
        input_data_scaled = scaler.transform(input_data)
        st.write("Scaled Input Data (after applying scaler):")
        st.write(input_data_scaled)
    except Exception as e:
        st.write(f'An error occurred during scaling: {e}')
        return

    # When the user clicks the 'Predict' button, make the prediction
    if st.button("Predict Heart Disease"):
        try:
            # Predict using the model
            prediction = lr_model.predict(input_data_scaled)

            # Show the result
            if prediction[0] == 1:
                st.write('The model predicts that this person has heart disease.')
            else:
                st.write('The model predicts that this person does not have heart disease.')
        except Exception as e:
            st.write(f'An error occurred during prediction: {e}')

if __name__ == '__main__':
    main()
