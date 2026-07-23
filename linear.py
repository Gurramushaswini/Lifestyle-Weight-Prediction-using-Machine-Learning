import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Lifestyle Weight Tracker")
st.title("🏋️ Lifestyle Weight Tracker")

st.write("Enter the details below:")

initial_weight = st.number_input(
    "Initial Weight (kg)", min_value=20.0, max_value=200.0, value=70.0
)

weight_change = st.number_input(
    "Weight Change (kg)", value=0.0
)

protein = st.number_input(
    "Protein (g)", min_value=0.0, value=50.0
)

carbs = st.number_input(
    "Carbs (g)", min_value=0.0, value=200.0
)

fat = st.number_input(
    "Fat (g)", min_value=0.0, value=60.0
)

if st.button("Predict Current Weight"):

    data = np.array([[initial_weight,
                      weight_change,
                      protein,
                      carbs,
                      fat]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Predicted Current Weight: {prediction[0]:.2f} kg")