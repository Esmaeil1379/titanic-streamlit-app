import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Title
st.title("🚢 Titanic Survival Prediction")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
fare = st.slider("Fare", 0.0, 500.0, 50.0)

# Convert sex to 0/1
sex_binary = 1 if sex == "male" else 0

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pclass, sex_binary, age, fare]])
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.success(f"Prediction: {result}")
