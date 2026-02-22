import streamlit as st
import pickle
import numpy as np

# Load your pre-trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Student Placement Predictor")

# Input from user
name = st.text_input("Enter your name")
score = st.slider("Enter your score", 0, 100)

# Predict button
if st.button("Predict"):
    prediction = model.predict(np.array([[score]]))[0]
    result = "Placed" if prediction == 1 else "Not Placed"
    st.success(f"{name}, your prediction: {result}")