import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Student Placement Predictor")

name = st.text_input("Enter your name")

tenth = st.number_input("10th Percentage", 0, 100)
twelfth = st.number_input("12th Percentage", 0, 100)
cgpa = st.number_input("CGPA", 0.0, 10.0)
internship = st.selectbox("Internship Experience (0 = No, 1 = Yes)", [0, 1])
communication = st.slider("Communication Skills (1-10)", 1, 10)
technical = st.slider("Technical Skills (1-10)", 1, 10)

if st.button("Predict"):
    features = np.array([[tenth, twelfth, cgpa, internship, communication, technical]])
    prediction = model.predict(features)[0]
    result = "Placed" if prediction == 1 else "Not Placed"
    st.success(f"{name}, your prediction: {result}")