import streamlit as st
import pandas as pd
import pickle

st.title("🎓 Student Placement Prediction System")

# Load the trained model
model = pickle.load(open("student_model.pkl", "rb"))

st.header("Enter Student Details:")

# Input fields
tenth = st.number_input("10th Score", min_value=0, max_value=100, value=80)
twelfth = st.number_input("12th Score", min_value=0, max_value=100, value=85)
cgpa = st.number_input("Degree CGPA", min_value=0.0, max_value=10.0, value=8.0)
internship = st.selectbox("Internship Done?", [0, 1])
communication = st.number_input("Communication Score", min_value=0, max_value=10, value=7)
technical = st.number_input("Technical Score", min_value=0, max_value=10, value=7)

# Predict button
if st.button("Predict"):
    data = pd.DataFrame({
        "10th": [tenth],
        "12th": [twelfth],
        "CGPA": [cgpa],
        "Internship": [internship],
        "Communication": [communication],
        "Technical": [technical]
    })
    prediction = model.predict(data)
    
    if prediction[0] == 1:
        st.success("✅ The student is likely to get placed")
    else:
        st.warning("❌ The student may not get placed")