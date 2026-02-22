import pickle
import pandas as pd  # Use pandas to include feature names

# Load the saved model
model = pickle.load(open("student_model.pkl", "rb"))

# Example new student data as DataFrame with column names
# The column names must match your training dataset exactly
new_student = pd.DataFrame({
    "10th": [82],
    "12th": [85],
    "CGPA": [8.0],
    "Internship": [1],
    "Communication": [7],
    "Technical": [7]
})

# Predict
prediction = model.predict(new_student)

# Show result
if prediction[0] == 1:
    print("✅ The student is likely to get placed")
else:
    print("❌ The student may not get placed")