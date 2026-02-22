import pickle
import pandas as pd

# Load the saved model
model = pickle.load(open("student_model.pkl", "rb"))

# Load new student data from CSV
df = pd.read_csv("new_students.csv")

# Predict placement for all students
df['Placement_Prediction'] = model.predict(df)

# Show results
print(df)