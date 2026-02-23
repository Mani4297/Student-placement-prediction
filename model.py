import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("placement_data.csv")

# Use all input features
X = data[["10th", "12th", "CGPA", "Internship", "Communication", "Technical"]]
y = data["Placed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully!")