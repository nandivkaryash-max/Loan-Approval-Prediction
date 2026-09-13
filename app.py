import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/loan_data.csv")

# Separate input features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Convert categorical columns into numbers
X = pd.get_dummies(X, drop_first=True)

# Convert target into numbers
y = y.map({"N": 0, "Y": 1})

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Preprocessing completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create the ML model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pickle

# Save the trained model
with open("model/loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the feature columns
with open("model/feature_columns.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)

print("\nModel saved successfully!")