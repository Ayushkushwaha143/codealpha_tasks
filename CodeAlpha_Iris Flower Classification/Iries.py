# Task 1: Iris Flower Classification

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset

df = pd.read_csv("Iris.csv")

print("First 5 Rows of Dataset:")
print(df.head())

# Remove Id column if it exists
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# 3. Data Exploration

print("\nDataset Information:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

print("\nSpecies Count:")
print(df["Species"].value_counts())

# 4. Correlation Heatmap

plt.figure(figsize=(6,4))
sns.heatmap(df.drop("Species", axis=1).corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# 5. Prepare Features and Target

X = df.drop("Species", axis=1)
y = df["Species"]

# 6. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 7. Train Machine Learning Model

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 8. Make Predictions

y_pred = model.predict(X_test)

# 9. Model Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Confusion Matrix Visualization

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 11. Conclusion

print("\nConclusion:")
print("The Random Forest model successfully classified Iris flowers")
print("based on sepal and petal measurements with high accuracy.")
