# Task 3: Car Price Prediction

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset

df = pd.read_csv("car data.csv")

print("First 5 Rows of Dataset:")
print(df.head())

# 3. Data Cleaning

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 4. Feature Engineering

# Convert Year into Car Age
df["Car_Age"] = 2024 - df["Year"]

# Remove Year column
df.drop("Year", axis=1, inplace=True)

# Convert categorical columns into numeric
df = pd.get_dummies(df, drop_first=True)

print("\nUpdated Dataset:")
print(df.head())

# 5. Data Visualization

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# 6. Define Features and Target

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# 7. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 8. Train Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# 9. Make Predictions

y_pred = model.predict(X_test)

# 10. Model Evaluation

print("\nModel Performance:")

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))

print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))

# 11. Visualization: Actual vs Predicted

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()

# 12. Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Impact": model.coef_
})

print("\nFeature Impact on Price:")
print(importance)

# 13. Conclusion

print("\nConclusion:")
print("The machine learning regression model successfully predicts car prices.")
print("Important factors affecting price include car age, present price, fuel type, and transmission.")
print("This model demonstrates how machine learning can help estimate used car values.")
