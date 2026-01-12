import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("Salary_dataset.csv")


X = df[['YearsExperience']]
y = df['Salary']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

MAEL = mean_absolute_error(y_test, y_pred)
MSEL = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error")
print(f"Linear Regression = {MAEL}")

print("\nMean Squared Error")
print(f"Linear Regression = {MSEL}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.5, label='Training Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prediction')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

