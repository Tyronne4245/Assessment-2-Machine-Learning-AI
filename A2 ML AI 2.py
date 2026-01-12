import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv("Salary_dataset.csv")

X = df[['YearsExperience']]
y = df['Salary']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

MAEK = mean_absolute_error(y_test, y_pred)
MSEK = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error")
print(f"KNN = {MAEK}")

print("\nMean Squared Error")
print(f"KNN = {MSEK}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.5, label='All Data')
plt.scatter(X_test, y_pred, color='red', marker='x', s=100, label='Predictions')
plt.xlabel('Years Experience'); plt.ylabel('Salary'); plt.legend(); plt.grid(True)
plt.show()