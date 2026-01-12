import matplotlib.pyplot as plt
import numpy as np

# Your data
states = ['RS42', 'RS43', 'RS44', 'RS45']

# Linear Regression
linear_mae = [6286, 6055, 6640, 6152]
linear_rmse = [7059, 6440, 7296, 6520]

# KNN
knn_mae = [4788, 4961, 3870, 3702]
knn_rmse = [5503, 5640, 4166, 4285]

# Plot
plt.figure(figsize=(12, 6))
x = np.arange(len(states))
width = 0.10  # Width for each bar group

# Create 4 bars for each state
plt.bar(x - width*1.5, linear_mae, width, label='Linear MAE', color='lightblue', edgecolor='black')
plt.bar(x - width*0.5, linear_rmse, width, label='Linear RMSE', color='blue', edgecolor='black')
plt.bar(x + width*0.5, knn_mae, width, label='KNN MAE', color='lightcoral', edgecolor='black')
plt.bar(x + width*1.5, knn_rmse, width, label='KNN RMSE', color='red', edgecolor='black')

plt.ylabel('Error', fontsize=12)
plt.xlabel('Random State', fontsize=12)
plt.title('Machine Learning AI Metric Evaluation Comparison')
plt.xticks(x, states)
plt.legend()

plt.tight_layout()
plt.show()