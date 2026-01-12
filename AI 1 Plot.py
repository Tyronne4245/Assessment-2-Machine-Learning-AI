import matplotlib.pyplot as plt
import numpy as np

# Your data
states = ['RS42', 'RS43', 'RS44', 'RS45']
mae = [6286, 6055, 6640, 6152]
rmse = [7059, 6440, 7296, 6520]  

plt.figure(figsize=(8, 5))
x = np.arange(len(states))  
width = 0.10  

plt.bar(x - width/2, mae, width, label='MAE', color='orange')
plt.bar(x + width/2, rmse, width, label='RMSE', color='brown')

plt.ylabel('Error')
plt.xlabel('Random State')
plt.title('Machine Learning AI Metric Evaluation')
plt.xticks(x, states)
plt.legend()

plt.show()