import matplotlib.pyplot as plt
import numpy as np

states = ['RS42', 'RS43', 'RS44', 'RS45']
knnMAE = [4788, 4961, 3870, 3702]
knnRMSE = [5503, 5640, 4166, 4285] 

plt.figure(figsize=(8, 5))
x = np.arange(len(states))  
width = 0.10  

plt.bar(x - width/2, knnMAE, width, label='MAE', color='darkgreen')
plt.bar(x + width/2, knnRMSE, width, label='RMSE', color='red')

plt.ylabel('Error')
plt.xlabel('Random State')
plt.title('Machine Learning AI 2 Metric Evaluation')
plt.xticks(x, states)
plt.legend()

plt.show()