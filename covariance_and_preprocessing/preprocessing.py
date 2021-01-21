"""
preprocessing 
"""

"""
mean normalization
"""
import numpy as np
  
import matplotlib.pyplot as plt

def mean_normalization(X):
    newX = X - np.mean(X, axis = 0)
    return newX

b = np.random.normal(2, 0.5, 20)

plt.plot(b, label = "original data")
plt.plot(mean_normalization(b), label = "after mean normalization")
plt.legend()
plt.grid()
plt.show()

"""
standardization
"""

def standardize(X):
    standarized= mean_normalization(X)/np.std(X, axis = 0)
    return standarized

a =  np.random.normal(3, 4, 300)
b = np.random.normal(4, 2, 300)

all_data = np.array([a,b]).T
standard_data = standardize(all_data)
standard_a, standard_b = standard_data[:, 0], standard_data[:, 1]

plt.scatter(a,b, label = "original")
plt.scatter(standard_a,standard_b, label = "standardized")

plt.xlim(-15,15)
plt.ylim(-15,15)
plt.legend()
plt.grid()
plt.show()

print(np.cov(standard_data, rowvar=False, bias=True))
