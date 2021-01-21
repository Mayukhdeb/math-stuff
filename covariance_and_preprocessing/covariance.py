def variance(data):
    ## mean
    mean = sum(data) / len(data)
    ## variance
    variance = sum((xi - mean) ** 2 for xi in data) / len(data)
    return variance



data = [1,2,3]
print(variance(data))

import numpy as np

print (np.var(data))

def covariance(a, b):

    a_mean = sum(data) / len(data)
    b_mean = sum(data) / len(data)

    sum_all = 0

    for i in range(0, len(a)):
        sum_all += ((a[i] - a_mean) * (b[i] - b_mean))
    return sum_all/(len(a)-1)

a = [1,2,3]
b = [4,4.2,9]

print(covariance(a,b))

print (np.cov(a,b)[0][1])

a = [10,8, 4]
b = [6,4,2]

print(np.cov(a,b))


