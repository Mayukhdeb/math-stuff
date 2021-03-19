import numpy as np

a = np.array([5,1,2,1])

coeffs_of_a_squared = (np.convolve(a,a))
print(coeffs_of_a_squared)

'''
[25 10 21 14  6  4  1]
'''

values = np.array([
    [-2.433, 0], 
    [-1,5], 
    [-0.333, 4.852], 
    [0,5],
    [-3,-7],
    [3, 53],
    [9, 905]
])

print(values*values)  ## simple element wise multiplication

'''
[[5.9194890e+00 0.0000000e+00]
 [1.0000000e+00 2.5000000e+01]
 [1.1088900e-01 2.3541904e+01]
 [0.0000000e+00 2.5000000e+01]
 [9.0000000e+00 4.9000000e+01]
 [9.0000000e+00 2.8090000e+03]
 [8.1000000e+01 8.1902500e+05]]
'''

g = np.fft.fft(np.array([5,1,2,1]))

print(g)

'''
[9.+0.j 3.+0.j 5.+0.j 3.+0.j]
'''