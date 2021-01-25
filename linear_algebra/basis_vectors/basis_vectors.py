import numpy as np

B = np.array([
                [2, -1],   ## i components 
                [1,  1]    ## j components
            ])

B_inverse = np.linalg.inv(B)

x = np.array([
                [3],
                [2]
            ])

print(np.dot(B_inverse, x))