import heapq
import numpy as np

valuess = [1, 2, 55, 444, 4, 10, 20, 550, 4404, 40]

val = np.asarray(valuess, dtype=np.float64)

print(val)

feature_indx = []
nb_features_to_select = 5
for i in range(nb_features_to_select):
    f_indx = np.argmax(val)
    feature_indx.append(f_indx)
    val[f_indx] = float('-inf')

print(feature_indx)