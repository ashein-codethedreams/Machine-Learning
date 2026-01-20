from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([
    [1], [2], [2.5],
    [10], [10.5], [50]
])

model = DBSCAN(eps=2, min_samples=2)
labels = model.fit_predict(X)

print(labels)
