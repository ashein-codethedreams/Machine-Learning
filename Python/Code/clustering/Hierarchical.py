from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([
    [10], [20], [30],
    [80], [90], [100]
])

model = AgglomerativeClustering(n_clusters=2)
labels = model.fit_predict(X)

print(labels)
