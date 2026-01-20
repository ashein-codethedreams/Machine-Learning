from sklearn.cluster import KMeans
import numpy as np

# Age, Income
X = np.array([
    [20, 15],
    [22, 18],
    [25, 20],
    [45, 60],
    [48, 65],
    [50, 70]
])

# Create model (2 clusters)
model = KMeans(n_clusters=2, random_state=0)
model.fit(X)

# Cluster labels
print(model.labels_)
