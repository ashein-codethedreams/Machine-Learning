from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[100], [200], [300]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
