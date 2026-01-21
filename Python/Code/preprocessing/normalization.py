from sklearn.preprocessing import MinMaxScaler
import numpy as np
X = np.array([[100], [200], [300]])

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

print(X_norm)
