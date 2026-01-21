import numpy as np
from sklearn.impute import SimpleImputer

X = np.array([
    [20],
    [25],
    [np.nan],
    [30]
])

imputer = SimpleImputer(strategy="mean")
X_filled = imputer.fit_transform(X)

print(X_filled)
