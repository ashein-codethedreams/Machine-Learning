from sklearn.linear_model import LogisticRegression
import numpy as np

# # Training data
# # Number of spam words in email
# X = np.array([[1], [2], [3], [5], [8]])
# # Labels: 0 = Not Spam, 1 = Spam
# y = np.array([0, 0, 0, 1, 1])

# # Create model
# model = LogisticRegression()

# # Train model
# model.fit(X, y)

# # Predict new email
# new_email = [[4]]
# prediction = model.predict(new_email)

# print("Spam (1) or Not Spam (0):", prediction[0])

# Study hours
X = np.array([[1], [2], [3], [4], [5]])
# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Predict
print(model.predict([[2]]))  # Fail
print(model.predict([[4]]))  # Pass