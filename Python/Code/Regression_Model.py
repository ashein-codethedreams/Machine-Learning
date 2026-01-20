from sklearn.linear_model import LinearRegression
import numpy as np

# # Training data
# # House size (sq ft)
# X = np.array([[500], [800], [1000], [1200], [1500]])
# # House price
# y = np.array([50000, 80000, 100000, 120000, 150000])

# # Create model
# model = LinearRegression()

# # Train model
# model.fit(X, y)

# # Predict price for a new house
# new_house = [[1100]]
# predicted_price = model.predict(new_house)

# print("Predicted price:", predicted_price[0])


# # Years of experience
# X = np.array([[0], [1], [2], [3], [4]])
# # Salary
# y = np.array([20000, 25000, 30000, 35000, 40000])

# model = LinearRegression()
# model.fit(X, y)

# print(model.predict([[5]]))  # Predict salary for 5 years experience


from sklearn.ensemble import RandomForestRegressor
import numpy as np

X = np.array([[10], [20], [30], [40], [50]])
y = np.array([100, 200, 300, 400, 500])

model = RandomForestRegressor()
model.fit(X, y)

print(model.predict([[35]]))
