from sklearn.preprocessing import LabelEncoder

data = ["Yes", "Yes", "Yes", "No"]

encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data)

print(encoded_data)
