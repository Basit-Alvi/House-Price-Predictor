from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Dataset 1: House size vs price.
X = [[500], [1000], [1500], [2000], [2500]]
y = [100000, 200000, 300000, 400000, 500000]

# Dataset 2: Experience vs salary.
A = [[2], [4], [5], [6], [8]]
B = [50000, 100000, 120000, 150000, 200000]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42)

# Create models
model1 = LinearRegression()
model2 = LinearRegression()

# Train models
model1.fit(X_train, y_train)
model2.fit(A_train, B_train)

# Predictions
pred1 = model1.predict(X_test)
pred2 = model2.predict(A_test)

# Evaluation.
print("House Price Model:")
print("MAE:", mean_absolute_error(y_test, pred1))
print("R2 Score:", r2_score(y_test, pred1))

print("\nSalary Model:")
print("MAE:", mean_absolute_error(B_test, pred2))
print("R2 Score:", r2_score(B_test, pred2))

# Custom predictions
print("\nPredicted price for 1800 sqft:", model1.predict([[1800]]))
print("Predicted salary for 7 years:", model2.predict([[7]]))
