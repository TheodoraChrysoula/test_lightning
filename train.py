import os
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

np.random.seed(42)
X = np.random.rand(100, 1) * 10 # Features (100 samples, 1 feature)
y = 2.5 * X + np.random.randn(100, 1) * 2

model=LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(f"Slope: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"MSE: {mean_squared_error(y,y_pred):.2f}")

# Plot
plt.scatter(X, y, color='blue', label='Real Data')
plt.scatter(X, y_pred, color='red', linewidth=2, label='Prediction')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression")
plt.savefig("outputs/regression_plot.png", dpi=300, bbox_inches='tught')
plt.show()

with open("outputs/results.txt", "w") as f:
    f.write(f"Slope: {model.coef_[0][0]:.4f}\n")
    f.write(f"Intercept: {model.intercept_[0]:.4f}\n")
    f.write(f"MSE: {mean_squared_error(y,y_pred):.4f}\n")
    f.write(f"R-squarred: {model.score(X,y):.4f}\n")

print("Results save to outputs/directory")