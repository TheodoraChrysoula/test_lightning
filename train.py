import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
plt.savefig("regression_plot.png")
plt.show()

with open("results.txt", "w") as f:
    f.write(f"Slope: {model.coef_[0][0]}\nIntercept: {model.intercept_[0]}\nMSE: {mean_squared_error(y, y_pred)}")