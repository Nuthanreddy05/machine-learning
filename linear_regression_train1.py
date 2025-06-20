from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, [2, 1]]
y = iris.data[:, [3]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=37
)

model = LinearRegression()
model.fit(X_train, y_train, batch_size=32, regularization=0, max_epochs=100, patience=3)
loss = model.losses

# save the model
model.save("model1_regression.json")

# Plot the loss
plt.plot(loss)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Model 1 Training Loss")
plt.savefig("model1_loss.png")
plt.show()


# With regularization
model_reg = LinearRegression()
model_reg.fit(
    X_train, y_train, batch_size=32, regularization=0.001, max_epochs=100, patience=3
)
loss_reg = model.losses

model_reg.save("model1_regression_reg.json")


# Plot the loss
plt.plot(loss)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Model 1 Training Loss (Regularization)")
plt.savefig("model1_loss_reg.png")
plt.show()
