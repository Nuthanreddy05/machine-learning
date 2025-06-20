from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.data[:, [0]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=37
)

# Create and train the LinearRegression model
model = LinearRegression()
model.fit(X_train, y_train, batch_size=32, regularization=0, max_epochs=100, patience=3)
loss = model.losses

# save the model
model.save("model2_regression.json")

# model.save_model_parameters('model_parameters1.json')

# Plot the loss
plt.plot(loss)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Model 2 Training Loss")
plt.savefig("model2_loss.png")
plt.show()
