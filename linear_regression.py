import numpy as np
import json


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(
        self,
        X,
        y,
        batch_size: int,
        regularization: float = 0,
        max_epochs: int = 100,
        patience: int = 3,
        learning_rate=0.01,
    ):

        # splitting into 90-10
        split_idx = int(0.9 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        # Initializing weights, bias
        self.weights = np.zeros((X.shape[1], y.shape[1]))
        self.bias = np.zeros(y.shape[1])
        best_weights = np.copy(self.weights)
        best_bias = np.copy(self.bias)
        best_val_loss = float("inf")
        consecutive_increases = 0

        for epoch in range(max_epochs):

            # Shuffling the training data
            permutation = np.random.permutation(len(X_train))
            X_train_perm, y_train_perm = X_train[permutation], y_train[permutation]

            # Mini-batch gradient descent
            for i in range(0, len(X_train), batch_size):
                X_batch, y_batch = (
                    X_train_perm[i : i + batch_size],
                    y_train_perm[i : i + batch_size],
                )
                y_pred = np.dot(X_batch, self.weights) + self.bias

                gradient_weights = (2.0 / len(X_batch)) * (X_batch.T.dot(y_pred - y_batch)) + (
                    2 * regularization * self.weights
                )
                gradient_bias = (2.0 / batch_size) * np.sum(y_pred - y_batch)

                self.weights -= learning_rate * gradient_weights
                self.bias -= learning_rate * gradient_bias

            # Eval on validation set
            y_pred = self.predict(X_val)
            val_loss = self.mse(y_pred, y_val)
            self.losses.append(val_loss)

            # Applying Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = np.copy(self.weights)
                best_bias = np.copy(self.bias)
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases == patience:
                    break

        # Update the model params
        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def score(self,X,y):
        y_pred=self.predict(X)
        return self.mse(y_pred,y)

    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def save(self, file_path: str):
        model_params = {"weights": self.weights.tolist(), "bias": self.bias.tolist()}
        with open(file_path, "w") as f:
            json.dump(model_params, f)

    def load(self, file_path: str):
        with open(file_path, "r") as f:
            model_params = json.load(f)
        self.weights = np.array(model_params["weights"])
        self.bias = np.array(model_params["bias"])
