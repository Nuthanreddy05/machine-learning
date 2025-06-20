import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    

    def fit(self, X, y, learning_rate=0.01, max_epochs=1000,patience=3,batch_size=32,regularization=0):
        # Initializing weights and bias
        n_samples, n_features = X.shape
        num_class=len(set(y))

        self.weights = np.zeros((num_class,n_features))
        self.bias = np.zeros(num_class)
        val_ratio = 0.1
        split_point = int(n_samples * val_ratio)

        X_train, X_val = np.split(X, [-split_point])
        y_train, y_val = np.split(y, [-split_point])

        best_loss=np.inf

        for epoch in range(max_epochs):
            
            for i in range(0, len(X_train), batch_size):
                #Splitting training set into batches
                X_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]

                for c in range(num_class):
                    y_binary = (y_batch == c) * 1
                    prediction = self.sigmoid(np.dot(X_batch, self.weights[c]) + self.bias[c])

                    #Compute the gradients of weight and bias
                    weight_grad = self.compute_weight_gradient(X_batch,prediction,y_binary,batch_size,regularization,c)
                    bias_grad = np.mean(prediction - y_binary)

                    self.weights[c] -= learning_rate * weight_grad
                    self.bias[c] -=learning_rate * bias_grad

                #Compute the loss
            val_loss = self.compute_loss(X_val, y_val)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        logits = np.dot(X, self.weights.T) + self.bias
        exp=np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return np.argmax(exp, axis=1)
    
    def compute_weight_gradient(self,X_b,pred,y_binary,batch_size,reg,c):
        weight_gradient = np.dot(X_b.T, pred - y_binary) / batch_size + reg * self.weights[c]
        return weight_gradient

    def compute_loss(self, X, y):
        logits = np.dot(X, self.weights.T) + self.bias
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(len(y)), y]))
        return loss

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def save(self, file_path: str):
        model_params = {"weights": self.weights, "bias": self.bias}
        with open(file_path, "wb") as f:
            pickle.dump(model_params, f)

    def load(self, file_path: str):
        with open(file_path, "rb") as f:
            model_params = pickle.load(f)
        self.weights = model_params["weights"]
        self.bias = model_params["bias"]

