import numpy as np
import pickle
class NeuralNetwork:
    def __init__(self):
        pass

    def propagate_forward(self, input_data):
        raise NotImplementedError("Forward propagation not implemented.")

    def propagate_backward(self, gradient):
        raise NotImplementedError("Backward propagation not implemented.")
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

class Sequential(NeuralNetwork):
    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers
    # Forward propagation
    def forward(self, input_data):
        for module in self.layers:
            input_data = module.forward(input_data)
        return input_data
    # Backward propagation
    def backward(self, gradient):
        for module in reversed(self.layers):
            gradient = module.backward(gradient)
        return gradient

class MeanSquaredError:
    def __init__(self):
        self.residuals = None
    # Forward pass
    def forward(self, predictions, targets):
        self.residuals = predictions - targets
        return np.mean(np.square(self.residuals))
    # Backward pass
    def backward(self):
        return 2 * self.residuals / self.residuals.size

class BinaryCrossEntropy(NeuralNetwork):
    def forward(self, predictions, targets):
        # Clip predictions
        self.predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        self.targets = targets
        # Compute binary cross-entropy loss
        loss = -np.sum(targets * np.log(self.predictions) + (1 - targets) * np.log(1 - self.predictions))
        return loss

    def backward(self, gradient=1):
        # Compute gradient 
        grad_output = gradient * ((1 - self.targets) / (1 - self.predictions) - self.targets / self.predictions)
        return grad_output
