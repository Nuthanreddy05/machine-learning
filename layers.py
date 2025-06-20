import numpy as np

class BaseLayer:
    def __init__(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, gradient):
        raise NotImplementedError("Must be implemented by subclass.")

class Linear(BaseLayer):
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        standard_dev = np.sqrt(2.0 / input_size)
        self.weights = np.random.normal(0, standard_dev, (input_size, output_size))
        self.biases = np.zeros(output_size)
    # Forward pass
    def forward(self, input_data):
        self.input_data = input_data
        forward_result = np.matmul(input_data, self.weights) + self.biases
        return forward_result
    # Backward pass
    def backward(self, gradient):
        self.gradient_weights = np.dot(self.input_data.T, gradient)
        self.gradient_biases = np.sum(gradient, axis=0)
        return np.dot(gradient, self.weights.T)

class Sigmoid(BaseLayer):
    # Forward pass
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    # Backward pass
    def backward(self, gradient):
        return gradient * (self.output * (1 - self.output))

class Tanh(BaseLayer):
    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, gradient):
        return gradient * (1 - np.square(self.output))

    
class ReLU(BaseLayer):
    def forward(self, input_data):
        self.input_data = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, gradient):
        # Gradient of ReLU is 1 for input_data > 0, otherwise it's 0.
        self.gradient_input = gradient * (self.input_data > 0)
        return self.gradient_input

class LossCrossEntropy(BaseLayer):
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        # Compute loss
        loss = -np.sum(targets * np.log(predictions + 1e-10))
        return loss

    def backward(self, gradient=1):
        # Compute gradient
        adjusted_predictions = np.clip(self.predictions, 1e-10, 1 - 1e-10)
        return gradient * (-self.targets / adjusted_predictions)
        
