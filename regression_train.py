#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from layers import *
from models import *

def model_regression_training(model, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100, batch_size=None):
    # Set batch size if not specified
    if batch_size is None:
        batch_size = len(X_train)

    # Initialize variables for storing best validation loss and storing losses
    best_validation_loss = np.inf
    train_losses = []
    validation_losses = []
    loss_layer = MeanSquaredError()  

    no_improvement_count = 0
    # Iterate over epochs
    for epoch in range(epochs):
        indices = np.arange(len(X_train))

        # Shuffling the indices
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Iterate over batches
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            # Forward pass
            y_pred = model.forward(X_batch)
            # Compute current loss and gradient
            curr_loss = loss_layer.forward(y_pred, y_batch)  
            curr_grad = loss_layer.backward() 
            # Backward pass
            model.backward(curr_grad)

            # Update weights and biases
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.weights -= learning_rate * layer.gradient_weights
                    layer.biases -= learning_rate * layer.gradient_biases
        
        # Compute validation loss
        validation_predictions = model.forward(X_val)
        validation_loss = loss_layer.forward(validation_predictions, y_val)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= 3:
                print(f"Stopping early at epoch {epoch}!")
                break
        # Append losses to lists for visualization
        train_losses.append(curr_loss)
        validation_losses.append(validation_loss)
        print(f'Epoch {epoch}/{epochs}, Training Loss: {curr_loss}, Validation Loss: {validation_loss}')
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(validation_losses, label='Validation Loss', color='red')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()
    return [learning_rate, batch_size, train_losses[-1], validation_losses[-1]]
