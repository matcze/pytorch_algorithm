# PyTorch Algorithm Demo

This project demonstrates the basic functionality of the PyTorch deep learning library. It focuses on the implementation workflow rather than achieving the highest accuracy or lowest loss.

PyTorch is a Python library for deep learning, developed by Meta Platforms and first released in 2016.

# Assumptions
- The project uses a simple dataset from the scikit-learn library.
- Only basic analysis and minimal preprocessing are performed.
- Preprocessing is restricted to essential steps required for the neural network to function.
- The goal is to demonstrate neural network implementation in PyTorch.

# Dataset

The Diabetes dataset from scikit-learn is used. It is ideal for a PyTorch demonstration because:
- Small and fast (442 samples, 10 features)
- Easy to convert to PyTorch tensors
- Illustrates the PyTorch workflow clearly
- For linear regression, only the BMI feature is used

# Analysis and Preprocessing

Preprocessing steps include:

1. Scaling the features using standard scaling
2. Converting the data to PyTorch tensors
3. Splitting the dataset into training and validation sets (80:20 ratio)

# First Model – Linear Regression

Two implementations of linear regression are demonstrated:
1. Manual Implementation
- Weights and biases defined using nn.Parameter
- Loss function: L1Loss
- Optimizer: SGD

2. Layer-based Implementation
- Linear layer defined using nn.Linear
- Loss function: MSELoss
- Optimizer: SGD

Training details (both implementations):
Learning rate: 0.1
Epochs: 300
Training loop:
```
for epoch in range(epochs):
    model.train()                    # Set model to training mode
    predictions = model(x_train)     # Forward pass
    loss = loss_function(predictions, y_train)  # Compute loss
    optimizer.zero_grad()            # Zero gradients
    loss.backward()                  # Backpropagation
    optimizer.step()                 # Update weights
```
After training, accuracy and loss curves, as well as actual vs. predicted values, are plotted.

# Second Model – Neural Network

The neural network has the following architecture:
- Input layer: 10 nodes
- Hidden layer: 256 nodes with ReLU activation
- Output layer: 1 node (predicted label)

Training setup:
- Loss function: MSELoss
- Optimizer: Adam
- Training loop is similar to the linear model.

# Example Neural Network Code
```
import torch
import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)
```
# Conclusion

- The project demonstrates building models in PyTorch.
- Linear regression and a neural network are implemented with minimal preprocessing.
- Model performance was not the focus; the goal was to illustrate the implementation workflow.
- This project provides a clear example of PyTorch model training, forward propagation, and optimization.  