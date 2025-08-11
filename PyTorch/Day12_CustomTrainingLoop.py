import torch

# Sample data: y = 3x + 2
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

# Model parameters (weights and bias), requires gradients
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Hyperparameters
learning_rate = 0.01
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass: compute predictions
    y_pred = x_train * w + b
    
    # Compute mean squared error loss
    loss = ((y_pred - y_train) ** 2).mean()
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update weights manually
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Zero gradients before next iteration
    w.grad.zero_()
    b.grad.zero_()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

print(f"\nTrained weight: {w.item():.4f}, bias: {b.item():.4f}")

# Test the model
x_test = torch.tensor([[5.0]])
y_test_pred = x_test * w + b
print(f"Prediction for input 5.0: {y_test_pred.item():.4f}")

