import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

# Create training data: y = 3x + 2
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

# Instantiate the model
model = LinearModel()

# Define different loss functions to try
loss_functions = {
    "MSELoss": nn.MSELoss(),
    "L1Loss": nn.L1Loss()
}

# Use SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Choose the loss function to train with
loss_fn = loss_functions["MSELoss"]  # Change to "L1Loss" to try the other one

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test prediction
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    prediction = model(test_input)
    print(f"Prediction for input 5.0: {prediction.item():.2f}")

