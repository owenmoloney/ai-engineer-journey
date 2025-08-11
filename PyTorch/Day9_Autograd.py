import torch

# Manual gradient calculation example
x = torch.tensor(3.0, requires_grad=True)  # Enable gradient tracking
y = x**2 + 2*x + 1  # y = x^2 + 2x + 1

# Compute gradient manually (dy/dx)
# dy/dx = 2x + 2
manual_grad = 2 * x.item() + 2
print(f"Manual gradient at x={x.item()}: {manual_grad}")

# Compute gradient with autograd
y.backward()  # Compute gradients automatically
print(f"Autograd gradient at x={x.item()}: {x.grad.item()}")

# Gradient of a vector function example
# z = 3x^3 - 4x^2 + 6x - 1
x.grad.zero_()  # Reset gradients before new backward pass
z = 3 * x**3 - 4 * x**2 + 6 * x - 1
z.backward()
print(f"Gradient of z at x={x.item()}: {x.grad.item()}")

# Working with tensors (vectors)
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = a * 2
c = b.sum()
c.backward()
print(f"Gradients for tensor a: {a.grad}")

