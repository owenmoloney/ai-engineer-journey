import torch

# 1. Tensor Creation
tensor_1d = torch.tensor([1, 2, 3, 4])
print("1D tensor:")
print(tensor_1d)

tensor_2d = torch.tensor([[1, 2], [3, 4]])
print("\n2D tensor:")
print(tensor_2d)

zeros = torch.zeros((3, 3))
print("\n3x3 zeros tensor:")
print(zeros)

ones = torch.ones((2, 4))
print("\n2x4 ones tensor:")
print(ones)

rand_tensor = torch.rand((2, 3))
print("\nRandom tensor 2x3:")
print(rand_tensor)

# 2. Tensor Manipulation
reshaped = tensor_1d.reshape(2, 2)
print("\nReshaped tensor_1d to 2x2:")
print(reshaped)

print("\nFirst row of tensor_2d:")
print(tensor_2d[0])

print("\nSecond column of tensor_2d:")
print(tensor_2d[:, 1])

tensor_2d[0, 1] = 10
print("\nChanged tensor_2d:")
print(tensor_2d)

# 3. Broadcasting
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([10, 20, 30])
print("\nTensor a:")
print(a)
print("\nTensor b:")
print(b)

c = a + b
print("\nResult of a + b (broadcasting):")
print(c)

d = a * b
print("\nResult of a * b (broadcasting):")
print(d)

# 4. Other useful operations
concat_dim0 = torch.cat([a, a], dim=0)
print("\nConcatenated along rows (dim=0):")
print(concat_dim0)

concat_dim1 = torch.cat([a, a], dim=1)
print("\nConcatenated along columns (dim=1):")
print(concat_dim1)

transposed = a.T
print("\nTransposed tensor a:")
print(transposed)

