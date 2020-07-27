import torch

a = torch.randn(2, 2, requires_grad=True)
b = torch.randn(2, 2, requires_grad=True)
x = torch.ones(2, 2, requires_grad=True)
Y = a * x + b
y1, y2 = Y.chunk(2, 0)
lam = 0.3
Y = y1 * lam + y2 * (1 - lam)


# Y.backward()

# print(Y.detach())