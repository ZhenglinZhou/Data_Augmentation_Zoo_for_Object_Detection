import torch
from torch import nn

class linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))
    def forward(self, x):
        x = x.matmul(self.w)
        y = x + self.b.expand_as(x)
        return y

class perception(nn.Module):
    def __init__(self, in_dim, hig_dim, out_dim):
        super(perception, self).__init__()
        self.layer = nn.Sequential(
            linear(in_dim, hig_dim),
            nn.Sigmoid(),
            linear(hig_dim, out_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.layer(x)
        return y

if __name__ == '__main__':
    model = perception(100, 200, 4)
    input = torch.randn(100)
    output = model(input)
    output = torch.randn(1, 5, requires_grad=True)
    label = torch.Tensor([1]).long()



    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)
    print(loss)
