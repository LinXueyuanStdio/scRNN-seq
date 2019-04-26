import torch.nn as nn
import torch

# With square kernels and equal stride
layer1 = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),   # batch x 32 x 28 x 28
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, 3, padding=1),   # batch x 32 x 28 x 28
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 28 x 28
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 28 x 28
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(5, 5)   # batch x 64 x 14 x 14
)
# non-square kernels and unequal stride and with padding
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(50, 64, 20, 10)
output = nn.ConvTranspose2d(64, 1, 6, 5, 1, 1)(input)
print(output.size())
