import torch.nn as nn
import torch



m = nn.Linear(20, 30)
print(m)
input = torch.randn(128, 20)
print(input)
print(input.size())
output = m(input)
print(output)
print(output.size())