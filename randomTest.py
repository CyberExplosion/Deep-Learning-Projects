import torch.nn as nn
import torch

m = nn.Softmax(dim=1)
input = torch.randn(2, 3)
output = m(input)
print(output)
maxOut = output.max(dim=1)[1]
print(maxOut)

loss =nn.CrossEntropyLoss()
if loss == 'CrossEntropyLoss()':
    print('lmao')
print(type(loss))
print(loss)

from collections import OrderedDict

one = OrderedDict(
    [
    ('1', 1),
    ('2', 2)
    ]
)
two = OrderedDict(
    [
    ('3', 3),
    ('4', 4)
    ]
)

res = one.update(two)
print(one)