import torch 

import torch.nn.functional as F


w = torch.tensor(2., requires_grad = True)

p = F.sigmoid(w)

with torch.no_grad():
    p_old = F.sigmoid(w)


print('pold and p is : ', p_old, p)
ratio = p/p_old
ratio.backward()

print(w.grad)


