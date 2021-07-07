# 자동 미분(Autograd)

# 1. 자동 미분(Autograd) 실습하기

import torch

w = torch.tensor(2.0, requires_grad=True)

# 수식 정의
y = w**2
z = 2*y + 5

z.backward()

print('수식을 w로 미분한 값 : {}'.format(w.grad))