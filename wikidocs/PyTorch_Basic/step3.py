# 텐서 조작하기

import torch
import numpy as np

# 1. View
t = np.array([[[0, 1, 2],
                [3, 4, 5]], 
               [[6, 7, 8], 
                [9, 10, 11]]])
ft = torch.FloatTensor(t)

print(ft.shape)

# 2. 3차원 텐서에서 2차원 텐서로 변경
print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

# 3. 3차원 텐서의 크기 변경
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

# 4. 스퀴즈(Squeeze)-1인 차원을 제거한다
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

# (3 x 1) -> squeeze -> (3,)
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

# 5. 언스퀴즈(Unsqueeze)-특정 위치에 1인 차원을 추가한다
ft = torch.Tensor([0, 1, 2])
print(ft.shape)

print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

print(ft.view(1, -1))
print(ft.view(1, -1).shape)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

# 6. 타입 캐스팅(Type Casting)
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)

print(bt.long())
print(bt.float())

# 7. 연결하기(concatenate)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat[x, y], dim=0)

# 차원을 인자로 준다
print(torch.cat[x, y], dim=1)

# 8. 스택킹(Stacking)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))

print(torch.cat[x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0)

print(torch.stack([x, y, z], dim=1))

# 9. ones_like & zeros_like
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채우기

# 0으로 채워진 텐서 생성
print(torch.zeros_like(x))

# 10. In-place Operation(덮어쓰기 연산)
x = torch.FloatTensor([[1, 2], [3, 4]])

# 곱하기 연산
print(x.mul(2.))
print(x)

# 기존의 값을 덮어쓰는 곱하기 연산
print(x.mul_(2.))
print(x)