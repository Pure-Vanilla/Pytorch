# 텐서(Tensor)

# 필요한 라이브러리 임포트

import torch
import numpy as np
from torch.functional import Tensor

# 1. 텐서(Tensor) 초기화

# 데이터로부터 직접(directly) 생성하기
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# NumPy 배열로부터 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 다른 텐서로부터 생성하기
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지한다
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # X_data의 속성을 덮어쓴다
print(f"Random Tensor: \n {x_rand} \n")

# 무작위(random) 또는 상수(constant) 값을 사용하기
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# 2. 텐서의 속성(Attribute)
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 3. 텐서 연산(Operation)

# GPU가 존재하면 텐서를 이동한다
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Numpy식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# 4. 텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 산술 연산

# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산한다. y1, y2, y3는 모두 같은 값을 가졌다
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
y4 = torch.matmul(tensor, tensor.T, out=y3)
print(y4)

# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖는다
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
z4 = torch.mul(tensor, tensor, out=z3)
print(z4)

# 단일 요소
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# 바꿔치기
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# 5. Numpy 변환(Bridge)
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# 텐서의 변경 사항이 NumPy 배열에 반영된다
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy 배열을 텐서로 변환하기
n = np.ones(5)
t = torch.from_numpy(n)

# NumPy 배열의 변경 사항이 텐서에 반영된다
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")