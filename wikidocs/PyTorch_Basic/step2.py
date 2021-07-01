# 넘파이로 텐서 만들기(벡터와 행렬 만들기)
import numpy as np

# 1. 1D with Numpy
t = np.array([0., 1., 2., 3., 4., 5., 6.])
# 파이썬으로 설정하면 list를 생성해서 np.array로 1차원 array로 변환함.
print(t)

# 1차원 벡터의 차원과 크기 출력
print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

# 2. Numpy 기초 이해하기
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # 인덱스를 통한 원소 접근

# 3. 2D with Numpy
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)

# 2차원 벡터의 차원과 크기 출력
print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

# 4. PyTorch Tensor Allocation
import torch

# 5. 1D with PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

# shape, size 사용하여 크기 확인
print(t.dim())
print(t.shape)
print(t.size())

# 인덱스를 통한 접근
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1]) # 슬라이싱
print(t[:2], t[3:]) # 슬라이싱

# 6. 2D with PyTorch
t = torch.FloatTensor([[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.],
                        [10., 11., 12.]
                        ])
print(t)

# 차원과 크기 확인
print(t.dim()) # rank
print(t.size()) # shape

print(t[:, :-1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원에서는 맨 마지막에서 첫번째를 제외하고 다 가져온다.

# 7. 브로드캐스팅(Broadcasting)

# 같은 크기일 때 연산
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalarws
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] -> [3, 3]
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1+m2)

# 8. 자주 사용되는 기능들

# 행렬 곱셈과 곰셈의 차이(Matrix Multiplication Vs. Multiplication)
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

# mul
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print("Shape of Matrix 1: ", m1.shape) # 2 x 2
print("Shape of Matrix 2: ", m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

# 평균(Mean)
t = torch.FloatTensor([1, 2])
print(t.mean())

# 2차원 행렬 선언
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.mean())

# 차원을 인자로 주는 경우
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

# 덧셈(Sum)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.sum())
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거

# 최대(Max)외 아그맥스(ArgMax)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max())

# 차원을 인자로 준다
print(t.max(dim=0))

print('Max: ', t.max(dim=0)[0])
print('ArgMax: ', t.max(dim=0)[1])

print(t.max(dim=1))
print(t.max(dim=-1))