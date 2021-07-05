# 파이토치로 선형 회귀 구현하기

# 1. 기본 셋팅

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에 같은 결과가 나오도록 랜덤 시드(random seed)를 준다.
torch.manual_seed(1)

# 2. 변수 선언

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# x_train과 x_train의 크기(shape)을 출력
print(x_train)
print(x_train.shape)

# y_train과 y_train의 크기(shape)을 출력
print(y_train)
print(y_train.shape)

# 3. 가중치와 편향의 초기화

# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
W = torch.zeros(1, requires_grad=True)
# 가중치 출력
print(W)

# 편향b도 0으로 초기화하고, 학습을 통해 값이 변경되는 변수임을 명시
b = torch.zeros(1, requires_grad=True)
print(b)

# 4. 가설 세우기

hypothesis = x_train * W + b
print(hypothesis)

# 5. 비용 함수 선언하기

# 앞서 배운 torch.mean으로 평균을 구한다
cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)

# 6. 경사 하강법 구현하기

optimizer = optim.SGD([W, b], lr=0.01)

# gradient를 0으로 초기화
optimizer.zero_grad()
# 비용 함수를 미분하여 gradient 계산
cost.backward()
# W와 b를 업데이트
optimizer.step()

# 7. 전체 코드

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:3f}, b: {:3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

# 8. optimizer.zero_grad()가 필요한 이유
import torch

w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    z = 2 * w

    z.backward()
    print('수식을 w로 미분한 값 : {}'.format(w.grad))

# 9. torch.manual_seed()를 하는 이유
import torch

torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1,3):
    print(torch.rand(1))

torch.manual_seed(5)
print('랜덤 시드가 5일 때')
for i in range(1,3):
    print(torch.rand(1))

torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1,3):
    print(torch.rand(1))