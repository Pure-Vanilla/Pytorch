# 모델 저장하고 불러오기 

import torch
import torch.onnx as onnx
import torchvision.models as models

# 1. 모델 가중치 저장하고 불러오기
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# 동일한 모덜의 인스턴스 생성 후 load_state_dict() 메소드를 사용하여 매개변수들을 불러온다
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 2. 모델의 형태를 포함하여 저장하고 불러오기
torch.save(model, 'model.pth')

model = torch.load('model.pth')

# 3. 모델을 ONNX로 내보내기
input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'model.onnx')