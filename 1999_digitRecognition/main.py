import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision

# 加载模型
model = torchvision.models.regnet.regnet_x_400mf()
model.load_state_dict(torch.load('/home/oxav10086/PycharmProjects/NLP/Mnsit handwritten digit recognition/MNIST_5_acc_0.9717.pth'))
model.eval()  # 设置为评估模式

# 图像预处理
def preprocess_image(image_path):
    image = Image.open(image_path)
    transformation = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    return transformation(image).unsqueeze(0)  # 添加批次维度

# 预测函数
def predict(model, image_path):
    image_tensor = preprocess_image(image_path)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# 使用模型进行预测
image_path = './img/test_2.png'  # 替换为您的图像路径
prediction = predict(model, image_path)
print(f'预测的数字是：{prediction}')