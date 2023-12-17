# 导入所需的库和模块
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

train_loss = list()
test_loss = list()

# 定义一个装饰器，用于记录函数的调用和返回
def log_function_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

# 定义一个绘制损失曲线的函数
def plot(train_loss, test_loss):
    plt.figure(figsize=(5, 5))
    plt.plot(train_loss, label='train_loss', alpha=0.5)
    plt.plot(test_loss, label='test_loss', alpha=0.5)
    plt.xlabel('训练总次数')
    plt.ylabel('损失')
    plt.title('使用RegNet处理MNIST数据集')
    plt.legend()
    plt.show()

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device('cuda') # 使用GPU设备
else:
    device = torch.device('cpu') # 使用CPU设备
print(f'使用的设备是：{device}')

# 加载MNIST数据集，并转换为Tensor格式，设置数据加载器
# 在转换为Tensor之前，增加一个将图像转换为3通道的步骤
# 使用pin_memory参数来加速数据传输
train_data_set = torchvision.datasets.MNIST(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor()
]), download=True)

test_data_set = torchvision.datasets.MNIST(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor()
]), download=True)

train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

print(f'训练集长度为{train_data_size}')
print(f'测试集长度为{test_data_size}')

train_data_loader = DataLoader(dataset=train_data_set, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)
test_data_loader = DataLoader(dataset=test_data_set, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)

# 创建模型实例，使用torchvision.models.regnet中的regnet_x_400mf模型
# 将模型转移到GPU上
mynet = torchvision.models.regnet.regnet_x_400mf().to(device)

# 设置学习率和优化器
learning_rate = 1e-2
optim = torch.optim.SGD(mynet.parameters(), learning_rate)

train_step = 0

epoch = 5

if __name__ == '__main__':
    for i in range(epoch):
        print(f'----------第{i + 1}轮训练----------')
        mynet.train()
        for images, labels in train_data_loader:
            mynet = mynet.to(device)
            images = images.to(device)
            labels = labels.to(device)

            outputs = mynet(images)

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_step += 1
            if train_step % 100 == 0:
                train_loss.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                corrects = torch.eq(preds, labels)
                accuracy = torch.true_divide(torch.sum(corrects), torch.numel(labels))
                print(f'第{train_step}次训练，loss={loss.item()},accuracy={accuracy.item()}')

        mynet.eval()
        total_accuracy = 0
        j = 0  # Initialize j before the loop

        with torch.no_grad():
            for images, labels in test_data_loader:
                images = images.to(device)
                labels = labels.to(device)

                mynet = mynet.to(device)

                outputs = mynet(images)

                loss = torch.nn.functional.cross_entropy(outputs, labels)

                accuracy = (outputs.argmax(1) == labels).sum()
                total_accuracy += accuracy.item()

                if j % 100 == 0:
                    test_loss.append(loss.item())
                j += 1

            average_accuracy = total_accuracy / test_data_size
            print(f'{i + 1}轮训练结束，准确率{average_accuracy}')
            mynet = mynet.to('cpu')
            torch.save(mynet.state_dict(), f'MNIST_{i+1}_acc_{average_accuracy}.pth')

    plot(train_loss, test_loss)
