import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 定义经典的卷积神经网络
class ClassicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassicCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平特征图
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')
    
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    print(f'训练集: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')
    
    return avg_loss, accuracy

# 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

# 主函数
def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # 创建模型
    model = ClassicCNN(num_classes=10).to(device)
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练模型
    epochs = 30 
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        print(f"\n第 {epoch} 轮训练")
        train_loss, train_acc = train(model, device, trainloader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, testloader, criterion)
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
    }, 'classic_cnn_model.pth')
    
    print("模型已保存!")
    
    # 绘制训练曲线（需要matplotlib）
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失')
        plt.plot(test_losses, label='测试损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.title('损失曲线')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='训练准确率')
        plt.plot(test_accuracies, label='测试准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.title('准确率曲线')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("训练曲线已保存!")
    except ImportError:
        print("未安装matplotlib，跳过绘图")

# 用于推理的函数
def predict(model, device, image_path, transform, class_names):
    """对单张图片进行预测"""
    from PIL import Image
    
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

# 加载已训练的模型
def load_model(model_path, device):
    model = ClassicCNN(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

if __name__ == "__main__":
    import os
    import time
    current_pid = os.getpid()
    print(f"当前进程 PID: {current_pid}")
    time.sleep(5)
    main()
