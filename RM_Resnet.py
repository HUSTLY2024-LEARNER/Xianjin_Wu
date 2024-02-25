import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 进度条

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 10
batch_size = 64
batch_size_test = 1000
learning_rate = 0.01
log_interval = 10
random_seed = 114514
torch.manual_seed(random_seed)

# 添加数据增强为训练数据
train_transform = transforms.Compose([
    # 随机旋转和平移
    #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 测试数据转换例程(不包括数据增强)，因为我们通常不会在测试集上做增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=train_transform),
    batch_size=batch_size, shuffle=True)

test_loader = DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=test_transform),
    batch_size=64, shuffle=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, mid_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != mid_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, mid_channels, stride, downsample))
        self.in_channels = mid_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, mid_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)


# 创建模型
model = ResNet50(Bottleneck, [3, 4, 6, 3]).to(device)
loss_f = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)


# 训练函数
def train(epoch, model, device, train_loader, optimizer, log_interval):
    model.train()
    # 追踪平均损失
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 在每个log_interval打印
        if batch_idx % log_interval == 0:
            current_loss = running_loss / log_interval if batch_idx > 0 else running_loss
            print(
                f'Train Epoch: {epoch}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Batch Loss: {loss.item():.6f}\t Average Loss: {current_loss:.6f}')
            running_loss = 0.0  # 重置追踪的损失


# 测试函数
def test(model, device, test_loader, loss_f):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_f(output, target).item() * data.size(0)  # 累计整个测试集的损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= total  # 计算平均损失
    accuracy = 100. * correct / total  # 计算准确率
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')


# 如果是执行主文件，则执行训练和测试
if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        train(epoch, model, device, train_loader, optimizer, log_interval)
        test(model, device, test_loader, loss_f)
        # SGD优化器调度
        scheduler.step(metrics=0.0, epoch=epoch)

    example_input = torch.rand(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, "traced_model.pt")
