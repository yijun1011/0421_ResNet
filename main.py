import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_root = r"G:\zyj\0421resnet\dataset\training_set"
val_root = r"G:\zyj\0421resnet\dataset\test_set"
show_dir = r"G:\zyj\0421resnet\show"
model_save = r"G:\zyj\0421resnet\show\model_save"


# 定义一个残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # self.in_channel = in_channel
        # self.out_channel = out_channel
        # self.stride = stride

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # print(out.shape)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 定义网络
class Resnet(nn.Module):
    def __init__(self,block,num_block):
        super(Resnet, self).__init__()
        self.in_channel = 64
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer1 = self._make_layer(block, 64, num_block[0])
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )

    def _make_layer(self, block, out_channel, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel

        for i in range(1, num_block):
            layers.append(block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x


# 设定超参数
EPOCH = 200
BATCH_SIZE = 16
LR = 0.001

# 处理数据
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(300),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    normalize
]
)

# 读取文件夹中的训练集
train_data = ImageFolder(train_root, transform=data_transform)

# 加载训练集数据
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# 读取文件夹中的验证集
test_data = ImageFolder(val_root, transform=data_transform)

# 加载验证集数据
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

# 创建对象
net = Resnet(ResidualBlock, [3, 4, 6, 3]).to(device)
# print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=15,
                                                       verbose=False, threshold=0.0001,
                                                       threshold_mode='rel', cooldown=0,
                                                       min_lr=0.000007, eps=1e-08)

# 开始训练
if __name__ == "__main__":
    train_loss_plt = []
    test_acc_plt = []

    for epoch in range(EPOCH):
        net.train()
        sum_train_batch_loss = 0
        sum_train_batch_acc = 0

        # 训练集
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs)
            # print(labels)
            optimizer.zero_grad()

            outputs = net(inputs)

            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            # print(outputs.data)

            if i % 20 == 19:
                print("[epoch %d, batch %d] loss: %.3f" % (epoch + 1, i + 1, train_loss.item()))

            sum_train_batch_loss += train_loss.item()  # 一个epoch内所有batch的loss之和

        av_train_epoch_loss = sum_train_batch_loss / len(train_loader)  # 每一个epoch的平均误差

        lr_update = optimizer.state_dict()['param_groups'][0]['lr']
        print("epoch %d av_train_loss: %.3f lr: %.5f" % (epoch + 1, av_train_epoch_loss, lr_update))

        train_loss_plt.append(av_train_epoch_loss)  # 列表中记录每一个epoch的平均loss
        scheduler.step(av_train_epoch_loss)

        # 验证集
        net.eval()
        with torch.no_grad():

            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                val_outputs = net(images)
                # print(val_outputs)
                # print(torch.max(val_outputs,1))
                test_loss = criterion(val_outputs, labels)

                _, test_pre = torch.max(val_outputs.data, 1)
                # print(labels)
                # print(test_pre)

                total += labels.size(0)
                correct += (test_pre == labels).sum()
                correct = correct.item()
            # print(correct)
            # print(total)
            acc = correct / total  # 记录的是每一个batch的平均误差

            print("epoch %d av_test_acc: %.3f" % (epoch + 1, acc), end='\n')

            test_acc_plt.append(acc)  # 列表中记录这每一个epoch的平均acc

        if epoch >= 1:
            xy = range(epoch + 1)

            plt.plot(xy, train_loss_plt, "oy--")
            plt .plot(xy, test_acc_plt, "om--")
            plt.xlabel("epoch")
            plt.ylabel('loss')
            plt.legend(["average train loss", "average test acc"])
            plt.savefig(show_dir + '/point.jpg')

            if test_acc_plt[-1] > max(test_acc_plt[:-1]):
                torch.save(net.state_dict(),"%s/epoch%d+loss%.3f+acc%.3f+lr%.5f.pth" % (model_save, (epoch+1), train_loss_plt[-1], test_acc_plt[-1],lr_update))


