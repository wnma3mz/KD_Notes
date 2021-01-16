# -*- coding: utf-8 -*-
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from Nets import BasicBlock, Bottleneck, ResNet
from utils import evlation
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class _ConvLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate):
        super(_ConvLayer, self).__init__()

        self.add_module(
            'conv',
            nn.Conv2d(num_input_features,
                      num_output_features,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('norm', nn.BatchNorm2d(num_output_features)),

        self.drop_rate = drop_rate

    def forward(self, x):
        x = super(_ConvLayer, self).forward(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('convlayer1', _ConvLayer(3, 32, 0.0))
        self.features.add_module('maxpool', nn.MaxPool2d(2, 2))
        self.features.add_module('convlayer3', _ConvLayer(32, 64, 0.0))
        self.features.add_module('avgpool', nn.AvgPool2d(2, 2))
        self.features.add_module('convlayer5', _ConvLayer(64, 128, 0.0))

        self.classifier = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, kernel_size=8,
                           stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 3, 4]

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

trainset = CIFAR10(root='./cifar10',
                   train=True,
                   download=False,
                   transform=transform_train)

testset = CIFAR10(root='./cifar10',
                  train=False,
                  download=False,
                  transform=transform_test)

trainloader = DataLoader(trainset,
                         batch_size=100 * len(device_ids),
                         shuffle=False,
                         num_workers=2)

testloader = DataLoader(testset,
                        batch_size=100 * len(device_ids),
                        shuffle=False,
                        num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

net = CNN()
net = nn.DataParallel(net, device_ids=device_ids).cuda()

criterion = nn.CrossEntropyLoss()
# mean会warning，batchmean的性能可能不如mean
# criterion2 = nn.KLDivLoss(reduction='batchmean')
criterion2 = nn.KLDivLoss(reduction='mean')  # 相对熵
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 载入教师模型
netT = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
netT = nn.DataParallel(netT, device_ids=device_ids)
netT.load_state_dict(torch.load('teacher2.pkl'))
netT = netT.cuda()

running_loss = 0.
batch_size = 128

# 越高分布越平缓。T为1时，分类效果近似为T-DNN。为了使S-DNN能够自己学到东西，不完全依赖T-DNN。
T = 2

alpha = 0.95
for epoch in range(3):
    time_start = time.time()

    data_train = DataLoader(trainset,
                            batch_size=batch_size * len(device_ids),
                            shuffle=True,
                            num_workers=0)

    for i, data in enumerate(data_train):

        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # 教师输出作为学生初始化输入
        soft_target = netT(inputs)
        # The first objective function is the cross entropy with the soft targets and this cross entropy is computed using the same high temperature in the softmax of the distilled model as was used for generating the soft targets from the cumbersome model.
        # 使用T-DNN生成的soft target与设定的temperature计算softmax
        optimizer.zero_grad()
        # 学生输出
        outputs = net(inputs)
        loss1 = criterion(outputs, labels)

        # The second objective function is the cross entropy with the correct labels. This is computed using exactly the same logits in softmax of the distilled model but at a temperature of 1.
        # 使用S-DNN的与T计算log softmax。温度是否固定为1？
        outputs_S = F.log_softmax(outputs / T, dim=1)
        outputs_T = F.softmax(soft_target / T, dim=1)
        # We found that the best results were generally obtained by using a condiderably lower weight on the second objective function. Since the magnitudes of the gradients produced by the soft targets scale as 1/T 2 it is important to multiply them by T 2 when using both hard and soft targets. This ensures that the relative contributions of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed while experimenting with meta-parameters.
        # 在第二目标函数上使用一定的低weight可获取最好结果。soft target的scale为1/(T^2)，所以最后需要*(T^2)。这样保证了在调整T时，soft target和hard target相对贡献不变
        loss2 = criterion2(outputs_S, outputs_T) * T * T
        loss = loss1 * (1 - alpha) + loss2 * alpha
        # loss = -(F.softmax(out_t / temperature, 1).detach() * \
        #         (F.log_softmax(out_s / temperature, 1) - \
        #          F.log_softmax(out_t / temperature, 1).detach())).sum()
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss: %.4f loss1: %.4f loss2: %.4f' %
          (epoch + 1,
           (i + 1) * batch_size, loss.item(), loss1.item(), loss2.item()))

    evlation(net, testloader, device, classes)

    # torch.save(net, 'student.pkl')
    time_end = time.time()
    print('Time cost:', time_end - time_start, "s")

print('Finished Training')

# torch.save(net, 'student.pkl')
# net = torch.load('student.pkl')
