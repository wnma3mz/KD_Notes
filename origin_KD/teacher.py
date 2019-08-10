# coding: utf-8
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from Nets import BasicBlock, Bottleneck, ResNet
from utils import evlation


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 3, 4]

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
 ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

trainset = CIFAR10(root='./cifar10',
                   train=True,
                   download=False,
                   transform=transform)

testset = CIFAR10(root='./cifar10',
                  train=False,
                  download=False,
                  transform=transform)

trainloader = DataLoader(trainset, batch_size=32*len(device_ids), shuffle=True, num_workers=0)

testloader = DataLoader(testset, batch_size=32*len(device_ids), shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

net = nn.DataParallel(net, device_ids=device_ids).cuda()
# net.to(device)

running_loss = 0.
batch_size = 32

for epoch in range(3):


    if epoch >= 100:
        optimizer = optim.SGD(net.parameters(),
                              lr=0.01,
                              momentum=0.9,
                              weight_decay=5e-4)
    if epoch >= 150:
        optimizer = optim.SGD(net.parameters(),
                              lr=0.001,
                              momentum=0.9,
                              weight_decay=5e-4)

    data_train = DataLoader(trainset,
                            batch_size=batch_size*len(device_ids),
                            shuffle=True,
                            num_workers=0)
    time_start = time.time()

    for i, data in enumerate(data_train):

        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()


        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_end = time.time()

    print('[%d, %5d] loss: %.4f time: %.4f' % (epoch + 1,
                                (i + 1) * batch_size, loss.item(), time_end - time_start))

    evlation(net, testloader, device, classes)
    #torch.save(net, 'teacher.pkl')
    print('Time cost:', time_end - time_start, "s")
torch.save(net.state_dict(), "teacher2.pkl")
print('Finished Training')

# torch.save(net, 'teacher.pkl')
# net = torch.load('teacher.pkl')
"""
model = torch.load('teacher.pkl')
torch.save(model.state_dict(), "teacher2.pkl")

model = ResNet(BasicBlock, [2, 2, 2, 2])
model.load_state_dict(torch.load("teacher2.pkl"))
evlation(model, testloader, device, classes)
# model.eval()

"""
# evlation(net, testloader, device, classes)
