# coding: utf-8
import time
import torch


def evlation(net, testloader, device, classes):
    # 评估
    net.eval()
    time_start = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 预测正确数
    print(correct, total)

    print('Accuracy of the network on the 10000 test images: %f %%' %
          (100 * correct / total))

    class_correct = [0. for _ in range(10)]
    class_total = [0. for _ in range(10)]
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()  # 删除张量中大小为1的维度
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2f %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))
    time_end = time.time()
    print('Time cost:', time_end - time_start, "s")


"""
S-DNN

x = torch.tensor([[ 0.2979,  0.0655, -0.0312,  0.0616,  0.0830, 
                   -0.1206, -0.2084, -0.0345,  0.2106, -0.0558]])
y = torch.tensor([5])
print(torch.log(torch.sum(torch.exp(x))) - x[0, y])

criterion = nn.CrossEntropyLoss()
print(criterion(x, y))

"""
