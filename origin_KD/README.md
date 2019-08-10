## 最原始的知识蒸馏方法

用的是[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)文章中的方法

`Nets.py`：ResNet的Pytorch官方实现

`student.py`：学生网络，最简单的CNN网络

`teacher.py`：教师网络，resnet18网络

`utils.py`：网络验证

这里选用的是CIFAR10数据集，全部代码参考自：<https://github.com/PolarisShi/distillation>



