import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import math

INPUT_NODE = 70
OUTPUT_NODE = 2
IMAGE_SIZE1 = 1
IMAGE_SIZE2 = 70
NUM_CHANNELS = 1
NUM_LABELS = 2
CONV1_DEEP = 32
CONV1_SIZE1 = 1
CONV1_SIZE2 = 5
CONV2_DEEP = 64
CONV2_SIZE1 = 1
CONV2_SIZE2 = 5
FC_SIZE = 512

DATA_SIZE = 400
BATCH_SIZE = 40
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.9


class myLeNet5(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        self.name="myLeNet5"
        #定义卷积层，1个输入通道，5个输出通道，1*5的卷积filter，外层补上了两圈0,因为输入的是1*70
        #nn.Conv2d (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))。

        self.conv1 = nn.Conv2d(CONV1_SIZE1, CONV1_DEEP,(CONV1_SIZE1,CONV1_SIZE2), padding=(0,2))

        #第二个卷积层，5个输入，8个输出，5*5的卷积filter
        self.conv2 = nn.Conv2d(CONV1_DEEP, CONV2_DEEP, (CONV2_SIZE1,CONV2_SIZE2), padding=(0,2))

        #最后是三个全连接层
        self.fc1 = nn.Linear(1088,128) #64*1*17, 128)  # 这里把第三个卷积当作是全连接层了
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  #output is 2

    def forward(self, x):
        '''前向传播函数'''
        #先卷积，然后调用relu激活函数，再最大值池化操作,relu之后作为输入input
        #torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        incov=self.conv1(x)
        input=F.relu(incov)
        x = F.max_pool2d(input, (1, 2),stride=(1,2),padding=0)#,stride=(0, 2), padding=(0,1))
        #第二次卷积+池化操作
        incov = self.conv2(x)
        input = F.relu(incov)
        x = F.max_pool2d(F.relu(self.conv2(x)), (1, 2),stride=(1, 2), padding=(0,0))
        #重新塑形,将多维数据重新塑造为二维数据，40*400
        x = x.view(-1, self.num_flat_features(x))
        #print('size', x.size())
        #第一个全连接
        x = F.relu(F.dropout(self.fc1(x),0.5,training=self.training))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        #x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size,现在40
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
