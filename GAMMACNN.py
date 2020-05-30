#this is for the paper End-to-end environmental sound classification using a 1D convolutional neural network
#this 1d-cnn name GAMMA
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time


INPUT_NODE = 70
OUTPUT_NODE = 2
IMAGE_SIZE1 = 1
IMAGE_SIZE2 = 70
NUM_CHANNELS = 1
NUM_LABELS = 2
CONV1_DEEP = 16
CONV1_SIZE1 = 1
CONV1_SIZE2 = 16
CONV2_DEEP = 16
CONV2_SIZE1 = 1
CONV2_SIZE2 = 8
CONV3_DEEP = 32
CONV3_SIZE1 = 1
CONV3_SIZE2 = 4
CONV4_DEEP = 64
CONV4_SIZE1 = 1
CONV4_SIZE2 = 2
FC_SIZE = 128

DATA_SIZE = 400
BATCH_SIZE = 40
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.9


#定义网络结构
class GAMMA(nn.Module):
    def __init__(self):
        super().__init__()

        # 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
        self.conv1 = nn.Conv2d(CONV1_SIZE1, CONV1_DEEP, (CONV1_SIZE1,CONV1_SIZE2), padding=(0,2)) #AlexCONV1(3,96, k=11,s=4,p=0)
        self.pool1 = nn.MaxPool2d((1, 2),stride=(1,2),padding=0)#AlexPool1(k=3, s=2)
        self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(CONV1_DEEP, CONV2_DEEP, (CONV2_SIZE1,CONV2_SIZE2), stride=(1,2), padding=(0,2))#AlexCONV2(96, 256,k=5,s=1,p=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))#AlexPool2(k=3,s=2)
        self.relu2 = nn.ReLU()


        self.conv3 = nn.Conv2d(CONV2_DEEP, CONV3_DEEP, (CONV3_SIZE1, CONV3_SIZE2),stride=(1,2), padding=(0,2))#AlexCONV3(256,384,k=3,s=1,p=1)
        # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(CONV3_DEEP, CONV4_DEEP, (CONV4_SIZE1, CONV4_SIZE2),stride=(1,2), padding=(0,2))#AlexCONV4(384, 384, k=3,s=1,p=1)
        self.conv5 = nn.Conv2d(CONV4_DEEP, CONV4_DEEP, (CONV4_SIZE1, CONV4_SIZE2),stride=(1,2), padding=(0,2))#AlexCONV5(384, 256, k=3, s=1,p=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))#AlexPool3(k=3,s=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(128, 64)  #AlexFC6(256*6*6, 4096)
        self.fc7 = nn.Linear(64, 32) #AlexFC6(4096,4096)
        self.fc8 = nn.Linear(32, 2)  #AlexFC6(4096,1000)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)    #input 40,16,1,43
        x = self.relu1(x)     #40 16 1 21
        x = self.conv2(x)
        x = self.pool2(x)     #40 16 1 5
        x = self.relu2(x)     #40 16 1 2
        x = self.conv3(x)
        x = self.conv4(x)     #40 32 1 4
        x = self.conv5(x)     #40 64 1 4
        x = self.pool3(x)     #40 64 1 4
        x = self.relu3(x)     #40 64 1 2
        x = x.view(-1, 128)#Alex: x = x.view(-1, 256*6*6)   40*
        x = self.fc6(x)
        x = F.relu(x)
        x=F.dropout(x,p=0.5,training=self.training)
        x = self.fc7(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc8(x)
        return x
