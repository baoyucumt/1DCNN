import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import math
from tensorboardX import SummaryWriter

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

#定义一些超参数
use_gpu = torch.cuda.is_available()  #是否使用显卡
kwargs = {'num_workers': 2, 'pin_memory': True}                              #DataLoader的参数
writer = SummaryWriter('runs/exp4')
global acct,losst
losst=0.0
acct=0.0

class myLeNet5(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        #定义卷积层，1个输入通道，5个输出通道，1*5的卷积filter，外层补上了两圈0,因为输入的是1*70
        #nn.Conv2d (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))。

        self.conv1 = nn.Conv2d(CONV1_SIZE1, CONV1_DEEP,(CONV1_SIZE1,CONV1_SIZE2), padding=(0,2))

        #第二个卷积层，5个输入，8个输出，5*5的卷积filter
        self.conv2 = nn.Conv2d(CONV1_DEEP, CONV2_DEEP, (CONV2_SIZE1,CONV2_SIZE2), padding=(0,2))

        #最后是三个全连接层
        self.fc1 = nn.Linear(64*1*17, 128)  # 这里把第三个卷积当作是全连接层了
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

def truncated_normal_(self, tensor, mean=0, std=0.09):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

#参数值初始化
def weight_init(m):  #m is model
    classname = m.__class__.__name__
    if classname.find('fc1') != -1:
        m.weight.data.normal_()
        truncated_normal_(m.weight,0,0.7)
    elif isinstance(m, nn.Conv2d): # 使用isinstance来判断m属于什么类型
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0,  math.sqrt(2. / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d): # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weigth.data.fill_(1)
        m.bias.data.zero_()

#训练函数
def train(epoch):
    #调用前向传播
    model.train()
    model.training=True
    running_corrects=0
    global acct
    global losst
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target.long())                    #定义为Variable类型，能够调用autograd
        #初始化时，要清空梯度
        optimizer.zero_grad()
        output = model(data)
        _, preds = torch.max(output, 1)
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()                                                     #相当于更新权重值
        running_corrects += torch.sum(preds == target.squeeze())
        acct+= torch.sum(preds == target.squeeze())
        losst=losst+loss.item()
        #print(batch_idx, loss.item(),losst)
        writer.add_scalar('train/lr',optimizer.state_dict()['param_groups'][0]['lr'],epoch)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t ACC:{:.3f}'.format(
               epoch, batch_idx , len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(),running_corrects.double()/BATCH_SIZE))
            writer.add_scalar('train/acc', (acct.double() / (BATCH_SIZE*(epoch*10-9))), epoch)
            writer.add_scalar('train/loss', loss.item(), epoch)
            writer.add_scalar('train/loss2', losst/ (BATCH_SIZE*(epoch*10-9)), epoch)

#定义测试函数
'''
def test():
    model.eval()                                                             #让模型变为测试模式，主要是保证dropout和BN和训练过程一致。BN是指batch normalization
    model.training=False
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #计算总的损失
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]                           #获得得分最高的类别
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))
'''

if __name__ == '__main__':
    '''装载训练集合
    string0 = np.loadtxt('train_data.txt', dtype=np.float32)
    train_y = string0[:, 0].reshape(-1, 1).T  # 1行，labels
    train_x = string0[:, 1:].reshape(280, -1).T  # 937个样本向量化
    
    over'''
    string0 = np.loadtxt('alldata.txt', dtype=np.float32)
    train_y = string0[:, 0].reshape(-1, 1)  # 1行
    train_x = string0[:, 1:].reshape(400, 1,1,70)
    string0 = np.loadtxt('test_data.txt', dtype=np.float32)
    test_y = string0[:, 0].reshape(-1, 1).T  # 1行
    test_x = string0[:, 1:].reshape(120, -1).T  # 937个样本向量化
    #a=[]
    #for i in range(len(train_x)):
    #    a.append(train_x[i])
    #b=torch.from_numpy(train_y.astype(np.int))
    train_x=torch.Tensor(train_x)
    train_y = torch.Tensor((train_y.astype(np.long)))
    print(train_y.size(0))
    print((train_x.size(0)))
    # 封装好数据和标签
    train_dataset = TensorDataset(train_x,train_y)
    #test_dataset = TensorDataset(test_x, test_y)

    # 定义数据加载器
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)
    #test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)

    #实例化网络
    model = myLeNet5()
    if use_gpu:
        model = model.cuda()
        print('USE GPU')
    else:
        print('USE CPU')

    # 定义代价函数，使用交叉熵验证
    criterion = nn.CrossEntropyLoss(size_average=False)
    # 直接定义优化器，而不是调用backward
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))  #lr=0.001
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # 调用参数初始化方法初始化网络参数
    model.apply(weight_init)

    for epoch in range(1, 10000):
        print('----------------start train-----------------')
        train(epoch)
        if(optimizer.state_dict()['param_groups'][0]['lr']>0.000001 and epoch%10==0):
            scheduler.step()

        #test()
    writer.close()
    '''
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1):
        train = Variable(torch.from_numpy(train_x)) #.long()
        labels = Variable(torch.from_numpy(train_y))

        optimizer.zero_grad()
        y_pred = model(train)
        print(y_pred.shape)
        print(y_pred[1])
        loss = criterion(y_pred, labels)

        loss.backward()
        optimizer.step()

        print("loss: ", loss)
    '''
