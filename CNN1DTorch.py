import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import math
import time
from tensorboardX import SummaryWriter
from MyLeNet5addone import MyLeNet5addone
from MyLeNet5 import myLeNet5

BATCH_SIZE=40

#定义一些超参数
use_gpu = torch.cuda.is_available()  #是否使用显卡
kwargs = {'num_workers': 2, 'pin_memory': True}                              #DataLoader的参数
writer = SummaryWriter('runs/3cov9')
ModelPATH = 'mymodel/enmodel.pt'
global acct,losst
losst=0.0
acct=0.0
timecostt=0.0

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
    train_datatimes=7  #训练集拆成的batch次数
    global acct
    global losst
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target.long())                    #定义为Variable类型，能够调用autograd
        #初始化时，要清空梯度
        optimizer.zero_grad()
        #if use_gpu:
        #torch.cuda.synchronize()
        start = time.time()
        output = model(data)
        #if use_gpu:
        #torch.cuda.synchronize()
        end = time.time()
        global timecostt
        timecostt+=(end-start)
        print(batch_idx, end-start,timecostt)
        _, preds = torch.max(output, 1)
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()                                                     #相当于更新权重值
        running_corrects += torch.sum(preds == target.squeeze())
        acct+= torch.sum(preds == target.squeeze())
        losst=losst+loss.item()
        #print(batch_idx, loss.item(),losst)
        writer.add_scalar('train/lr',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
        #print(epoch, running_corrects, acct)
        if batch_idx % train_datatimes == train_datatimes-1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t ACC:{:.3f}'.format(
               epoch, batch_idx , len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(),running_corrects.double()/(BATCH_SIZE*train_datatimes)))
            writer.add_scalar('train/acc', (acct.double() / (BATCH_SIZE*((epoch)*train_datatimes))), epoch)
            writer.add_scalar('train/acc2', running_corrects.double()/BATCH_SIZE*train_datatimes, epoch)
            writer.add_scalar('train/loss', loss.item(), epoch)
            writer.add_scalar('train/loss2', losst/ (BATCH_SIZE*((epoch)*train_datatimes)), epoch)
            print(timecostt/(BATCH_SIZE*((epoch)*train_datatimes)))


if __name__ == '__main__':
    string0 = np.loadtxt('train_data.txt', dtype=np.float32)  #alldata is 400
    train_y = string0[:, 0].reshape(-1, 1)  # 1行
    train_x = string0[:, 1:].reshape(280, 1,1,70)
    #string0 = np.loadtxt('test_data.txt', dtype=np.float32)
    #test_y = string0[:, 0].reshape(-1, 1).T  # 1行
    #test_x = string0[:, 1:].reshape(120, -1).T  # 937个样本向量化
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
    #model = MyLeNet5addone()
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

    for epoch in range(1, 500):
        print('----------------start train-----------------')
        train(epoch)
        if(optimizer.state_dict()['param_groups'][0]['lr']>0.000001 and epoch%10==0):
            scheduler.step()

        #test()
    writer.close()
    """save/load Entire Model"""

    #torch.save(model, ModelPATH)
    torch.save({'state_dict': model.state_dict()},ModelPATH)
    #model = torch.load('mymodel/enmodel.pt')

