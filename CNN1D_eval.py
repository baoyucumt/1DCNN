import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import math
from tensorboardX import SummaryWriter
from MyLeNet5addone import MyLeNet5addone
import  matplotlib.pyplot as plt

BATCH_SIZE=40


#定义一些超参数
use_gpu = torch.cuda.is_available()  #是否使用显卡
kwargs = {'num_workers': 2, 'pin_memory': True}                              #DataLoader的参数
writer = SummaryWriter('runs/3cov8')
global acct,losst
losst=0.0
acct=0.0
xx = []
yy = []

#定义测试函数
def test(epoch):
    model.eval()                                                             #让模型变为测试模式，主要是保证dropout和BN和训练过程一致。BN是指batch normalization
    model.training=False
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    for data, target in test_loader:
        global yy
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        #计算总的损失
        data, target = Variable(data, volatile=True), Variable(target).long()
        output = model(data)
        #yy=torch.from_numpy(np.array(yy).T)*np.array(cov1).detach()
        _, preds = torch.max(output, 1)

        # 计算总的损失
        test_loss = criterion(output, target.squeeze())#.data[0]
        pred = output.data.max(1, keepdim=True)[1]  # 获得得分最高的类别
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def feature_imshow(inp, title=None):
    """Imshow for Tensor."""

    inp = inp.detach().numpy().transpose((1, 2, 0))

    mean = np.array([0.5, 0.5, 0.5])

    std = np.array([0.5, 0.5, 0.5])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    '''装载test集合'''
    string0 = np.loadtxt('test_data.txt', dtype=np.float32)
    test_y = string0[:, 0].reshape(-1, 1)  # 1行
    test_x = string0[:, 1:].reshape(120, 1,1,70) # 937个样本向量化
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor ((test_y.astype(np.long)))
    print(test_y.size(0))
    print((test_x.size(0)))
    # 封装好数据和标签

    X_Show = test_x.reshape(test_x.shape[0], 1, 70)
    X_Show /= 2.0
    print("img shape:{}".format(X_Show[0].shape))
    test_img = X_Show[0]
    #for item in X_Show[1:20]:
    #    test_img = np.append(test_img, item, axis=1)

    for i in range(70):
        xx.append(i)
        yy.append(X_Show[1][0][i])
    fig1=plt.plot(xx,yy)
    #plt.show(fig1)
    #cv2.imshow("test1", test_img)

    #train_dataset = TensorDataset(train_x,train_y)
    test_dataset = TensorDataset(test_x, test_y)

    # 定义数据加载器
    #train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=40, **kwargs)

    #实例化网络
    model = MyLeNet5addone()
    #model = describe_model()
    checkpoint = torch.load('mymodel/enmodel.pt')
    model.load_state_dict(checkpoint['state_dict'])


    if use_gpu:
        model = model.cuda()
        print('USE GPU')
    else:
        print('USE CPU')
    #criterion = torch.nn.BCELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 定义代价函数，使用交叉熵验证
    criterion = nn.CrossEntropyLoss(size_average=False)
    # 直接定义优化器，而不是调用backward
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))

    for epoch in range(1):

        test(epoch)
        #optimizer.step()
    feature_output1 = model.featuremap1 #.transpose(1, 0).cpu()
    #out = torchvision.utils.make_grid(feature_output1)
    #feature_imshow(out)
    zz2 = []
    xx2 = []
    global fig2
    if model.showlayer==1:
        for k in range(32):
            zz = []
            for j in range(12):
                zz.append(0)
            for j in range(45):
                zz.append(feature_output1[0][k][0][j])
            for j in range(13):
                zz.append(0)
            zz2.append(zz)
            fig2 = plt.plot(xx, zz2[k])
    elif model.showlayer == 2:
        for k in range(32):
            zz = []
            for j in range(24):
                zz.append(0)
            for j in range(22):
                zz.append(feature_output1[0][k][0][j])
            for j in range(24):
                zz.append(0)
            zz2.append(zz)
            fig2 = plt.plot(xx, zz2[k])
    elif model.showlayer == 3:
        for j in range(11):
            xx2.append(j)
        for k in range(64):
            zz = []
            #for j in range(29):
            #    zz.append(0)
            for j in range(11):
                zz.append(feature_output1[0][k][0][j])
            #for j in range(30):
            #    zz.append(0)
            zz2.append(zz)
            fig2 = plt.plot(xx2, zz2[k])
    elif model.showlayer == 4:
        for j in range(6):
            xx2.append(j)
        for k in range(64):
            zz = []
            #for j in range(32):
            #    zz.append(0)
            for j in range(6):
                zz.append(feature_output1[0][k][0][j])
            #for j in range(32):
            #    zz.append(0)
            zz2.append(zz)
            fig2 = plt.plot(xx2, zz2[k])
    elif model.showlayer == 5:
        zz = []
        for j in range(128):
            xx2.append(j)
        for k in range(128):
            zz.append(feature_output1[0][k])
        fig2 = plt.plot(xx2, zz)
    elif model.showlayer == 6:
        zz = []
        for j in range(64):
            xx2.append(j)
        for k in range(64):
            zz.append(feature_output1[0][k]*10)
        fig2 = plt.plot(xx2, zz)
    plt.show(fig2)