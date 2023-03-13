import os
from tqdm import tqdm
from termcolor import colored

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

from early_stopping import EarlyStopping
import matplotlib.pyplot as plt


batch_size = 128


################## 数据读取 归一化 重新排列 ##################

# 从txt中读取数据
dataset_X = np.genfromtxt("X_5种按压头.txt",delimiter=',')
dataset_Y = np.genfromtxt("Y_5种按压头.txt",delimiter=',')
# 每行最后有一个逗号，导致最后一列是nan，删除最后一列
dataset_X = np.delete(dataset_X,-1,1)
dataset_Y = np.delete(dataset_Y,-1,1)
print(dataset_X.shape) # (12500, 48)
print(dataset_Y.shape) # (12500, 10)
data_len = dataset_X.shape[0] # 12500
check_flag = input("Check the size of data (y/n) ")

# 对 Bx,By,Bz 分别进行归一化（认为16个Bx是一样的，Bx,By,Bz是不一样的）
dataset_X = dataset_X.reshape(-1,3) # (57600, 3)
scaler = MinMaxScaler()
scaler2 = Normalizer()
scaler3 = StandardScaler()
scaler.fit(dataset_X)
scaler2.fit(dataset_X)
scaler3.fit(dataset_X)
dataset_X = scaler3.transform(dataset_X)
print(dataset_X)
print(len(dataset_X))
check_flag = input("Check the size of data (y/n) ")
# 叠成一张张图 (25000,4,4,3)
imgset_X = []
for i in range(data_len):
    img_i = dataset_X[i*16 : (i+1)*16, :]
    img_i = img_i.reshape(4,4,3)
    imgset_X.append(img_i)
    
# print(np.array(imgset_X).shape) # (3600, 4, 4, 3)



################## 训练集、验证集、测试集准备 ##################

# 使用Pytorch 的DataSet和DataLoader，方便使用mini-batch和shuffle
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
 
    def __getitem__(self, idx):
        return self.data[idx]
 
    def __len__(self):
        return len(self.data)
#将数据按照batch_size的形式输入到pytorch里面
def process(dataX, dataY, batch_size, shuffle=True):
    seq = []
    for i in range(len(dataX)):
        x = torch.FloatTensor(dataX[i])
        y = torch.FloatTensor(dataY[i])
        seq.append((x,y))    
    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return seq


# 随机划分训练集、验证集和测试集
test_ratio = 0.1
val_ratio = 0.15
train_X, test_X, train_Y, test_Y = train_test_split(imgset_X, dataset_Y, test_size=test_ratio, random_state=4000) 
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=val_ratio, random_state=4000) 

# 将X和Y匹配好，加载到Dataloader里
Dtr = process(train_X, train_Y, batch_size, True)
val = process(val_X, val_Y, batch_size, True)
Dte = process(test_X, test_Y, batch_size, True)
# print(len(val))
# for x,y in Dtr:
#     print("y.shape="+str(y.shape))
#     print("x.shape="+str(x.shape))
#     break



################## 神经网络结构定义 ##################

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d( in_channels = 3, out_channels = 32, kernel_size = 3, padding='same' ),
            nn.ReLU(),
            # nn.MaxPool2d( kernel_size = 2 ),
            nn.BatchNorm2d( num_features = 32 )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d( in_channels = 32, out_channels = 32, kernel_size = 3, padding='same' ),
            nn.ReLU(),
            # nn.MaxPool2d( kernel_size = 2 ),
            nn.BatchNorm2d( num_features = 32 )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d( in_channels = 32, out_channels = 32, kernel_size = 3, padding='same'),
            nn.ReLU(),
            # nn.MaxPool2d( kernel_size = 2 ),
            nn.BatchNorm2d( num_features = 32 )
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d( in_channels = 32, out_channels = 32, kernel_size = 3, padding='same'),
            nn.ReLU(),
            # nn.MaxPool2d( kernel_size = 2 ),
            nn.BatchNorm2d( num_features = 32 )
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d( in_channels = 32, out_channels = 32, kernel_size = 3, padding='same'),
            nn.ReLU(),
            # nn.MaxPool2d( kernel_size = 2 ),
            nn.BatchNorm2d( num_features = 32 )
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d( in_channels = 32, out_channels = 32, kernel_size = 3, padding='same'),
            nn.ReLU(),
            # nn.MaxPool2d( kernel_size = 2 ),
            nn.BatchNorm2d( num_features = 32 )
        )
        self.fc1 = nn.Linear(32*4*4, 5)
        
    def forward(self, x):
        #                x.shape: [batch_size, h=4, w=4, c=3]
        x = x.permute(0,3,1,2)  # [batch_size,  c, h, w]
        x = self.conv1(x)       # [batch_size, 16, h, w]
        x = self.conv2(x)       # [batch_size, 16, h, w]
        x = self.conv3(x)       # [batch_size, 16, h, w]
        x = self.conv4(x)       # [batch_size, 16, h, w]
        x = self.conv5(x)       # [batch_size, 16, h, w]
        x = self.conv6(x)       # [batch_size, 16, h, w]
        x = x.reshape(x.shape[0],-1) # [batch_size, 16*h*w]
        x = self.fc1(x)         # [batch_size, 3]
        return x
    



################## 训练 ##################

model_dir = './saved_CNN'
# initialize early stopping instance and defining model saved path
early_stopping = EarlyStopping(model_dir)

# initialize a network instance
model = CNN()
# print(model)

# # warm start if applicable
# model_path = os.path.join(model_dir, 'best_model.pth')
# if os.path.exists(model_path):
#     print(colored('------------loading model for warm start-------------', 'green'))
#     model.load_state_dict(torch.load(model_path))
#     print(colored('-----------------model load complete-----------------', 'green'))

# define model's loss function, optimizer, and scheduler
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


######################### training #########################

epochs = 100
print(colored('-------------Training-------------', 'blue'))
for e in range(epochs):
        
    # 使用训练集训练模型
    correct = 0
    total = 0
    train_loss = []
    model.train() # 模型进入训练模式
    for (seq, label) in Dtr:
        # 前向传播
        y_pred = model(seq)
        loss = loss_func(y_pred, label)
        train_loss.append(loss.item())
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算准确率
        _, y_pred_idx = torch.max(y_pred, dim=1)  # 返回每行最大值和最大值所在的下标
        _, label_idx = torch.max(label, dim=1)  # 返回每行最大值和最大值所在的下标
        total += y_pred_idx.size(0)
        correct += (y_pred_idx == label_idx).sum().item()
        #print(y_pred_idx)
        #print(label_idx)
        #check_flag = input("Right (y/n) ")
        #if y_pred_idx != label_idx:
        #    print (y_pred_idx,label_idx)

    scheduler.step()
    train_loss = np.mean(train_loss)
    train_accr = correct / total * 100
    
    
    # 使用验证集验证模型
    correct = 0
    total = 0
    val_loss = []
    model.eval() # 模型进入预测模式
    with torch.no_grad(): # 不需要反向传播, 不需要生成计算图
        for (seq, label) in val:
            y_pred = model(seq)
            l = loss_func(y_pred, label)
            val_loss.append(l.item())
            # 计算准确率
            _, y_pred_idx = torch.max(y_pred, dim=1)  # 返回每行最大值和最大值所在的下标
            _, label_idx = torch.max(label, dim=1)  # 返回每行最大值和最大值所在的下标
            total += y_pred_idx.size(0)
            correct += (y_pred_idx == label_idx).sum().item()
            #print(y_pred_idx)
            #print(label_idx)
            #check_flag = input("Right (y/n) ")
    val_loss = np.mean(val_loss)
    val_accr = correct / total * 100
    
    # 输出本epoch的训练损失和验证集损失
    print('epoch {:03d} train_loss {:.8f} val_loss {:.8f} train_accuracy {:.8f} val_accuracy {:.8f}'.format(e+1, train_loss, val_loss, train_accr, val_accr))
        
    # 将本epoch的验证集损失与此前损失比较, 判断是否满足早停止条件, 满足时early_stop会被置为True
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break # 跳出迭代，结束训练
    
    


######################### testing #########################

print(colored('-------------Testing-------------', 'blue'))


correct = 0
total = 0
test_loss = []
number = 0
model.eval() # 模型进入预测模式
with torch.no_grad(): # 不需要反向传播, 不需要生成计算图
    for (seq, label) in Dte:
        y_pred = model(seq)
        l = loss_func(y_pred, label)
        test_loss.append(l.item())
        # 计算准确率
        _, y_pred_idx = torch.max(y_pred, dim=1)  # 返回每行最大值和最大值所在的下标
        _, label_idx = torch.max(label, dim=1)  # 返回每行最大值和最大值所在的下标
        total += y_pred_idx.size(0)
        correct += (y_pred_idx == label_idx).sum().item()
        print(y_pred_idx)
        print(label_idx)
        #check_flag = input("Right (y/n) ")
        number = number +1
test_loss = np.mean(test_loss)
test_accr = correct / total * 100

print('test result: test_loss {:.8f} test_accr {:.8f}'.format(test_loss, test_accr))
print(number)
