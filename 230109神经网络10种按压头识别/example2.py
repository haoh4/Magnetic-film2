from tqdm import tqdm
from termcolor import colored
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch
import os
from sklearn.model_selection import train_test_split
def loadDataFrame(path1,path2):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #删除报错
    dataset_X = np.genfromtxt(path1,delimiter=',')      
    dataset_Y = np.genfromtxt(path2,delimiter=',')  



    dataset_X = np. delete(dataset_X,[48],1) #删除最后一个逗号
    dataset_Y = np. delete(dataset_Y,[3],1)  #删除最后一个逗号

    '''
    num_max = 0
    num_min = 0
    for i in range(len(dataset_X)):
        for j in range(len(dataset_X[i])):
            if dataset_X[i][j]>=num_max:
                num_max = dataset_X[i][j]  #找出整列数组中的最大值
            if dataset_X[i][j]<=num_min:
                num_min= dataset_X[i][j]  #找出整列数据中的最小值

    dataset_X1 = (dataset_X+(-num_min))/(num_max-num_min) # 把dataset_X转换到（0，1）区间上
    '''
    # normalization 把X值进行一个的normalize
    scalerX = MinMaxScaler()
    scalerX.fit(dataset_X)
    dataset_X2 = scalerX.transform(dataset_X)

    check_flag = input("Check the size of data (y/n) ")

    return dataset_X2, dataset_Y, dataset_X2.shape, dataset_Y.shape

data_path1 = '汇总.txt'
data_path2 = 'Y.txt'
dataX, dataY, shapeX, shapeY = loadDataFrame(data_path1,data_path2)
print(dataX)
print(dataY)
print(shapeX)
print(shapeY)


def splitDataFrame(dataX, dataY, percentTest=0.15,random_state=1000):
    """ split dataframe into training set, validaton set, and test set
        Args:
            dataX (np.array): array of features
            dataY (np.array): array of labels
            train_ratio (float, optional): ratio of training set. Defaults to 0.8. Normally Dtr:val:Dte = 8:1:1 or 6:2:2
            val_ratio (float, optional): ratio of validation set. Defaults to 0.1. Normally Dtr:val:Dte = 8:1:1 or 6:2:2
        Returns: Dtr, val, Dte
            Dtr (tuple): training set   (feature as np.array, label as np.array)
            val (tuple): validation set (feature as np.array, label as np.array)
            Dte (tuple): test set       (feature as np.array, label as np.array)
    """
    percentTest = 0.15 #百分之二十的数据为测试数据
    #将数据集进行无序打乱，生成训练集和测试集
    train_X, test_X, train_Y, test_Y= train_test_split(dataX,dataY,test_size=float(percentTest),random_state=1000) 
    # random_state用来产生数组的无序性
    # Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

    return (train_X, train_Y), (test_X, test_Y)

Dtr, Dte = splitDataFrame(dataX, dataY, percentTest=0.15,random_state=1000)

print(Dtr)
print(Dte)