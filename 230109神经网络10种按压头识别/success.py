import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import os

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import activations

# from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LeakyReLU

from plott import *
#import utils.plot as my_plot
#import utils.process as my_process
#from models.Response import Response

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #删除报错
dataset_X = np.genfromtxt("汇总.txt",delimiter=',')      # 820 * (999)
dataset_Y = np.genfromtxt("Y.txt",delimiter=',')      # 820 * (2004)



dataset_X = np. delete(dataset_X,[48],1) #删除最后一个逗号
dataset_Y = np. delete(dataset_Y,[3],1)  #删除最后一个逗号
print(dataset_X) 
num_max = 0
num_min = 0
for i in range(len(dataset_X)):
    for j in range(len(dataset_X[i])):
        if dataset_X[i][j]>=num_max:
            num_max = dataset_X[i][j]  #找出整列数组中的最大值
        if dataset_X[i][j]<=num_min:
            num_min= dataset_X[i][j]  #找出整列数据中的最小值a

dataset_X = dataset_X+(-num_min)/(num_max-num_min) # 把dataset_X转换到（0，1）区间上
dataset_X = tf.clip_by_value(dataset_X, clip_value_min=0., clip_value_max=1.).numpy()
dataset_Y = tf.clip_by_value(dataset_Y, clip_value_min=0., clip_value_max=1.).numpy()
print(num_max)
print(num_min)

print("train_X shape",dataset_X.shape)       # 3600 * 48
print("train_Y shape",dataset_Y.shape)       # 3600 * 3
print(dataset_X) 
check_flag = input("Check the size of data (y/n) ")

percentTest = 0.15 #百分之二十的数据为测试数据

train_X, test_X, train_Y, test_Y= train_test_split(dataset_X,dataset_Y,test_size=float(percentTest),random_state=1000) #将数据集进行无序打乱，生成训练集和测试集
# random_state用来产生数组的无序性
# Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.










n_hidden=300
x_size = train_X.shape[1]
y_size = train_Y.shape[1]
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)

model = keras.Sequential([
keras.layers.Dense(n_hidden,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="dense_1", input_shape=(x_size,)),
keras.layers.Dense(n_hidden,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="dense_2"),
keras.layers.Dense(n_hidden,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="dense_3"),
keras.layers.Dense(n_hidden,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="dense_4"),
keras.layers.Dense(n_hidden,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="dense_5"),
keras.layers.Dense(n_hidden,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="dense_6"),
keras.layers.Dense(n_hidden,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="dense_add1"),
keras.layers.Dense(n_hidden,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="dense_add2"),
# keras.layers.Dense(y_size,kernel_initializer=initializer,bias_initializer=initializer,activation="relu", name="predictions"),
keras.layers.Dense(y_size,kernel_initializer=initializer,bias_initializer=initializer, name="predictions")])


#optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

my_callbacks = [
# tf.keras.callbacks.EarlyStopping(patience=30),   # restore_best_weights=False/True
# tf.keras.callbacks.ModelCheckpoint(filepath='./model_saved_dim/model.{epoch:02d}-{val_loss:.2f}.h5'),
tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

model.compile(optimizer='adam',loss = 'mean_squared_error', 
                metrics=['accuracy']) 

history = model.fit(train_X,train_Y, callbacks=my_callbacks, epochs=20, batch_size=300, validation_data=(test_X, test_Y))
    
print(model.summary())

fig = plt.figure(figsize = (8,6))
tick_mul = 1.3   # for ticks
local_linewidth = 2.5
truth_local_linewidth = 3.4
ax = plt.gca()
x = range(1, len(history.history['accuracy']) + 1)
plt.plot(x, history.history['accuracy'], linewidth=truth_local_linewidth, color='#FFC93C', label='Training accuracy')
plt.plot(x, history.history['val_accuracy'], linewidth=truth_local_linewidth, color='#7E0CF5', label='Validation accuracy')
plt.legend(loc='upper left')
plt.show()