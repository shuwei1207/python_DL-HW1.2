# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:07:14 2020

@author: SeasonTaiInOTA
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from skimage import img_as_float
import tensorflow as tf #ver 1.14.0

print('-----------loading data-----------')
#資料載入
train_data = pd.read_csv('train.csv')
#train_length = len(train_data)
train_length = 400  # memory

print('-----------data loaded-----------')

train_img_list =[]
train_label_list =[]

print('-----------processing data-----------')
#資料處理與切割
for i in range(train_length):
    name = train_data.iloc[i].filename
    img = cv2.imread('images/'+name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.rectangle(img, (train_data.iloc[i].xmin,train_data.iloc[i].ymin), (train_data.iloc[i].xmax,train_data.iloc[i].ymax), (0,255,0), 2)
    
    #資料切割
    c_img = img[train_data.iloc[i].ymin:train_data.iloc[i].ymax, train_data.iloc[i].xmin:train_data.iloc[i].xmax]
    c_img = cv2.resize(c_img,dsize=(28,28), interpolation = cv2.INTER_CUBIC)
    c_img = c_img.reshape(784)  #28*28
    c_img = img_as_float(c_img)
    #final_img = np.expand_dims(c_img,axis=0)
    label = train_data.iloc[i].label
    if label == 'good':
        label = 0
    elif label == 'bad':
        label = 1
    else:
        label = 2
    #cv2.imshow('img', c_img)
    #cv2.waitKey()
    
    train_img_list.append(c_img)
    train_label_list.append(label)

#轉換資料型態
train_img =np.array(train_img_list, dtype=np.float32)

#轉換label
number = 3
train_label = np.array([[0]*number]*train_length)
for i in range(train_length):
    train_label[i][train_label_list[i]] = 1 

#資料載入
test_data = pd.read_csv('test.csv')
test_length = len(test_data)


test_img_list =[]
test_label_list =[]

#資料處理與切割
for i in range(test_length):
    name = test_data.iloc[i].filename
    img = cv2.imread('images/'+name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.rectangle(img, (train_data.iloc[i].xmin,train_data.iloc[i].ymin), (train_data.iloc[i].xmax,train_data.iloc[i].ymax), (0,255,0), 2)
    
    c_img = img[test_data.iloc[i].ymin:test_data.iloc[i].ymax, test_data.iloc[i].xmin:test_data.iloc[i].xmax]
    c_img = cv2.resize(c_img,dsize=(28,28), interpolation = cv2.INTER_CUBIC)
    c_img = c_img.reshape(784)
    c_img = img_as_float(c_img)
    #final_img = np.expand_dims(c_img,axis=0)
    label = test_data.iloc[i].label
    if label == 'good':
        label = 0
    elif label == 'bad':
        label = 1
    else:
        label = 2
    #cv2.imshow('img', c_img)
    #cv2.waitKey()
    
    test_img_list.append(c_img)
    test_label_list.append(label)

#轉換資料型態
test_img =np.array(test_img_list, dtype=np.float32)

#轉換label
number = 3
test_label = np.array([[0]*number]*test_length)
for i in range(test_length):
    test_label[i][test_label_list[i]] = 1 

print('-----------data processed-----------')

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_xs, prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys, prob: 1})
    return result

def compute(v_xs):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_xs, prob: 1})
    y_pre = tf.argmax(y_pre,1)
    return y_pre

def weight_variable(shape):
    #初始值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #初始值
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def con2d(x, W):
    # 1, x_move, y_move, 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # 1, x_move, y_move, 1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print('-----------start training-----------')
x = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
y = tf.placeholder(tf.float32, [None, 3])
prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])

## conv1 layer
W_con1 = weight_variable([ 4, 4, 1,32]) # 3*3, in 1, out 32
b_con1 = bias_variable([32])
h_con1 = tf.nn.relu(con2d(x_image, W_con1) + b_con1)  # output 28*28*32
h_pool1 = max_pool_2x2(h_con1)   # output 14*14*32

## conv2 layer
W_con2 = weight_variable([4 , 4, 32, 64]) # 3*3, in 32, out 64
b_con2 = bias_variable([64])
h_con2 = tf.nn.relu(con2d(h_pool1, W_con2) + b_con2) # output 14*14*64
h_pool2 = max_pool_2x2(h_con2)   # output 7*7*64

## fc1 layer
W_1 = weight_variable([7*7*64, 1024])
b_1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_1) + b_1)
h_1_drop = tf.nn.dropout(h_1, prob)

## fc2 layer
W_2 = weight_variable([1024, 3])
b_2 = bias_variable([3])
prediction = tf.nn.softmax(tf.matmul(h_1_drop, W_2) + b_2)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()

# tf.initialize_all_variables()的處理
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

train_acc = []
test_acc = []

for i in range(1000):
    batch_x, batch_y = train_img,train_label
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, prob: 0.5})
    #predict_y = compute(test_img)
    
    if i % 20 == 0:
        train_acc.append(compute_accuracy(train_img, train_label))
        test_acc.append(compute_accuracy(test_img, test_label))
        print('iteration',i,': train_acc:',compute_accuracy(train_img, train_label),'test_acc:',compute_accuracy(test_img, test_label))
print('-----------end training-----------')  


plt.plot(train_acc)
plt.show()
plt.plot(test_acc)
plt.show()  