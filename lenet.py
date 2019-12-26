# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

# Getting training and test set
training_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
training_set.head()
# Getting training and test set

# Creating the features and labels in training set
training_set = pd.get_dummies(training_set,columns=["label"])
features = training_set.iloc[:, : -10].values
labels = training_set.iloc[:,-10 :].values
# Creating the features and labels in training set

#Separting the train set and test set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(features, labels, test_size = 0.2, random_state = 1212)
X_test, X_validation, y_test, y_validation =  train_test_split(X_test, y_test, test_size = 0.2, random_state = 0)
#Separting the train set and test set split

#Reshaping the data format into 28*28*1
image_size = 28
n_labels = 10
n_channels = 1

def reshape_format(dataset, labels):
    dataset = dataset.reshape((-1, image_size,image_size,n_channels)).astype(np.float32)
    labels = (np.arange(n_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reshape_format(X_train, y_train)
validation_dataset, validation_labels = reshape_format(X_validation, y_validation)
test_dataset, test_labels = reshape_format(X_test, y_test)

test_set = test_set.as_matrix().reshape((-1,image_size,image_size,n_channels)).astype(np.float32)

print ('Training set :', train_dataset.shape, train_labels.shape)
print ('Validation set :', validation_dataset.shape, validation_labels.shape)
print ('Test set :', test_dataset.shape, test_labels.shape)
#Reshaping the data format into 28*28*1

#Padding the sets
X_train = np.pad(train_dataset, ((0,0),(2,2),(2,2),(0,0)),'constant')
X_validation = np.pad(validation_dataset, ((0,0),(2,2),(2,2),(0,0)),'constant')
X_test = np.pad(test_dataset, ((0,0),(2,2),(2,2),(0,0)),'constant')

print ('Training set after padding 2x2    :', X_train.shape, train_labels.shape)
print ('Validation set after padding 2x2  :', X_validation.shape, validation_labels.shape)
print ('Test set after padding 2x2        :', X_test.shape, test_labels.shape)
print ('Submission data after padding 2x2 :', X_test.shape)
#Padding the sets

#LeNet Architecture Using Tensorflow
def LeNetArchitecture(x):
    
    #Layer 1 Conv, Activation, Pooling (MAX)
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 6], mean = 0, stddev = 0.1))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides = [1, 1, 1, 1], padding = ('VALID')) + conv1_b
    conv1 = tf.nn.relu(conv1)
    pool_1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding= ('VALID'))
    
    #Layer 2 Conv, Activation, Pooling (MAX), flatten
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 6, 16], mean = 0, stddev = 0.1 ))
    conv2_b = tf.Variable(tf.zeros(6))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1, 1, 1, 1], padding = 'VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    pool_2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    fc1 = flatten(pool_2)
    
    #Layer 3 Fully Connected, Activation
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400, 120), mean = 0, stddev = 0.1))
    fc2_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)
    
    #Layer 4 Fully Connected, Activation
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120, 84), mean = 0, stddev = 0.1))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc2, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)
    
    #layer 5 Fully Connected
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84, 10), mean = 0, stddev = 0.1))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc3, fc3_w) + fc3_b
    
    return logits    
#LeNet Architecture Using Tensorflow

#Placeholders for x and y
x = tf.placeholder(tf.float32, shape = [None, 32, 32, 1])
y_ = tf.placeholder(tf.int32, (None))
#Placeholders for x and y

#Softmax Classifier with AdamOptimizer
logits = LeNetArchitecture(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)
#Softmax Classifier with AdamOptimizer

#Train and evaluate the Model
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    n_examples = len(X_data)
    t_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, n_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset: offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict = { x: batch_x, y_ : batch_y})
        t_accuracy += (accuracy * len(batch_x))
    return t_accuracy / n_examples
#Train and evaluate the Model
    
#Initialize Session and Run

EPOCHS = 10
BATCH_SIZE = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_examples = len(X_train)
    print("Training... with dataset - ", n_examples)
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, n_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x , batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict = {x: batch_x, y_: batch_y})
            validation_accuracy = evaluate(X_validation, y_validation)
             print("EPOCH {} ...".format(i+1))
             print("Validation Accuracy = {:.3f}".format(validation_accuracy))
             print()
             
    saver = tf.train.Saver()
    save_path = saver.save(sess, '/tmp/lenet.ckpt')
    print("Modal saved %s "%save_path)
    test_accuracy = evalauate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
#Initialize Session and Run
    
#Launch the model 
with tf.Session() as sess:
    saver.restore(sess, '/tmp/lenet.ckpt')
    print("Modal Restored")
    Z = logits.eval(feed_dict = {x: submission_test})
    y_pred = np.argmax(Z, axis = 1)
    print(len(y_pred))
    
    submission = pd.DataFrame({
            "ImageId" : list(range(1, len(y_pred)+1)),
            "Label" : y_pred
            })
    submission.to_csv('minst.csv', index = False)







