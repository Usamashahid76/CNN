
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
X_train, X_test, y_train, y_test =  train_test_split(features, labels, test_size = 0.2, random_state = 0)
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

#LeNet Architecture Using Tensorflow