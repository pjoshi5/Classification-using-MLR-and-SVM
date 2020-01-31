#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC


# In[18]:


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


# In[19]:


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# In[20]:


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))
    
    ##################
    # YOUR CODE HERE #
    ##################
    initialWeights = initialWeights.reshape((n_feature+1,1))
    
    # HINT: Do not forget to add the bias term to your input data
    train_data = np.concatenate((np.ones((train_data.shape[0],1)),train_data), axis = 1)

    error = -(np.sum(labeli*np.log(sigmoid(np.dot(train_data,initialWeights)))+ (1-labeli)*np.log(1-sigmoid(np.dot(train_data,initialWeights)))))/len(train_data)
    
    error_grad = np.sum((sigmoid(np.dot(train_data,initialWeights)) - labeli) * train_data, 0)/len(train_data)
    
    return error, error_grad


# In[21]:


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data = np.concatenate((np.ones((data.shape[0],1)),data), axis = 1)
    
    #Every class Max 
    label = np.argmax(sigmoid(np.dot(data, W)), axis=1)
    label = label.reshape((data.shape[0],1))
    
    return label


# In[22]:


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    initialWeights_b = params.reshape((n_feature+1,10))

    # HINT: Do not forget to add the bias term to your input data
    train_data = np.concatenate((np.ones((train_data.shape[0],1)),train_data), axis = 1)        
    
    denominator = np.sum(np.exp(np.dot(train_data,initialWeights_b)),axis=1)
    denominator = np.expand_dims(np.sum(np.exp(np.dot(train_data,initialWeights_b)), 1),1)

    error = - (np.sum(np.sum(labeli*np.log(np.exp(np.dot(train_data,initialWeights_b))/denominator))))
    
    error_grad = np.dot(train_data.transpose(),((np.exp(np.dot(train_data,initialWeights_b))/denominator)-labeli))
    error_grad = error_grad.ravel()
    
    return error, error_grad


# In[23]:


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data = np.concatenate((np.ones((data.shape[0],1)),data), axis = 1)
    
    denominator = np.sum(np.exp(np.dot(data,W)),axis=1)
    denominator = np.expand_dims(np.sum(np.exp(np.dot(data,W)), 1),1)
    
    #Every class Max 
    label = np.argmax(((np.exp(np.dot(data,W)))/denominator),axis=1)
    label = label.reshape(data.shape[0],1)
    
    return label


# In[28]:


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
blr_trainConf = confusion_matrix(train_label,predicted_label)
print("\n BLR Training Confusion Matrix \n", blr_trainConf)
blr_TrainClassificationReport = metrics.classification_report(train_label,predicted_label)
print("\n BLR Training Classification Report \n",blr_TrainClassificationReport)

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
blr_testConf = confusion_matrix(test_label,predicted_label)
print("\n BLR Test Confusion Matrix \n", blr_testConf)
blr_TestClassificationReport = metrics.classification_report(test_label,predicted_label)
print("\n BLR Test Classification Report \n",blr_TestClassificationReport)


# In[21]:


"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

train_data_idx_svm = np.random.randint(n_train, size=10000)
train_data_svm = train_data[train_data_idx_svm,:]
train_label_svm = train_label[train_data_idx_svm,:]


# In[23]:


## Using linear kernel to fit the data

lin_model_svm = svm.SVC(kernel='linear')
lin_model_svm.fit(train_data_svm, train_label_svm.flatten())
print('Training set Accuracy with SVM Linear model:' + str(100*lin_model_svm.score(train_data_svm, train_label_svm)) + '%')
print('Validation set Accuracy with SVM Linear model:' + str(100*lin_model_svm.score(validation_data, validation_label)) + '%')
print('Test set Accuracy with SVM Linear model:' + str(100*lin_model_svm.score(test_data, test_label)) + '%')


# In[24]:


## using Radial Basis function with gamma setting to 1

radial_model_gamma1_svm = svm.SVC(kernel='rbf', gamma = 1.0)
radial_model_gamma1_svm.fit(train_data_svm, train_label_svm.flatten())
print('Training set Accuracy with SVM Radial basis gamma set to 1 model:' + str(100*radial_model_gamma1_svm.score(train_data_svm, train_label_svm)) + '%')
print('Validation set Accuracy with SVM Radial basis gamma set to 1 model:' + str(100*radial_model_gamma1_svm.score(validation_data, validation_label)) + '%')
print('Test set Accuracy with SVM Radial basis gamma set to 1 model:' + str(100*radial_model_gamma1_svm.score(test_data, test_label)) + '%')


# In[25]:


## using Radial Basis function with gamma setting to default

radial_model_gammadef_svm = svm.SVC(kernel='rbf')
radial_model_gammadef_svm.fit(train_data_svm, train_label_svm.flatten())
print('Training set Accuracy with SVM radial basis with default gamma:' + str(100*radial_model_gammadef_svm.score(train_data_svm, train_label_svm)) + '%')
print('Validation set Accuracy with SVM radial basis with default gamma:' + str(100*radial_model_gammadef_svm.score(validation_data, validation_label)) + '%')
print('Test set Accuracy with SVM radial basis with default gamma:' + str(100*radial_model_gammadef_svm.score(test_data, test_label)) + '%')


# In[26]:


## using Radial Basis function with gamma setting to default and varying c values(1,10,20 ,..., 100)

c_val = np.zeros(11)
train_acc = np.zeros(11)
validation_acc = np.zeros(11)
test_acc = np.zeros(11)

c_val[1:] = [x for x in np.arange(10, 101, 10)]
c_val[0] = 1

for i in range(len(c_val)):
    radial_model_gammadef_varc_svm = svm.SVC(kernel='rbf',C = c_val[i])
    radial_model_gammadef_varc_svm.fit(train_data_svm, train_label_svm.flatten())
    train_acc[i] = 100 * radial_model_gammadef_varc_svm.score(train_data_svm, train_label_svm)
    validation_acc[i] = 100*radial_model_gammadef_varc_svm.score(validation_data, validation_label)
    test_acc[i] = 100*radial_model_gammadef_varc_svm.score(test_data, test_label)


# In[39]:


## Plot of variation in accuracies with respect to c values

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(111)
ax1.plot(c_val,train_acc,linestyle='-', c='b', marker="o", label='training accuracies')
ax1.plot(c_val,validation_acc,linestyle='-', c='r', marker="o", label='validation accuracies')
ax1.plot(c_val,test_acc, c='y', marker="o", label='test accuracies')
plt.title("comparison of tranning validation and test accuracy in radial SVM for different c values")
plt.xlabel("C values")
plt.ylabel("Accuracies")
plt.legend(loc='upper left');
plt.show

fig.savefig("accuracy.pdf", bbox_inches='tight')


# In[40]:


print("train_acc",train_acc)
print("validation_acc",validation_acc)
print("test_acc",test_acc)


# In[41]:


## Training with the whole dataset

svm_modl_fit = svm.SVC(kernel='rbf', C = 70)
svm_modl_fit.fit(train_data, train_label.flatten())
print('Training set Accuracy with SVM best model:' + str(100*radial_model_gammadef_svm.score(train_data, train_label)) + '%')
print('Validation set Accuracy with SVM best model:' + str(100*radial_model_gammadef_svm.score(validation_data, validation_label)) + '%')
print('Test set Accuracy with SVM best model:' + str(100*radial_model_gammadef_svm.score(test_data, test_label)) + '%')


# In[29]:


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
mlr_trainConf = confusion_matrix(train_label,predicted_label_b) 
print(" MLR Training Confusion Matrix \n", mlr_trainConf)
mlr_TrainClassificationReport = metrics.classification_report(train_label,predicted_label_b)
print("\n MLR Training Classification Report \n",mlr_TrainClassificationReport)

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
mlr_testConf = confusion_matrix(test_label,predicted_label_b)
print("\n MLR Test Confusion Matrix \n",mlr_testConf)
mlr_TestClassificationReport = metrics.classification_report(test_label,predicted_label_b)
print("\n MLR Test Classification Report \n",mlr_TestClassificationReport)

