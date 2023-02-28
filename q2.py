import sys
import numpy as np # linear algebra
# np.set_printoptions(threshold=sys.maxsize)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.datasets import fashion_mnist

(x, y), (x_test, y_test) = fashion_mnist.load_data()
print(x.shape, y.shape, x_test.shape, y_test.shape)

x_temp = []
x_new = []
for i in range(x.shape[0]):
    x_temp.append(np.ravel(x[i]))
x = np.array(x_temp)
x_new = np.insert(x, 0, y, axis = 1)
x = np.array(x_new)



m, n = x.shape
np.random.shuffle(x) # shuffle before splitting into validation and training sets

data_val = x[0:1000].T
Y_val = data_val[0]
X_val = data_val[1:n]
X_val = X_val / 255.

data_train = x[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def sigmoid(self,x):
return 1.0 / (1.0 + np.exp(-x))

def tanh(self,x):
return np.tanh(x)

def relu(self, x):
    return np.maximum(0, x)

def deriv_sigmoid(self,x):
return self.sigmoid(x)*(1.0 - self.sigmoid(x))

def deriv_tanh(self,x):
return 1.0 - self.tanh(x)*self.tanh(x) 

def deriv_relu(self, x):
    return x > 0

def activation(self,x,n='sigmoid'): # Default: Sigmoid
    if n == 'sigmoid':
        return self.sigmoid(x)
    elif n == 'tanh':
        return self.tanh(x)
    elif n == 'relu':
        return self.relu(x)

def d_activation(self,x,n='sigmoid'): # Default: Sigmoid
    if n == 'sigmoid':
        return self.deriv_sigmoid(x)
    elif n == 'tanh':
        return self.deriv_tanh(x)
    elif n=='ReLu':
        return self.deriv_ReLu(x)

def softmax(self, x):
    return np.exp(x) / sum(np.exp(x))

def one_hot(Y):
    one_hot_Y = np.eye(np.max(Y) + 1)[Y]
    one_hot_Y = one_hot_Y.T
return one_hot_Y


    
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2,W1, W2, X, Y):
    m = Y.size
    # We need to do One-hot Encoding of Y
    dZ2 = A2 - one_hot(Y) # One-hot Encoded Value
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): # alpha - Learning Rate
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2,W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        #Printing every 50th term
        if i % 50 == 0:
            print('Iteration: ', i)
            print('Accuracy: ', get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2
    
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)
