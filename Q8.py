

# !pip install wandb
# !wandb login --relogin

#importing required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import wandb
import argparse
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

np.random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument("-wp","--wandb_project", default='Testing', type=str, required=True, help='Project name used to track experiments in Weights & Biases dashboard')

parser.add_argument("-we", "--wandb_entity", default='dl_research', type=str, required=True, help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')

parser.add_argument("-d", "--dataset", default='fashion_mnist', type=str, required=False, help='choices: ["mnist", "fashion_mnist"]', choices=["mnist", "fashion_mnist"])

parser.add_argument("-e", "--epochs", default=20, type=int, required=False, help='Number of epochs to train neural network.')

parser.add_argument("-b", "--batch_size", default=64,  type=int, required=False,help='Batch size used to train neural network.')

parser.add_argument("-l", "--loss", default='cross_entropy',  type=str, required=False,help='choices: ["mean_squared_error", "cross_entropy"]', choices=["mean_squared_error", "cross_entropy"])

parser.add_argument("-o", "--optimizer", default='nadam',  type=str, required=False,help='choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])

parser.add_argument("-lr", "--learning_rate", default=0.005,  type=float, required=False,help='Learning rate used to optimize model parameters')

parser.add_argument("-m", "--momentum", default=0.9,  type=float, required=False,help='Momentum used by momentum and nag optimizers.')

parser.add_argument("-beta", "--beta", default=0.9,  type=float, required=False,help='Beta used by rmsprop optimizer')

parser.add_argument("-beta1", "--beta1", default=0.9,  type=float, required=False,help='Beta1 used by adam and nadam optimizers.')

parser.add_argument("-beta2", "--beta2", default=0.999, type=float, required=False,help='Beta2 used by adam and nadam optimizers.')

parser.add_argument("-eps", "--epsilon", default=1e-8,  type=float, required=False,help='Epsilon used by optimizers.')

parser.add_argument("-w_d", "--weight_decay", default=0.0 , type=float, required=False,help='Weight decay used by optimizers.')

parser.add_argument("-w_i", "--weight_init",  type=str, required=False,default='Xavier', help='choices: ["random", "Xavier"]', choices= ["random", "Xavier"])

parser.add_argument("-nhl", "--num_layers",  type=int, required=False,default=5, help='Number of hidden layers used in feedforward neural network.')

parser.add_argument("-sz", "--hidden_size",  type=int, required=False,default=512, help='Number of hidden neurons in a feedforward layer.')

parser.add_argument("-a", "--activation", default='sigmoid',  type=str, required=False,help='choices: ["identity", "sigmoid", "tanh", "ReLU"]', choices=["identity", "sigmoid", "tanh", "ReLU"])


args = parser.parse_args()

wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
dataset = args.dataset
epochs = args.epochs
batch_size = args.batch_size
loss = args.loss
optimizer = args.optimizer
learning_rate = args.learning_rate
momentum = args.momentum
beta = args.beta
beta1 = args.beta1
beta2 = args.beta2
epsilon = args.epsilon
weight_decay = args.weight_decay
weight_init = args.weight_init
num_layers = args.num_layers
hidden_size = args.hidden_size
activation = args.activation


hidden_layers = [hidden_size for layer in range(num_layers)]
if activation == 'ReLU':
    activation = 'relu'
if weight_init == 'Xavier':
    weight_init = 'xavier'
if optimizer == 'nag':
    optimizer = 'nesterov'
if loss == 'mean_squared_error':
    loss = 'squared-error'
elif loss == 'cross_entropy':
    loss = 'cross-entropy'


# print(hidden_layers, type(hidden_layers))
# print(epochs, type(epochs), wandb_project, type(hidden_size),activation)
# wandb.init(project="Testing", name="Assignment 1 Question 2,3")

# Load the dataset
if dataset=='fashion_mnist':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
elif dataset=="mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)

# print(f"Training data shape: {x_train.shape}, Training label shape: {y_train.shape}")
# print(f"Validation data shape: {x_val.shape}, Validation label shape: {y_val.shape}")

# Reshape the input data for training, validation, and testing sets
X_train = np.reshape(x_train, (x_train.shape[0], -1)).T
X_val = np.reshape(x_val, (x_val.shape[0], -1)).T
X_test = np.reshape(x_test, (x_test.shape[0], -1)).T

# Normalize the input data to have values between 0 and 1
X_train = X_train / 255.
X_val = X_val / 255.
X_test = X_test / 255.

# Convert the target labels into one-hot encoded vectors
Y_train = np.eye(np.max(y_train) + 1)[y_train].T
Y_val = np.eye(np.max(y_val) + 1)[y_val].T
Y_test = np.eye(np.max(y_test) + 1)[y_test].T

print(f"Training data shape: {X_train.shape}, Training label shape: {Y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation label shape: {Y_val.shape}")
print(f"Testing data shape: {X_test.shape}, Testing label shape: {Y_test.shape}")

class FeedForwardNN:
    def __init__(self,config=None,epochs=epochs,hidden_layers=hidden_layers,weight_decay=weight_decay,learning_rate=learning_rate,optimizer=optimizer,batch_size=batch_size,weight_initialization=weight_init,activations=activation,loss_function=loss,output_function='softmax',gamma=momentum,beta=beta,beta1=beta1,beta2=beta2,eps=epsilon):

        # Constructor that initializes the neural network
        # if config is not None:
        #     # If a config dictionary is passed, use its values to initialize the parameters
        #     self.epochs = config["epochs"]
        #     self.learning_rate = config["learning_rate"]
        #     self.weight_decay = config["weight_decay"]
        #     self.optimizer = config["optimizer"]
        #     self.batch_size = config["batch_size"]
        #     self.weight_initialization = config["weight_initialization"]
        #     self.activations = config["activations"]
        #     self.hidden_layers = [config["hidden_layers_size"] for x in range(config["no_hidden_layers"])]
        # else:
        # If no config dictionary is passed, use the default values to initialize the parameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.weight_initialization = weight_initialization
        self.activations = activations
        self.hidden_layers = hidden_layers

        # Set the remaining parameters for the neural network
        self.loss_function = loss_function
        self.output_function = output_function
        self.gamma = gamma
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.run_name = "lr_{}_ac_{}_in_{}_op_{}_bs_{}_ep_{}_nn_{}".format(self.learning_rate, self.activations, self.weight_initialization, self.optimizer, self.batch_size, self.epochs, self.hidden_layers)
        # Initialize the neural network
        self.initialize()

    def initialize(self):
        # Set the number of neurons in each layer of the neural network
        layers = self.hidden_layers + [Y_train.shape[0]]

        # Initialize the weights and biases for each layer of the neural network
        self.theta = self.initialize_parameters(X_train.shape[0],layers,self.weight_initialization)

        # Calculate the regularization parameter
        self.lambd = self.weight_decay/self.learning_rate

        # Set the number of layers in the neural network
        self.L = len(layers)



    def sigmoid(self,x):
        return (1.0 / (1.0 + np.exp(-x)) )
    
    def tanh(self,x):
        return np.tanh(x)

    def relu(self,x):
        return np.maximum(0, x)

    def d_sigmoid(self,x):
        return (self.sigmoid(x)*(1.0 - self.sigmoid(x)))
    
    def d_tanh(self,x):
        return 1.0 - (self.tanh(x)**2)

    def d_relu(self,x):
        return np.greater(x, 0).astype(int)

    def activation(self,x,activation_function='sigmoid'):
        if activation_function == 'sigmoid':
              return self.sigmoid(x)
        elif activation_function == 'tanh':
              return self.tanh(x)
        elif activation_function == 'relu':
              return self.relu(x)

    def d_activation(self,x,activation_function='sigmoid'):
        if activation_function == 'sigmoid':
              return self.d_sigmoid(x)
        elif activation_function == 'tanh':
              return self.d_tanh(x)
        elif activation_function=='relu':
              return self.d_relu(x)


    def softmax(self,x):
        e = np.exp(x)
        return e / np.sum(e,axis=0)

    def output(self, x, output_function='softmax'):
        if output_function == 'softmax':
            return self.softmax(x)

    def cross_entropy_error(self, Y, inputs):
        Y_hat = inputs[1][-1]
        return -1 * np.sum(Y * (np.log(Y_hat)))
    
    def squared_error(self, Y, inputs):
        Y_hat = inputs[1][-1]
        return (1 / 2) * np.sum((Y_hat - Y) ** 2)

    def squared_error_val(self, Y, inputs):
        Y_hat = inputs[1][-1]
        W, B = self.theta
        m = Y.shape[1]
        return (1 / m) * np.sum((Y_hat - Y) ** 2) + \
               (self.lambd / (2 * m)) * (self.frobenius(W ** 2) + self.frobenius(B ** 2))

    def error(self, Y, inputs, loss_function='cross-entropy'):
        if loss_function == 'cross-entropy':
            return self.cross_entropy_error(Y, inputs)
        elif loss_function == 'squared-error':
            return self.squared_error(Y, inputs) 

    def cross_entropy_error_val(self, Y, inputs):
        W, B = self.theta
        Y_hat = inputs[1][-1]
        m = Y.shape[1]
        return (-1/m) * np.sum(Y * (np.log(Y_hat))) + (self.lambd/(2*m)) * (self.frobenius(W ** 2) + self.frobenius(B ** 2))

    def val_error(self, Y, inputs, loss_function='cross-entropy'):
        if loss_function == 'cross-entropy':
            return self.cross_entropy_error_val(Y, inputs)
        if loss_function == 'squared-error':
            return self.squared_error_val(Y, inputs)

#     def frobenius(self, M):
#         return np.sqrt(np.sum(np.square(M)))

    def initialize_params_random(self, n, layers):
        L = len(layers)
        biases = [np.float128(np.zeros((layers[i], 1))) for i in range(L)]
        weights = [np.float128(np.random.randn(layers[i], n) if i == 0 else np.random.randn(layers[i], layers[i - 1])) for i in range(L)]
        return (np.array(weights), np.array(biases))

    def initialize_params_xavier(self, n, layers):
        
        L=len(layers)
        biases = []
        weights = []
        for i in range(L):
            bias=np.float128(np.zeros((layers[i], 1)))
            if i==0:
                 weight=np.float128(np.random.randn(layers[i],n))
            else:
                 weight = np.float128(np.random.randn(layers[i],layers[i - 1]) * np.sqrt(1 / layers[i - 1]))
            biases.append(bias)
            weights.append(weight)

        return (np.array(weights),np.array(biases))


    def initialize_parameters(self,n,layers,param_init_type):
        if param_init_type == 'random':
            return self.initialize_params_random(n,layers)
        elif param_init_type == 'xavier':
            return self.initialize_params_xavier(n,layers)

    def frobenius(self,X):
        s=0
        for x in X:
          s += np.sum(x)
        return s


    def feedforward(self,X,theta,L):
        H = X
        weights ,biases = theta
        activations = []
        pre_activations = []
        for k in range(L-1):
              A = biases[k] + (weights[k] @ H)
              H = self.activation(A,self.activations)
              pre_activations.append(A)
              activations.append(H)
        
        AL = biases[L-1] + (weights[L-1] @ H)
        Y_hat = self.output(AL,self.output_function)
        pre_activations.append(AL)
        activations.append(Y_hat)
        return (np.array(pre_activations),np.array(activations))

    def backprop(self,X,Y,inputs,theta,batch_size,L):
        # Initialize empty lists for storing gradients
        d_biases, d_weights = [], []
        d_biases2 = []
        d_weights2 = []
        # Extract pre-activations and activations from the inputs
        pre_activations, activations = inputs
        # Get the predicted output
        Y_hat = activations[-1]


        if self.loss_function == 'squared-error':
          d_AL = Y_hat*(Y_hat - Y)*(1 - Y_hat)
        elif self.loss_function == 'cross-entropy':
          d_AL = Y_hat - Y
        # Loop over the layers in reverse order to calculate the gradients
        for k in range(L-1, -1, -1):
            # Calculate the gradients for the weights and biases
            d_W = (1/batch_size)*(d_AL @ activations[k-1].T) if k > 0 else (1/batch_size)*(d_AL @ X.T)
            d_W2 = (1 / batch_size) * (d_AL ** 2 @ (activations[k-1].T) ** 2) if k>0 else (1 / batch_size) * (d_AL ** 2 @ (X.T) ** 2)
            d_B = (1/batch_size)*np.sum(d_AL, axis=1, keepdims=True)
            d_B2 = (1 / batch_size) * np.sum(d_AL ** 2, axis=1, keepdims=True)

            # Calculate the derivative of the activation function and backpropagate the error to the previous layer
            if k > 0:
                d_AL = (theta[0][k].T @ d_AL) * self.d_activation(pre_activations[k-1], self.activations)
            # Add the gradients to the lists
            d_weights.insert(0, d_W)
            d_biases.insert(0, d_B)
            d_weights2.insert(0, d_W2)
            d_biases2.insert(0, d_B2)
        d_theta = (np.array(d_weights),np.array(d_biases))
        d_theta2 = (np.array(d_weights2), np.array(d_biases2))
        
        return (d_theta, d_theta2)

    # Function to update weights and biases based on the calculated gradients and learning rate
    def update_params(self, theta, d_theta, learning_rate):
        weights, biases = theta
        d_weights, d_biases = d_theta
        updated_weights = (1 - self.weight_decay)*weights - learning_rate*d_weights
        updated_biases = (1 - self.weight_decay)*biases - learning_rate*d_biases
        return updated_weights, updated_biases

    # Function to perform mini-batch gradient descent on the given data
    def sgd(self, X, Y, theta, learning_rate, batch_size, L):
        m = X.shape[1]
        total_error = 0
        # loop over mini-batches
        for i in range(0, m, batch_size):
            start = i
            stop = i + batch_size
            inputs = self.feedforward(X[:, start:stop], theta, L) # compute output of the network
            W, B = theta
            # compute L2 regularization term
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            # compute error
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization
            # compute gradients
            d_theta, _ = self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, batch_size, L) 
            # update weights and biases                                                                    
            theta = self.update_params(theta, d_theta, learning_rate)
        # handle the last mini-batch if it is not a multiple of batch_size
        if m % batch_size != 0:
            start = m - m % batch_size
            stop = m
#             start = i*batch_size
            inputs = self.feedforward(X[:, start:stop], theta, L) # compute output of the network
            d_theta, _ = self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, m % batch_size, L) # compute gradients
            theta = self.update_params(theta, d_theta, learning_rate) # update weights and biases
            W, B = theta

            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization 
            
        # Calculate the average error
        avg_err = total_error/m
        # Return the updated theta and average error
        return (theta, avg_err)
    
    def update_params_momentum(self, theta, d_theta, learning_rate, gamma, prev_weight, prev_bias):
        weights, biases = theta
        d_weights,d_biases = d_theta
        
        # Calculate the velocity for weights and biases
        v_weight = gamma * prev_weight + learning_rate * d_weights
        v_bias = gamma * prev_bias + learning_rate * d_biases

        # Apply weight decay to the weights
        decay = (1 - self.weight_decay)

        # Update weights and biases using the velocity and decayed weights
        updated_weights = decay*weights - v_weight
        updated_biases = decay*biases - v_bias

        return (updated_weights, updated_biases, v_weight, v_bias)

    def gd_momentum(self, X, Y, theta, learning_rate, batch_size, gamma, L):
        m = X.shape[1] # number of training examples
        prev_weights = 0 # initialize previous weights to zero
        prev_biases = 0 # initialize previous biases to zero
        total_error = 0 # initialize total error to zero

        # loop over mini-batches
        for i in range(0, m, batch_size):
            start = i
            stop = i + batch_size
            inputs = self.feedforward(X[:, start:stop], theta, L) # compute output of the network
            W, B = theta
            # compute L2 regularization term
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            # compute error
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization
            # compute gradients
            d_theta, _ = self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, batch_size, L) 
            # update weights and biases using momentum                                                                     
            weights, biases, prev_weights, prev_biases = self.update_params_momentum(theta, d_theta, learning_rate, gamma, prev_weights, prev_biases)
            theta = weights, biases
        # handle the last mini-batch if it is not a multiple of batch_size
        if m % batch_size != 0:
            start = m - m % batch_size
            stop = m
#             start = i*batch_size
            inputs = self.feedforward(X[:, start:stop], theta, L) # compute output of the network
            d_theta = self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, m % batch_size, L) # compute gradients
        
            weights, biases, prev_weights, prev_biases = self.update_params_momentum(theta, d_theta, learning_rate, gamma, prev_weights, prev_biases) # update weights and biases using momentum
            regularization = (self.lambd / 2) * (self.frobenius(weights**2) + self.frobenius(biases**2) )
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization 
            
            

            theta = weights, biases
        
        total_error /= m # average total error across all mini-batches
        return (theta, total_error) # return updated weights and biases and the total error

    def update_params_nesterov(self, theta, d_theta, learning_rate, gamma, prev_weight, prev_bias):
        weights, biases = theta
        d_weights,d_biases = d_theta
        
        # Calculate the velocity for weights and biases
        v_weight = gamma * prev_weight + learning_rate * d_weights
        v_bias = gamma * prev_bias + learning_rate * d_biases

        # Apply weight decay to the weights
        decay = (1 - self.weight_decay)

        # Update weights and biases using the velocity and decayed weights
        updated_weights = decay*weights - v_weight
        updated_biases = decay*biases - v_bias

        return (updated_weights, updated_biases, v_weight, v_bias)

    def gd_nesterov(self, X, Y, theta, learning_rate, batch_size, gamma, L):
        m = X.shape[1] # number of training examples
        prev_weights = 0 # initialize previous weights to zero
        prev_biases = 0 # initialize previous biases to zero
        total_error = 0 # initialize total error to zero

        weights, biases = theta
        # loop over mini-batches
        for i in range(0, m, batch_size):
            start = i
            stop = i + batch_size
            # compute output of the network
            inputs = self.feedforward(X[:, start:stop], theta, L) 
            W, B = theta
            # compute L2 regularization term
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            # compute error
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization
            
            # Compute gradients using backpropagation
            v_weight=gamma*prev_weights
            v_biases=gamma*prev_biases
            theta2=weights-v_weight,biases-v_biases
            d_theta, _ = self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta2, batch_size, L) 
            
            # update weights and biases using momentum                                                                     
            weights, biases, prev_weights, prev_biases = self.update_params_nesterov(theta, d_theta, learning_rate, gamma, prev_weights, prev_biases)
            theta = weights, biases
        # handle the last mini-batch if it is not a multiple of batch_size
        if m % batch_size != 0:
            start = m - m % batch_size
            stop = m
            inputs = self.feedforward(X[:, start:stop], theta, L) # compute output of the network
            
            # Compute gradients using backpropagation
            v_weight=gamma*prev_weights
            v_biases=gamma*prev_biases
            theta2=weights-v_weight,biases-v_biases
            d_theta, _= self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta2, m % batch_size, L) # compute gradients
        
            weights, biases, prev_weights, prev_biases = self.update_params_nesterov(theta, d_theta, learning_rate, gamma, prev_weights, prev_biases) # update weights and biases using momentum
            regularization = (self.lambd / 2) * (self.frobenius(weights**2) + self.frobenius(biases**2) )
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization 

            theta = weights, biases
        
        total_error /= m # average total error across all mini-batches
        return (theta, total_error) # return updated weights and biases and the total error
    
    def update_params_rmsprop(self, theta, d_theta, d_theta2, prev_weights2, prev_biases2, learning_rate, beta, eps):
        weights, biases = theta
        d_weights, d_biases = d_theta
        d_weights2, d_biases2 = d_theta2

        # Compute the exponential moving averages of squared gradients
        prev_weights2 = beta * prev_weights2 + (1 - beta) * ((d_weights)**2)
        prev_biases2 = beta * prev_biases2 + (1 - beta) * ((d_biases)**2)

        # Compute the RMSProp update
        W_ = learning_rate / ((prev_weights2)**0.5 + eps)
        B_ = learning_rate / ((prev_biases2)**0.5 + eps)

        # Update the parameters
        # Apply weight decay to the weights
        decay = (1 - self.weight_decay)
        # Update weights and biases using the velocity and decayed weights
        updated_weights = decay * weights - W_ * d_weights
        updated_biases = decay * biases - B_ * d_biases

        return ((np.array(updated_weights), np.array(updated_biases)), prev_weights2, prev_biases2)

    def rmsprop(self, X, Y, theta, learning_rate, beta, eps, batch_size, L):
        m = X.shape[1] # number of training examples
        prev_weights2 = 0 # initialize previous weights to zero
        prev_biases2 = 0 # initialize previous biases to zero
        total_error = 0 # initialize total error to zero

        # loop over mini-batches
        for i in range(0, m, batch_size):
            start = i
            stop = i + batch_size
            # compute output of the network
            inputs = self.feedforward(X[:, start:stop], theta, L) 
            W, B = theta
            # compute L2 regularization term
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            # compute error
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization
            
            # Compute gradients using backpropagation
            d_theta, d_theta2 = self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, batch_size, L) 
            
            # update weights and biases using RMSProp                                                                    
            theta, prev_weights, prev_biases = self.update_params_rmsprop(theta, d_theta, d_theta2, prev_weights2, prev_biases2, learning_rate, beta, eps)

        # handle the last mini-batch if it is not a multiple of batch_size
        if m % batch_size != 0:
            start = m - m % batch_size
            stop = m
            inputs = self.feedforward(X[:, start:stop], theta, L) # compute output of the network
            
            # Compute gradients using backpropagation
            d_theta, d_theta2= self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, batch_size, L) # compute gradients
        
            theta, prev_weights2, prev_biases2 = self.update_params_rmsprop(theta, d_theta, d_theta2, prev_weights2, prev_biases2, learning_rate, beta, eps)
            W, B = theta
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization 

        
        total_error /= m # average total error across all mini-batches
        return (theta, total_error) # return updated weights and biases and the total error
  

    def update_params_adam(self,theta,d_theta,d_theta2,prev_weights,prev_bias,prev_weights2,prev_biases2,learning_rate,beta1,beta2,eps,t):
        weights, biases = theta
        d_weights,d_biases = d_theta
        d_weights2,d_biases2 = d_theta2
        
        # update the exponentially weighted averages of the gradients
        prev_weights = beta1*prev_weights + (1-beta1)*d_weights
        prev_bias = beta1*prev_bias + (1-beta1)*d_biases

         # update the exponentially weighted averages of the squared gradients
        prev_weights2=beta2*prev_weights2 + (1-beta2)*(d_weights2)
        prev_biases2=beta2*prev_biases2 + (1-beta2)*(d_biases2)
    
        # bias correction to the weighted averages of the gradients
        corr_m_w = prev_weights/(1-(beta1**t))
        corr_m_b = prev_bias/(1-(beta1**t))

        # bias correction to the weighted averages of the squared gradients
        corr_v_w = prev_weights2/(1-(beta2**t))
        corr_v_b = prev_biases2/(1-(beta2**t))

        # calculate the update parameters using the bias-corrected averages of the gradients and squared gradients
        corr_v_w = learning_rate/((corr_v_w)**0.5 + eps)
        corr_v_b = learning_rate/((corr_v_b)**0.5 + eps)

        # update the weights and biases using the update parameters and L2 regularization
        weights = (1 - self.weight_decay)*weights - corr_v_w*corr_m_w
        biases = (1 - self.weight_decay)*biases - corr_v_b*corr_m_b

        theta = (np.array(weights),np.array(biases))
        return (theta,prev_weights,prev_bias,prev_weights2,prev_biases2)

    def adam(self,X,Y,theta,learning_rate,beta1,beta2,eps,batch_size,L):
        m = X.shape[1] # number of training examples
        prev_weights, prev_weights2 = 0,0 # initialize previous weights to zero
        prev_biases, prev_biases2 = 0,0 # initialize previous biases to zero
        total_error = 0 # initialize total error to zero

        # loop over mini-batches
        for i in range(0, m, batch_size):
            start = i
            stop = i + batch_size
            # compute output of the network
            inputs = self.feedforward(X[:, start:stop], theta, L) 
            W, B = theta
            # compute L2 regularization term
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            # compute error
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization
            
            # Compute gradients using backpropagation
            d_theta, d_theta2 = self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, batch_size, L) 
            
            # update weights and biases using Adam                                                                    
            theta, prev_weights, prev_biases, prev_weights2, prev_biases2 = self.update_params_adam(theta, d_theta, d_theta2, prev_weights, prev_biases, prev_weights2, prev_biases2, learning_rate, beta1, beta2, eps, i+1)

        # handle the last mini-batch if it is not a multiple of batch_size
        if m % batch_size != 0:
            start = m - m % batch_size
            stop = m
            inputs = self.feedforward(X[:, start:stop], theta, L) # compute output of the network
            
            # Compute gradients using backpropagation
            d_theta, d_theta2= self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, batch_size, L) # compute gradients
        
            theta, prev_weights, prev_biases, prev_weights2, prev_biases2 = self.update_params_adam(theta, d_theta, d_theta2, prev_weights, prev_biases, prev_weights2, prev_biases2, learning_rate, beta1, beta2, eps, i+1)
            W, B = theta
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization 

        
        total_error /= m # average total error across all mini-batches
        return (theta, total_error) # return updated weights and biases and the total error
  

    def update_params_nadam(self,theta,d_theta,d_theta2,prev_weights,prev_bias,prev_weights2,prev_biases2,learning_rate,beta1,beta2,eps,t):
        weights, biases = theta
        d_weights,d_biases = d_theta
        d_weights2,d_biases2 = d_theta2
        
        # update the exponentially weighted averages of the gradients
        prev_weights = beta1*prev_weights + (1-beta1)*d_weights
        prev_bias = beta1*prev_bias + (1-beta1)*d_biases

         # update the exponentially weighted averages of the squared gradients
        prev_weights2=beta2*prev_weights2 + (1-beta2)*(d_weights2)
        prev_biases2=beta2*prev_biases2 + (1-beta2)*(d_biases2)
    
        beta_t = 1-(beta1**t)
        beta2_t = 1-(beta2**t)
        # bias correction to the weighted averages of the gradients
        corr_m_w = beta1*prev_weights/beta_t + ((1-beta1)/beta_t)*d_weights
        corr_m_b = beta1*prev_bias/beta_t + ((1-beta1)/beta_t)*d_biases

        # bias correction to the weighted averages of the squared gradients
        corr_v_w = prev_weights2/beta2_t
        corr_v_b = prev_biases2/beta2_t

        # calculate the update parameters using the bias-corrected averages of the gradients and squared gradients
        corr_v_w = learning_rate/((corr_v_w)**0.5 + eps)
        corr_v_b = learning_rate/((corr_v_b)**0.5 + eps)

        # update the weights and biases using the update parameters and L2 regularization
        weights = (1 - self.weight_decay)*weights - corr_v_w*corr_m_w
        biases = (1 - self.weight_decay)*biases - corr_v_b*corr_m_b

        theta = (np.array(weights),np.array(biases))
        return (theta,prev_weights,prev_bias,prev_weights2,prev_biases2)

    def nadam(self,X,Y,theta,learning_rate,beta1,beta2,eps,batch_size,L):
        m = X.shape[1] # number of training examples
        prev_weights, prev_weights2 = 0,0 # initialize previous weights to zero
        prev_biases, prev_biases2 = 0,0 # initialize previous biases to zero
        total_error = 0 # initialize total error to zero

        weights, biases = theta
        
        # loop over mini-batches
        for i in range(0, m, batch_size):
            start = i
            stop = i + batch_size
            # compute output of the network
            inputs = self.feedforward(X[:, start:stop], theta, L) 
            W, B = theta
            # compute L2 regularization term
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            # compute error
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization
            
            # Compute gradients using backpropagation
            d_theta, d_theta2 = self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, batch_size, L) 
            
            # update weights and biases using Adam                                                                    
            theta, prev_weights, prev_biases, prev_weights2, prev_biases2 = self.update_params_nadam(theta, d_theta, d_theta2, prev_weights, prev_biases, prev_weights2, prev_biases2, learning_rate, beta1, beta2, eps, i+1)

        # handle the last mini-batch if it is not a multiple of batch_size
        if m % batch_size != 0:
            start = m - m % batch_size
            stop = m
            inputs = self.feedforward(X[:, start:stop], theta, L) # compute output of the network
            
            # Compute gradients using backpropagation
            d_theta, d_theta2= self.backprop(X[:, start:stop], Y[:, start:stop], inputs, theta, batch_size, L) # compute gradients
        
            theta, prev_weights, prev_biases, prev_weights2, prev_biases2 = self.update_params_nadam(theta, d_theta, d_theta2, prev_weights, prev_biases, prev_weights2, prev_biases2, learning_rate, beta1, beta2, eps, i+1)
            W, B = theta
            regularization = (self.lambd / 2) * (self.frobenius(W**2) + self.frobenius(B**2) )
            total_error += self.error(Y[:, start:stop], inputs, self.loss_function) + regularization 

        
        total_error /= m # average total error across all mini-batches
        return (theta, total_error) # return updated weights and biases and the total error
  
    # Function to perform optimization based on the specified optimizer
    def optimizations(self, theta, L):
        # If optimizer is stochastic gradient descent
        if self.optimizer == 'sgd':
            # Perform mini-batch gradient descent on the training data
            return self.sgd(X_train, Y_train, theta, self.learning_rate, 1, L)
        elif self.optimizer == 'momentum':
            return self.gd_momentum(X_train,Y_train,theta,self.learning_rate,self.batch_size,self.gamma,L)
        elif self.optimizer == 'nesterov':
            return self.gd_nesterov(X_train,Y_train,theta,self.learning_rate,self.batch_size,self.gamma,L)
        elif self.optimizer == 'rmsprop':
            return self.rmsprop(X_train,Y_train,theta,self.learning_rate,self.beta,self.eps,self.batch_size,L)
        elif self.optimizer == 'adam':
            return self.adam(X_train,Y_train,theta,self.learning_rate,self.beta1,self.beta2,self.eps,self.batch_size,L)
        elif self.optimizer == 'nadam':
            return self.nadam(X_train,Y_train,theta,self.learning_rate,self.beta1,self.beta2,self.eps,self.batch_size,L)
      


    def fit(self):
        # perform optimization on the model's parameters (theta) and get train loss
        self.theta, train_loss = self.optimizations(self.theta, self.L)

        # make predictions on the training set
        outputs_train = self.feedforward(X_train, self.theta, self.L)
        Y_pred_train = np.argmax(outputs_train[1][-1], axis=0)
        Y_true_train = np.argmax(Y_train, axis=0)

        # calculate training accuracy
        train_acc = accuracy_score(Y_true_train, Y_pred_train)

        # make predictions on the validation set
        outputs_val = self.feedforward(X_val, self.theta, self.L)

        # calculate validation loss
        val_loss = self.val_error(Y_val, outputs_val, self.loss_function)
        Y_pred_val = np.argmax(outputs_val[1][-1], axis=0)
        Y_true_val = np.argmax(Y_val, axis=0)

        # calculate validation accuracy
        val_acc = accuracy_score(Y_true_val, Y_pred_val)

        # return training and validation accuracies and losses
        return train_acc, train_loss, val_acc, val_loss
    
    def fit_test(self):
        # perform optimization on the model's parameters (theta) and get train loss
        self.theta, train_loss = self.optimizations(self.theta, self.L)

        # make predictions on the training set
        outputs_train = self.feedforward(X_train, self.theta, self.L)
        Y_pred_train = np.argmax(outputs_train[1][-1], axis=0)
        Y_true_train = np.argmax(Y_train, axis=0)

        # calculate training accuracy
        train_acc = accuracy_score(Y_true_train, Y_pred_train)

        # make predictions on the test set
        outputs_val = self.feedforward(X_test, self.theta, self.L)

        # calculate test loss
        test_loss = self.val_error(Y_test, outputs_val, self.loss_function)
        Y_pred_val = np.argmax(outputs_val[1][-1], axis=0)
        Y_true_val = np.argmax(Y_test, axis=0)

        # calculate validation accuracy
        test_acc = accuracy_score(Y_true_val, Y_pred_val)

        # return training and validation accuracies and losses
        return train_acc, train_loss, test_acc, test_loss
       
    def predict(self, X_test):
        # get the number of hidden layers
        L = len(self.hidden_layers) + 1

        # make predictions on the test set
        outputs = self.feedforward(X_test, self.theta, L)
        Y_pred = np.argmax(outputs[1][-1], axis=0)

        # return predicted labels
        return Y_pred


sweep_config_train = {
  "name" : "cs6910_assignment1_fashion-mnist_sweep",
  "method" : "bayes",
  "metric" : {
      "name" : "validation_accuracy",
      "goal" : "maximize"
  },
  "parameters" : {
    "epochs" : {
      "values" : [epochs]
    },
    "learning_rate" :{
      "values" : [learning_rate]
    },
    "no_hidden_layers":{
        "values" : [num_layers]
    },
    "hidden_layers_size":{
        "values" : [hidden_size]
    },
    "weight_decay":{
      "values": [weight_decay] 
    },
    "optimizer":{
        "values": [optimizer]
    },
    "batch_size":{
        "values":[batch_size]
    },
    "weight_initialization":{
        "values": [weight_init]
    },
    "activations":{
        "values": [activation]
    }
  }
}

sweep_id_train = wandb.sweep(sweep_config_train,project=wandb_project, entity=wandb_entity)

tuned_models = []
def train():
    with wandb.init() as run:


        config = wandb.config
        model = FeedForwardNN(config=None)
        name = model.run_name
        # print("Hyperparameter Settings: {}".format(name))
        train_acc,train_loss,val_acc,val_loss = 0,0,0,0
        for epoch in range(epochs):
            train_acc,train_loss,val_acc,val_loss = model.fit()  # model training code here
            metrics = {
            "accuracy":train_acc,
                "loss":train_loss,
            "validation_accuracy": val_acc,
            "validation_loss": val_loss,
                "epochs":epoch
                }
            print({
            "epochs": epoch,
            "accuracy":train_acc,
            "loss":train_loss,
            "validation_accuracy": val_acc,
            "validation_loss": val_loss,
            })
            wandb.log(metrics) 
        tuned_models.append({
            "accuracy":train_acc,
            "loss":train_loss,
            "validation_accuracy": val_acc,
            "validation_loss": val_loss,
            "model": name
        })          
# number of runs to execute
# count = 2

wandb.agent(sweep_id_train, function=train, count=1)
def best_model(tuned_models):
  sorted_list = sorted(tuned_models, key = lambda d : d['validation_accuracy'],reverse=True)
  best_model = sorted_list[0]["model"]
  return (sorted_list, best_model)

(models_sorted, bestmodel) = best_model(tuned_models)
print("Final Scores: \nModel Hyperparameters: {}\nAccuracy: {}\nLoss: {}\nValidation Accuracy: {}\nValidation Loss {}".format(tuned_models[0]['model'], tuned_models[0]['accuracy'], tuned_models[0]['loss'], tuned_models[0]['validation_accuracy'], tuned_models[0]['validation_loss']))