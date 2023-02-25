import wandb
# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from keras.datasets import fashion_mnist


wandb.init(project="CS6910 Assignment 1", name="Assignment 1 Question 1")

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(x, y), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Number of features in the dataset
num_features = 784

# Number of classes
num_classes = len(set(y_train))

# sample_indices contain index of first occurence image of each class
sample_indices = [list(y_train).index(i) for i in range(num_classes)]

sample_images = []
sample_labels = []

for i in range(len(classes)):
    sample_images.append(x_train[sample_indices[i]])
    sample_labels.append(classes[i])

wandb.log({"Sample Image from each class": [wandb.Image(image, caption=label) for image, label in zip(sample_images, sample_labels)]})

