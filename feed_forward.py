import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
tf.keras.layers.serialize
tf.keras.utils.CustomObjectScope
tf.keras.utils.register_keras_serializable



# Load the training and testing data MNIST
# Import dataset & split into train and test data
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
# Length of the training dataset
len(x_train)
len(y_train)
60000
# Length of the testing dataset
len(x_test)
len(y_test)
10000
# Shape of the training dataset
x_train.shape
(60000, 28, 28)
# Shape of the testing dataset
x_test.shape
(10000, 28, 28)
# See first Image Matrix
x_train[0]



plt.matshow(x_train[0])



# Normalize the iamges by scaling pixel intensities to the range 0,1
x_train=x_train/255
x_test=x_test/255
# See first Naormalize Image Matrix
x_train[0]



# Define the network architecture using Keras
model=keras.Sequential([
# Input Layer
keras.layers.Flatten(input_shape = (28,28)),
# Hidden Layer
keras.layers.Dense(128,activation ='relu'),
# Output Layer
keras.layers.Dense(10,activation = 'softmax')
])
model.summary()



# Compile the Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#Train the model using SGD
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10)



test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)



# Making Prediction on New Data
n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()



predicted_value=model.predict(x_test)
print("Handwritten number is = %d" %np.argmax(predicted_value[n]))



# Plot the training loss and accuracy
history.history.keys()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss & Accuracy')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
plt.show()  