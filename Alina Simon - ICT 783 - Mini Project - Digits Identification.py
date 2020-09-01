# -*- coding: utf-8 -*-
"""
@author: alina.simoncuevas
"""
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#import matplotlib.pyplot as plt
#plt.imshow(x_train[0], cmap='gray')

x_train = x_train.reshape(60000, 28, 28, 1)/255
x_test = x_test.reshape(10000, 28, 28, 1)/255

from keras.utils import to_categorical;

from keras.models import Sequential
model = Sequential()

from keras.layers import Conv2D
model.add(Conv2D(23, (3, 3), input_shape=(32, 32, 3), activation='relu'))

model.add(Conv2D(32, (3, 3), activation='relu'))

from keras.layers import MaxPooling2D
model.add(MaxPooling2D(pool_size=(2, 2)))

### dropout layer ???

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

### dropout layer ???

from keras.layers import Flatten
model.add(Flatten())

from keras.layers import Dense
model.add(Dense(units=512, activation='relu'))

### dropout layer ???

model.add(Dense(units=100, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs = 100, batch_size = 32)

model.fit(x_train, y_train)

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)