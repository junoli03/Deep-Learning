import tensorflow.keras as keras
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()


# show the first image in the training dataset

image = x_train[0]
plt.imshow(image, cmap='gray')
plt.show()

# flatten the images
x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)

# normalize the images

x_train = x_train / 255
x_valid = x_valid / 255

# convert the labels to one-hot encoded

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Create the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid))
