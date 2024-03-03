import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout

# Load the data
train_df = pd.read_csv("../data/sign_mnist_train.csv")
valid_df = pd.read_csv("../data/sign_mnist_valid.csv")

# Separate the labels
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

# Separate out the image vectors
x_train = train_df.values
x_valid = valid_df.values

# Normalize the data
x_train = x_train / 255
x_valid = x_valid / 255

# Category the labels
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Reshape the data
x_train = x_train.reshape(-1, 28, 28, 1)
x_valid = x_valid.reshape(-1, 28, 28, 1)

# Build the model
model = Sequential()
model.add(Conv2D(75, kernel_size=(3, 3), strides=1, padding="same", activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(50, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(25, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation='softmax'))

# Summary of the model
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10, verbose=1)


