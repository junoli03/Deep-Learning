import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


train_df = pd.read_csv("data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/sign_mnist_valid.csv")

# Explore the data
print(train_df.head())

y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values

# Visualize the data
# plt.figure(figsize=(40, 40))
#
# num_images = 20
# for i in range(num_images):
#     row = x_train[i]
#     label = y_train[i]
#
#     image = row.reshape(28, 28)
#     plt.subplot(1, num_images, i + 1)
#     plt.title(label, fontdict={'fontsize': 30})
#     plt.axis('off')
#     plt.imshow(image, cmap='gray')
#     plt.show()

# Normalize the data
x_train = x_train / 255
x_valid = x_valid / 255

# Category the labels
num_classes = 26
# avoid running multiple times
if not y_train.shape[-1] == num_classes:
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Summary of the model
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))


