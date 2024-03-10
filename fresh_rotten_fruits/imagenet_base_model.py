# Description: This file contains the code to load the pre-trained model from the ImageNet dataset and use it to
# recognize fresh and rotten fruits.

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load the pre-trained model
base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False
)

# Freeze the base model
base_model.trainable = False

# Add layers to model
input = keras.Input(shape=(224, 224, 3))
x = base_model(input, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(6, activation='softmax')(x)
model = keras.Model(input, outputs)

# Summary of the model
# model.summary()

# Compile the model
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# Augment the data
datagen_train = ImageDataGenerator(
    samplewise_center=True,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)
datagen_valid = ImageDataGenerator(samplewise_center=True)

# Load and iterate the data
train_it = datagen_train.flow_from_directory(
    'fruits/train',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
)

valid_it = datagen_valid.flow_from_directory(
    'fruits/valid',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
)

# Train the model
model.fit(train_it, validation_data=valid_it, steps_per_epoch=train_it.samples/train_it.batch_size, validation_steps=valid_it.samples/valid_it.batch_size, epochs=10)

# Unfreeze the model for fine-tuning
base_model.trainable = True
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()],
    optimizer=keras.optimizers.RMSprop(learning_rate=0.00001)
)

model.fit(train_it, validation_data=valid_it, steps_per_epoch=train_it.samples/train_it.batch_size, validation_steps=valid_it.samples/valid_it.batch_size, epochs=10)


# Save the model
# model.save('fruits_model.h5')

# Evaluate the model
# model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)