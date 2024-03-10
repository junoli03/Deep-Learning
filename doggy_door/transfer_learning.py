# Use Pre-trained ImageNet model to classify dog, let doggy in if it is a Golden Retriever

# Download the pre-trained model
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)

# Summary of the model
# base_model.summary()

# Freeze the base model
base_model.trainable = False

# Add new layers
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# Summary of the model
# model.summary()

# Compile the model
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

# Augment the data
datagen_train = ImageDataGenerator(
    samplewise_center=True,
    rotation_range=15,
    zoom_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)

datagen_valid = ImageDataGenerator(samplewise_center=True)

# Load the data
train_data = datagen_train.flow_from_directory('presidential_doggy_door/train', target_size=(224, 224), batch_size=8, class_mode='binary', color_mode='rgb')
valid_data = datagen_valid.flow_from_directory('presidential_doggy_door/valid', target_size=(224, 224), batch_size=8, class_mode='binary', color_mode='rgb')

# Train the model
model.fit(train_data, steps_per_epoch=12, validation_data=valid_data, validation_steps=4, epochs=20)