# This model can only lets dogs in and other animals out, except for the cats. And use the pre-trained model ImageNet.

# Load the pre-trained model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

# Load the VGG16 network on the ImageNet dataset
model = VGG16(weights='imagenet')


# Summary of the model
# base_model.summary()

# Loading an image
def show_image(image_path):
    img = mpimg.imread(image_path)
    print(img.shape)
    plt.imshow(img)
    plt.show()


# show_image("doggy_door_images/happy_dog.jpg")

# Preprocessing the Image

def load_and_process_image(image_path):
    # Print the image's original shape
    print('Original image shape:', mpimg.imread(image_path).shape)
    # Load the image
    img = image_utils.load_img(image_path, target_size=(224, 224))
    # Convert the image to an array
    img = image_utils.img_to_array(img)
    # Add a dimension
    img = img.reshape(1, 224, 224, 3)
    # Preprocess the image
    img = preprocess_input(img)
    # Print the image's shape after processing
    print('Processed image shape:', img.shape)
    return img


# processed_image = load_and_process_image("doggy_door_images/brown_bear.jpg")

# Predicting the Image
def predict_image(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process the image
    img = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(img)
    # Print predictions
    print('Predicted:', decode_predictions(predictions, top=3))


# predict_image("doggy_door_images/happy_dog.jpg")
# predict_image("doggy_door_images/brown_bear.jpg")
# predict_image("doggy_door_images/sleepy_cat.jpg")

# Only dogs are allowed in the house, so we need to check if the image contains a dog.
def doggy_door(image_path):
    # Load and pre-process the image
    img = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(img)
    # Check if the image contains a dog
    if 151 <= np.argmax(predictions) <= 268:
        print('Doggy come in!')
    elif 281 <= np.argmax(predictions) <= 285:
        print("Kitty stay inside!")
    else:
        print('You are not allowed in!')


doggy_door("doggy_door_images/happy_dog.jpg")
doggy_door("doggy_door_images/brown_bear.jpg")
doggy_door("doggy_door_images/sleepy_cat.jpg")
