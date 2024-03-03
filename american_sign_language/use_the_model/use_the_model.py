import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image as image_utils

model = keras.models.load_model("../improve_by_augmentation/sign_language_model_augmentation.h5")


# model.summary()


# Show the images
def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.show()


# show_image("../data/asl_images/a.png")


# Scale the image
def load_and_scale_image(image_path):
    img = image_utils.load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    plt.imshow(img)
    plt.show()
    return img

alphabet = "abcdefghiklmnopqrstuvwxy"   # j and z are not included


def predict_letter(file_path):
    # Show the image
    show_image(file_path)
    # Load and scale the image
    image = load_and_scale_image(file_path)
    # Reshape the image
    image = image_utils.img_to_array(image)
    # Normalize the image
    image = image / 255
    # Make a prediction
    prediction = model.predict(image.reshape(1, 28, 28, 1))
    # Convert the prediction to a letter
    letter = alphabet[np.argmax(prediction)]
    return letter

print(predict_letter("../data/asl_images/u.jpeg"))
