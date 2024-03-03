import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras

model = keras.models.load_model("../improve_by_augmentation/sign_language_model_augmentation.h5")

model.summary()


# Show the images
def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.show()

show_image("../data/american_sign_language.png")
