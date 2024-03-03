import pandas as pd

train_df = pd.read_csv('data/sign_mnist_train.csv')
valid_df = pd.read_csv('data/sign_mnist_valid.csv')

# Separate the labels from the images
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

# Separate the images
x_train = train_df.values
x_valid = valid_df.values

# Turn the labels into one-hot encoded
num_classes = 24

