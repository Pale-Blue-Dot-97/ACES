"""Script run simple CNN on Voyager 2 magnetometer data for binary classification

TODO:
    * Construct model

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
from PIL import Image
import numpy as np
import glob
import random
import os


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_images(path, n_images):
    """Loads in images and their names from file

    Args:
        path (str): Path to folder containing images

    Returns:
        images ([[float]]): 3D array of all the 2D images
        names ([str]): List of names of each image without path and .png

    """

    images = []
    names = []

    i = 0
    while i < n_images:
        for name in os.listdir(path):
            # Normalize pixel values to be between 0 and 1
            images.append([image.imread(fname=(path + name), format='PNG') / 255.0])
            names.append(name.replace('Blocks\\', '').replace('.png', ''))
        i += 1

    return images, names


def load_labels():
    """

    Returns:
        labels (DataFrame):

    """

    labels = pd.read_csv('VOY2_JE_LABELS.csv', names=('NAME', 'LABEL'), dtype=str, header=0)

    def bool_to_binary(label):
        if label == 'True':
            return [0, 1]
        if label == 'False':
            return [1, 0]

    labels['LABEL'] = labels['LABEL'].apply(bool_to_binary)

    return labels


def split_data(data, labels, n):
    """Splits data into training and testing data

    Args:
        data (DataFrame): Table of images with filenames
        labels (DataFrame): Table of labels for images with filenames
        n (int): Number of training images desired

    Returns:
        train_images ([[[float]]]): All training images
        test_images ([[[float]]]): All testing images
        train_labels ([[int]]): All accompanying training labels
        test_labels ([[int]]): All accompanying testing labels

    """
    # Fixes seed number so results are replicable
    random.seed(42)

    names = data['NAME']

    train_names = []

    # Randomly selects the desired number of training images
    for i in range(n):
        train_names.append(random.choice(names))

    # Takes the difference of lists to find remaining names must be for testing
    test_names = list(set(names).difference(set(train_names)))

    # Uses these to find those names in data to make cut
    train_images = np.array(data.loc[data['NAME'].isin(train_names)]['IMAGE'].tolist())
    test_images = np.array(data.loc[data['NAME'].isin(test_names)]['IMAGE'].tolist())

    train_labels = np.array(labels.loc[labels['NAME'].isin(train_names)]['LABEL'].tolist())
    test_labels = np.array(labels.loc[labels['NAME'].isin(test_names)]['LABEL'].tolist())

    return train_images, test_images, train_labels, test_labels


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():

    print('***************************** CNN1 ********************************************')

    print('\nLOAD IMAGES')
    # Load in images
    images, names = load_images('Blocks/', 5000)

    # Construct DataFrame matching images to their names
    data = pd.DataFrame()
    data['NAME'] = names
    data['IMAGE'] = images

    # Deletes variables no longer needed from memory
    del names, images

    print('\nLOAD LABELS')
    # Load in accompanying labels into separate randomly ordered DataFrame
    labels = load_labels()

    print('\nSPLIT DATA INTO TRAIN AND TEST')
    # Split images into test and train
    train_images, test_images, train_labels, test_labels = split_data(data, labels, 1000)

    # Deletes variables no longer neeeded
    del data, labels

    print(train_images.shape)

    print(train_images[0])
    print(train_images[0][0])

    print('\nBEGIN MODEL CONSTRUCTION')

    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=9, activation='relu', input_shape=(4, 4096),
                            data_format='channels_first'))
    print('Input shape:')
    print(model.input_shape)
    print('Output shape:')
    print(model.output_shape)

    model.add(layers.MaxPooling1D(2))
    print('Input shape:')
    print(model.input_shape)
    print('Output shape:')
    print(model.output_shape)

    model.add(layers.Conv1D(64, 9, activation='relu'))
    print('Input shape:')
    print(model.input_shape)
    print('Output shape:')
    print(model.output_shape)

    model.add(layers.MaxPooling1D(2))
    print('Input shape:')
    print(model.input_shape)
    print('Output shape:')
    print(model.output_shape)

    model.add(layers.Conv1D(64, 9, activation='relu'))
    print('Input shape:')
    print(model.input_shape)
    print('Output shape:')
    print(model.output_shape)

    # Build detection layers
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()

    # Define algorithms
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train and test model
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # Plot history of model train and testing
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('cnn_test.png')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('Test accuracy: %s' % test_acc)


if __name__ == '__main__':
    main()
