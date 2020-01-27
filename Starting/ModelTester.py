"""Script to produce large quantity of CNN models with varying hyper-parameters

TODO:
    * Move model building into method and loop over
    * Add way to save model performance to file

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
from PIL import Image
import numpy as np
import glob
import random
import os
import json

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

    filenames = os.listdir(path)

    random.seed(42)

    ran_indices = random.sample(range(0, len(filenames)), n_images)
    for i in ran_indices:
        name = filenames[i]
        # Normalize pixel values to be between 0 and 1
        images.append([image.imread(fname=(path + name), format='PNG') / 255.0])
        names.append(name.replace('Blocks\\', '').replace('.png', ''))

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

    names = data['NAME'].tolist()

    print('Length of names: %s' % len(names))

    if len(names) != len(set(names)):
        print(len(set(names)))

    train_names = []

    # Randomly selects the desired number of training images
    for i in random.sample(range(0, len(names)), n):
        train_names.append(names[i])

    print('length of train names: %s' % len(train_names))

    if len(train_names) != len(set(train_names)):
        print(len(set(train_names)))

    # Takes the difference of lists to find remaining names must be for testing
    test_names = list(set(names).difference(set(train_names)))
    print('Length of test names: %s' % len(test_names))

    # Uses these to find those names in data to make cut
    train_images = np.array(data.loc[data['NAME'].isin(train_names)]['IMAGE'].tolist())
    test_images = np.array(data.loc[data['NAME'].isin(test_names)]['IMAGE'].tolist())

    train_labels = np.array(labels.loc[labels['NAME'].isin(train_names)]['LABEL'].tolist())
    test_labels = np.array(labels.loc[labels['NAME'].isin(test_names)]['LABEL'].tolist())

    return train_images.reshape((len(train_images), 4, 4096)), test_images.reshape((len(test_images), 4, 4096)), \
           train_labels, test_labels


def build_model(train_images, test_images, train_labels, test_labels, kernel=(3, 3, 3), activation='relu',
                loss='categorical_crossentropy', optomiser='adam', filters=(64, 64, 128), n_conv=3, n_den=3,
                n_epochs=5):

    print('\nBEGIN MODEL CONSTRUCTION')

    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv1D(filters=filters[0], kernel_size=kernel[0], activation=activation, input_shape=(4096, 4)))
    model.add(layers.MaxPooling1D(2))

    for i in range(n_conv - 1):
        model.add(layers.Conv1D(filters[i + 1], kernel[i + 1], activation=activation))
        model.add(layers.MaxPooling1D(2))

    # Build detection layers
    model.summary()
    model.add(layers.Flatten())

    for i in range(n_den):
        model.add(layers.Dense(filters[i], activation=activation))

    model.add(layers.Dense(2, activation='softmax'))
    model.summary()

    # Define algorithms
    model.compile(optimizer=optomiser, loss=loss, metrics=['accuracy'])

    # Train and test model
    history = model.fit(train_images, train_labels, epochs=n_epochs, validation_data=(test_images, test_labels))

    # Plot history of model train and testing
    #plt.plot(history.history['accuracy'], label='accuracy')
    #plt.plot(history.history['val_accuracy'], label='val_accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.ylim([0.5, 1])
    #plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy: %s' % test_acc)

    log = {'NCONV': n_conv,
           'NDEN': n_den,
           'NEPOCH': n_epochs,
           'KERNELS': kernel,
           'FILTERS': filters,
           'LOSS': loss,
           'ACTIVATE': activation,
           'OPT': optomiser,
           'VAL_ACC': test_acc}

    return log


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    print('***************************** MODELTESTER ********************************************')

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    train_frac = 0.4
    kernels = ((21, 19, 17, 15, 13, 11, 9, 7, 5, 3), #(3, 5, 7, 9, 11, 13, 15, 17, 19, 21),
               #(21, 15, 15, 9,  9,  5,  5, 3, 3, 3), #(9, 9, 9, 9, 9,  9,  9,  9,  9,  9),
               (3, 3, 3, 3, 3,  3,  3,  3, 3, 3, 3))

    filters = ((32, 64, 64, 64, 128, 128, 128, 256, 256, 512), #(64, 64, 128, 128, 128, 128, 256, 256, 512, 512),
               #(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096),
               #(512, 512, 1024, 1024, 2048, 2048, 2048, 4096, 4096, 4096),
               (128, 512, 1024, 2048, 4096, 2048, 1024, 512, 128, 32))

    activation = 'relu'
    loss = 'categorical_crossentropy'
    optomiser = 'adam'
    n_conv = 3
    n_den = 3
    n_epochs = 5

    print('\nLOAD IMAGES')
    # Load in images
    images, names = load_images('Blocks/', 500)

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
    train_images, test_images, train_labels, test_labels = split_data(data, labels, 200)

    # Deletes variables no longer needed
    del data, labels

    train_images = np.swapaxes(train_images, 1, 2)
    test_images = np.swapaxes(test_images, 1, 2)

    log = {}

    i = 0 # Logs model number
    for a in kernels:
        for b in filters:
            for c in range(n_conv):
                for d in range(n_den):
                    for e in range(n_epochs):
                        i = i + 1
                        log['%s' % i] = build_model(train_images, test_images, train_labels, test_labels, kernel=a,
                                                    filters=b, n_conv=c, n_den=d, n_epochs=e, activation=activation,
                                                    loss=loss, optomiser=optomiser)
                        print('MODEL %s COMPLETE' % i)

    # Writes output to JSON file
    with open('test.json', 'w') as fp:
        json.dump(log, fp)


if __name__ == '__main__':
    main()
