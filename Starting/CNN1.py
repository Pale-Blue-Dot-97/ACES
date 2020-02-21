"""Script run simple CNN on Voyager 2 magnetometer data for binary classification

TODO:
    * Construct model

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models, backend
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

    labels['CLASS'] = labels['LABEL']
    labels['LABEL'] = labels['CLASS'].apply(bool_to_binary)

    return labels


def split_data(data, labels, n, image_length, n_channels, stratify=True):
    """Splits data into training and testing data

    Args:
        data (DataFrame): Table of images with filenames
        labels (DataFrame): Table of labels for images with filenames
        n (int): Number of training images desired
        image_length (int): Length of each image
        n_channels (int): Number of channels of each image
        stratify (bool): Whether to startify training data in equal True/ False cases. Default True

    Returns:
        train_images ([[[float]]]): All training images
        test_images ([[[float]]]): All testing images
        train_labels ([[int]]): All accompanying training labels
        test_labels ([[int]]): All accompanying testing labels

    """
    # Fixes seed number so results are replicable
    random.seed(42)

    names = data['NAME'].tolist()

    true_names = labels[labels['CLASS'] == 'True']['NAME'].tolist()
    false_names = labels[labels['CLASS'] == 'False']['NAME'].tolist()

    print('Length of names: %s' % len(names))

    if len(names) != len(set(names)):
        print(len(set(names)))

    train_names = []

    true_train_names = []
    false_train_names = []

    if stratify is True:
        print('STRATIFYING TRAINING DATA')

        # Randomly selects the desired number of true case training images
        for i in random.sample(range(0, len(true_names)), int(0.5 * n)):
            true_train_names.append(true_names[i])

        # Randomly selects the desired number of false case training images
        for i in random.sample(range(0, len(false_names)), int(0.5 * n)):
            false_train_names.append(false_names[i])

        print('length of true train names: %s' % len(true_train_names))
        print('length of false train names: %s' % len(false_train_names))

        train_names = true_train_names + false_train_names

    if stratify is False:
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

    print('Length of train images: %s' % len(train_images))
    print('Length of train labels: %s' % len(train_labels))
    print('Length of test images: %s' % len(test_images))
    print('Length of test labels: %s' % len(test_labels))

    return train_images.reshape((len(train_images), n_channels, image_length)), \
           test_images.reshape((len(test_images), n_channels, image_length)), train_labels, test_labels


def test_metric(y_true, y_pred):
    return backend.mean(y_pred)


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    print('***************************** CNN1 ********************************************')
    image_length = 1024
    n_channels = 4
    in_filt = 8
    filt_mult = 2

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print('\nLOAD IMAGES')
    # Load in images
    images, names = load_images('Blocks/', 40000)

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
    train_images, test_images, train_labels, test_labels = split_data(data, labels, 10000, image_length, n_channels)

    # Deletes variables no longer needed
    del data, labels

    train_images = np.swapaxes(train_images, 1, 2)
    test_images = np.swapaxes(test_images, 1, 2)

    print('\nBEGIN MODEL CONSTRUCTION')

    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv1D(filters=in_filt, kernel_size=9, activation='relu', #batch_size=100,
                            input_shape=(image_length, n_channels)))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    model.add(layers.Conv1D(in_filt*pow(filt_mult, 1), 9, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    model.add(layers.Conv1D(in_filt*pow(filt_mult, 2), 9, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    #model.add(layers.Conv1D(in_filt*pow(filt_mult, 2), 9, activation='relu'))
    #model.add(layers.MaxPooling1D(2, strides=filt_mult))

    # Build detection layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.summary()

    # Define algorithms
    model.compile(optimizer=tf.keras.optimizers.SGD(),#learning_rate=0.001, momentum=0.0),
                  loss='binary_crossentropy', metrics=['binary_accuracy'])

    # Train and test model
    history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

    # Plot history of model train and testing
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['binary_accuracy'], label='accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('cnn_test.png')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy: %s' % test_acc)


if __name__ == '__main__':
    main()
