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
import tensorflow.keras


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

    print('Random selection of all image indices: ')
    print(random.sample(range(0, len(filenames)), n_images)[0:19])

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


def split_data(data, labels, n, image_length, n_channels):
    """Splits data into training and testing data

    Args:
        data (DataFrame): Table of images with filenames
        labels (DataFrame): Table of labels for images with filenames
        n (int): Number of training images desired
        image_length (int): Length of each image
        n_channels (int): Number of channels of each image

    Returns:
        train_images ([[[float]]]): All training images
        test_images ([[[float]]]): All testing images
        train_labels ([[int]]): All accompanying training labels
        test_labels ([[int]]): All accompanying testing labels

    """
    # Fixes seed number so results are replicable
    random.seed(42)

    names = data['NAME'].tolist()

    print('Names of images selected:')
    print(names[0:19])

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
    test_names = sorted(list(set(names).difference(set(train_names))))
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
           test_images.reshape((len(test_images), n_channels, image_length)), train_labels, test_labels, train_names, \
           test_names


def test_metric(y_true, y_pred):
    return backend.mean(y_pred)


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
# def main():


if __name__ == '__main__':
    # np.random.seed(1)
    # tf.set_random_seed(42)
    print('***************************** CNN2 ********************************************')
    image_length = 1024
    n_channels = 4

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print('\nLOAD IMAGES')
    # Load in images
    images, names = load_images('Blocks/', 10000)

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
    train_images, test_images, train_labels, test_labels, train_names, test_names = split_data(data, labels, 3000,
                                                                                               image_length, n_channels)

    # Deletes variables no longer needed
    del data, labels

    train_images = np.swapaxes(train_images, 1, 2)
    test_images = np.swapaxes(test_images, 1, 2)
    print('\nBEGIN MODEL CONSTRUCTION')

    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv1D(filters=2, strides=1, kernel_size=3, activation='relu', input_shape=(image_length,n_channels)))
    model.add(layers.MaxPooling1D(2))
    # model.add(layers.Conv2D(filters=4, kernel_size=9, strides=2, activation='relu'))
    # model.add(layers.MaxPooling1D(4))
    # model.add(layers.Conv1D(8, 9, strides=4, activation='relu'))
    # model.add(layers.MaxPooling1D(2))
    # model.add(layers.Conv1D(2, 9, strides=2, activation='relu'))
    # model.add(layers.MaxPooling1D(2))

    # Build detection layers
    # model.summary()
    model.add(layers.Flatten())
    # model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.summary()

    # Define algorithms
    model.compile(optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0), loss='mean_squared_error', metrics=['binary_accuracy'])
    # print(model.weights)
    # Train and test model
    history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    # print(model.weights)

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

    pred_labels = model.predict(test_images)
    pred_labels[pred_labels>0.5] = 1.0
    pred_labels[pred_labels<0.5] = 0.0

    pp = np.where(np.sum(pred_labels == test_labels, axis=1) == 0)[0]
    print(pp[0:9])

    bad_image_names = []
    for i in pp:
        bad_image_names.append(test_names[i])

    print(bad_image_names[0:9])

    data_names = ['BR', 'BTH', 'BPH', 'BMAG', 'UNIX TIME', 'BR_norm', 'BTH_norm', 'BPH_norm', 'BMAG_norm']
    data = pd.read_csv('VOY2_JE_PROC.csv', names=data_names, dtype=float, header=0)

    bad_images = []

    print('Test_names:')
    print(test_names[0:19])

    for name in bad_image_names:
        bad_images.append(float(name.split('_')[0]))

    plt.plot(data['BR_norm'])
    plt.plot(bad_images, [0]*len(bad_images), '|')
    # plt.plot(roll['BR_norm'].std().tolist())
    plt.show()
