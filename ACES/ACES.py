"""ACES

TODO:
    * Construct model

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models, backend, utils
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
import numpy as np
import random
import os
from collections import Counter


# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
image_length = 4096
n_channels = 4


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_images(path, n_images):
    """Loads in images and their names from file

    Args:
        path (str): Path to folder containing images
        n_images (int): Number of images to load

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

    labels = pd.read_csv('Block_Labels.csv', names=('NAME', 'LABEL'), dtype=str, header=0)

    def bool_to_binary(label):
        if label == 'False':
            return [1, 0, 0, 0]
        elif label == 'CSC':
            return [0, 1, 0, 0]
        elif label == 'NSC':
            return [0, 0, 1, 0]
        elif label == 'MP':
            return [0, 0, 0, 1]

    labels['CLASS'] = labels['LABEL']
    labels['LABEL'] = labels['CLASS'].apply(bool_to_binary)

    return labels


def split_data(data, labels, n, image_length, n_channels):
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

    print('Length of train images: %s' % len(train_images))
    print('Length of train labels: %s' % len(train_labels))
    print('Length of test images: %s' % len(test_images))
    print('Length of test labels: %s' % len(test_labels))

    return train_images.reshape((len(train_images), n_channels, image_length)), \
           test_images.reshape((len(test_images), n_channels, image_length)), train_labels, test_labels


def plot_subpopulations(class_labels):
    modes = Counter(class_labels).most_common()
    classes = []
    counts = []
    n_images = len(class_labels)

    for label in modes:
        classes.append('%s (%s)' % (label[0], (label[1] / n_images)))
        counts.append(label[1])

    plt.pie(counts, labels=classes)

    plt.show()


def evaluate_model(train_images, train_labels, test_images, test_labels):
    verbose, epochs, batch_size = 1, 10, 128
    n_timesteps, n_features, n_outputs = train_images.shape[1], train_images.shape[2], train_labels.shape[1]

    # head 1
    inputs1 = layers.Input(shape=(n_timesteps, n_features))
    conv1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    drop1 = layers.Dropout(0.5)(conv1)
    pool1 = layers.MaxPooling1D(pool_size=2)(drop1)
    flat1 = layers.Flatten()(pool1)

    # head 2
    inputs2 = layers.Input(shape=(n_timesteps, n_features))
    conv2 = layers.Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
    drop2 = layers.Dropout(0.5)(conv2)
    pool2 = layers.MaxPooling1D(pool_size=2)(drop2)
    flat2 = layers.Flatten()(pool2)

    # head 3
    inputs3 = layers.Input(shape=(n_timesteps, n_features))
    conv3 = layers.Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
    drop3 = layers.Dropout(0.5)(conv3)
    pool3 = layers.MaxPooling1D(pool_size=2)(drop3)
    flat3 = layers.Flatten()(pool3)

    # merge
    merged = layers.concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = layers.Dense(100, activation='relu')(merged)
    outputs = layers.Dense(n_outputs, activation='softmax')(dense1)
    model = models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    # save a plot of the model
    #utils.vis_utils.plot_model(model, show_shapes=True, to_file='multichannel.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit([train_images, train_images, train_images], train_labels, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate model
    _, accuracy = model.evaluate([test_images, test_images, test_images], test_labels, batch_size=batch_size, verbose=0)

    return accuracy


def sequential_CNN(train_images, train_labels, test_images, test_labels, in_filt=8, filt_mult=2):

    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv1D(filters=in_filt, kernel_size=9, activation='relu', batch_size=128,
                            input_shape=(image_length, n_channels)))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    model.add(layers.Conv1D(in_filt * pow(filt_mult, 1), 9, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    model.add(layers.Conv1D(in_filt*pow(filt_mult, 2), 9, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    model.add(layers.Conv1D(in_filt*pow(filt_mult, 2), 9, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))

    # Build detection layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4, activation='sigmoid'))
    model.summary()

    # Define algorithms
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Train and test model
    history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy: %s' % test_acc)

    return history, model


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    print('***************************** ACES ********************************************')
    in_filt = 8
    filt_mult = 2

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

    print('\nDISPLAYING CLASS DISTRIBUTION')
    plot_subpopulations(labels['CLASS'])

    print('\nSPLIT DATA INTO TRAIN AND TEST')
    # Split images into test and train
    train_images, test_images, train_labels, test_labels = split_data(data, labels, 30000, image_length, n_channels)

    # Deletes variables no longer needed
    del data, labels

    train_images = np.swapaxes(train_images, 1, 2)
    test_images = np.swapaxes(test_images, 1, 2)

    print('\nBEGIN MODEL CONSTRUCTION')

    history, model = sequential_CNN(train_images, train_labels, test_images, test_labels)

    # Plot history of model train and testing
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    pred_labels = model.predict_classes(test_images, batch_size=128)

    #pred_labels = [str(i) for i in pred_labels]

    classes = ['False', 'CSC', 'NSC', 'MP']
    number = [0, 1, 2, 3]

    class_labels = []

    for i in range(len(pred_labels)):
        for j in range(len(classes)):
            if pred_labels[i] == number[j]:
                class_labels.append(classes[j])

    plot_subpopulations(class_labels)

    #print('Test accuracy: %s' % evaluate_model(train_images, train_labels, test_images, test_labels))


if __name__ == '__main__':
    main()
