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
from sklearn.utils import class_weight


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

    # Construct DataFrame matching images to their names
    data = pd.DataFrame()
    data['NAME'] = names
    data['IMAGE'] = images

    return data


def load_labels():
    """

    Returns:
        labels (DataFrame):

    """

    labels = pd.read_csv('Block_Labels.csv', names=('NAME', 'LABEL'), dtype=str, header=0)

    # Finds class names from labels
    classes = [item[0] for item in Counter(labels['LABEL']).most_common()]

    n_classes = len(classes)

    identity = np.identity(n_classes, dtype=int)

    for i in range(n_classes):
        print(classes[i], identity[i])

    def bool_to_binary(label):
        for j in range(n_classes):
            if label == classes[j]:
                return identity[j]

    labels['CLASS'] = labels['LABEL']
    labels['LABEL'] = labels['CLASS'].apply(bool_to_binary)

    return labels, n_classes, classes, identity


def balance_data(data, classes, verbose=0):

    if verbose == 1:
        plot_subpopulations(data['CLASS'])

    modes = Counter(data['CLASS']).most_common()

    min_size = modes[len(modes) - 1][1]

    dataframes = {}

    for classification in classes:
        class_df = data[data['CLASS'] == classification].reset_index(drop=True)

        class_size = len(class_df)
        names = [class_df['NAME'][i] for i in random.sample(range(class_size), class_size - min_size)]

        dataframes[classification] = class_df[~class_df['NAME'].isin(names)]

    new_data = pd.concat(dataframes)

    if verbose == 1:
        plot_subpopulations(new_data['CLASS'])

    return new_data


def split_data(data, train_frac):
    """Splits data into training and testing data

    Args:
        data (DataFrame): Table of images with filenames and labels
        train_frac (float): Fraction of images desired for training

    Returns:
        train_images ([[[float]]]): All training images
        test_images ([[[float]]]): All testing images
        train_labels ([[int]]): All accompanying training labels
        test_labels ([[int]]): All accompanying testing labels

    """

    # Finds number of images to select for training
    n = int(len(data.index) * train_frac)

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

    print('Length of train names: %s' % len(train_names))

    if len(train_names) != len(set(train_names)):
        print(len(set(train_names)))

    # Takes the difference of lists to find remaining names must be for testing
    test_names = list(set(names).difference(set(train_names)))
    print('Length of test names: %s' % len(test_names))

    # Uses these to find those names in data to make cut
    train_images = np.array(data.loc[data['NAME'].isin(train_names)]['IMAGE'].tolist())
    test_images = np.array(data.loc[data['NAME'].isin(test_names)]['IMAGE'].tolist())

    train_labels = np.array(data.loc[data['NAME'].isin(train_names)]['LABEL'].tolist())
    test_labels = np.array(data.loc[data['NAME'].isin(test_names)]['LABEL'].tolist())

    print('Length of train images: %s' % len(train_images))
    print('Length of train labels: %s' % len(train_labels))
    print('Length of test images: %s' % len(test_images))
    print('Length of test labels: %s' % len(test_labels))

    return train_images.reshape((len(train_images), n_channels, image_length)), \
           test_images.reshape((len(test_images), n_channels, image_length)), \
           train_labels, test_labels


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


def multi_head_CNN(train_images, train_labels, test_images, test_labels, verbose=1, epochs=50, batch_size=32):

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


def sequential_CNN(train_images, train_labels, test_images, test_labels, n_classes,
                   epochs=5, batch_size=32, class_weights=None, in_filt=8, filt_mult=2, verbose=0):

    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv1D(filters=in_filt, kernel_size=9, activation='relu', batch_size=batch_size,
                            input_shape=(image_length, n_channels)))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    model.add(layers.Conv1D(in_filt * pow(filt_mult, 1), 9, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    model.add(layers.Conv1D(in_filt*pow(filt_mult, 2), 9, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))
    model.add(layers.Conv1D(in_filt*pow(filt_mult, 3), 9, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))

    # Build detection layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.summary()

    # Define algorithms
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4, momentum=0.0, clipvalue=0.5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Train and test model
    history = model.fit(train_images, train_labels,
                        class_weight=class_weights,
                        epochs=epochs,
                        validation_data=(test_images, test_labels))

    if verbose is 1 or 2:
        plot_history(history)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('Test accuracy: %s' % test_acc)

    return history, model


def plot_history(history):
    # Plot history of model train and testing
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.show()


def plot_predictions(model, test_images, batch_size, n_classes, classes):
    pred_labels = model.predict_classes(test_images, batch_size=batch_size)

    class_labels = []

    for i in range(len(pred_labels)):
        for j in range(n_classes):
            if pred_labels[i] == j:
                class_labels.append(classes[j])

    plot_subpopulations(class_labels)


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    print('***************************** ACES ********************************************')
    in_filt = 32
    filt_mult = 2
    batch_size = 32
    model_type = 'sequential'
    verbose = 0

    print('\nLOAD IMAGES')
    # Load in images
    data = load_images('Blocks/', 80000)

    print('\nLOAD LABELS')
    # Load in accompanying labels into separate randomly ordered DataFrame
    labels, n_classes, classes, identity = load_labels()

    # Merges data and labels together
    data = pd.merge(data, labels, on='NAME')

    # Deletes un-needed variable
    del labels

    print('\nBALANCING DATA')
    data = balance_data(data, classes, verbose=0)

    print('\nSPLIT DATA INTO TRAIN AND TEST')
    # Split images into test and train
    train_images, test_images, train_labels, test_labels = split_data(data, 0.8)

    train_images = np.swapaxes(train_images, 1, 2)
    test_images = np.swapaxes(test_images, 1, 2)

    print('\nBEGIN MODEL CONSTRUCTION')
    if model_type is 'sequential':
        history, model = sequential_CNN(train_images, train_labels, test_images, test_labels, n_classes,
                                        epochs=50, batch_size=batch_size, in_filt=in_filt, filt_mult=filt_mult,
                                        verbose=verbose)

        if verbose == 2:
            plot_predictions(model, test_images, batch_size, n_classes, classes)

    if model_type is 'multi-head':
        print('Test accuracy: %s' % multi_head_CNN(train_images, train_labels, test_images, test_labels))


if __name__ == '__main__':
    main()
