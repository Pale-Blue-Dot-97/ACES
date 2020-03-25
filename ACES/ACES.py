"""ACES - Autonomous Communications Enhancement System

TODO:
    * Implement Cassini data as test
"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend, utils
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
import numpy as np
import random
import os
from collections import Counter
import seaborn as sns

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
image_length = 2048
n_channels = 4


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_images(path, n_images=40000, load_all=False):
    """Loads in images and their names from file

    Args:
        path (str): Path to folder containing images
        n_images (int): Number of images to load
        load_all (bool): If true, loads all images found in path

    Returns:
        images ([[float]]): 3D array of all the 2D images
        names ([str]): List of names of each image without path and .png

    """
    # Lists to hold the image data and their names
    images = []
    names = []

    # All the filenames in the path given
    filenames = os.listdir(path)

    if load_all:
        for name in filenames:
            # Normalize pixel values to be between 0 and 1
            images.append([image.imread(fname=(path + name), format='PNG') / 255.0])
            names.append(name.replace(path, '').replace('.png', ''))

    if not load_all:
        # Sets the seed so results are replicable
        random.seed(42)

        # Randomly selects files to be added to the dataset
        ran_indices = random.sample(range(len(filenames)), n_images)
        for i in ran_indices:
            name = filenames[i]

            # Normalize pixel values to be between 0 and 1
            images.append([image.imread(fname=(path + name), format='PNG') / 255.0])
            names.append(name.replace(path, '').replace('.png', ''))

    # Construct DataFrame matching images to their names
    data = pd.DataFrame()
    data['NAME'] = names
    data['IMAGE'] = images

    return data


def load_labels(filename, classes=None, n_classes=None, identity=None):
    """Loads in the file containing the labels for each block, and converts these to OHE

    Returns:
        labels (DataFrame): Dataframe containing names of blocks with their labels as OHE and class name
        n_classes (int): The number of classes detected in labels file
        classes ([str]): List of class names
        identity ([[int]]): The identity matrix of order n_classes

    """

    # Loads in the labels from file as a Pandas.DataFrame
    labels = pd.read_csv(filename, names=('NAME', 'LABEL'), dtype=str, header=0)

    # Finds the class names and creates the identity matrix if not provided
    if classes is None or n_classes is None or identity is None:
        # Finds class names from labels
        classes = [item[0] for item in Counter(labels['LABEL']).most_common()]

        n_classes = len(classes)

        # Creates the identity matrix of order n_classes
        identity = np.identity(n_classes, dtype=int)

    # Prints out the classes and their corresponding OHE label in the data to screen for reference
    for i in range(n_classes):
        print(classes[i], identity[i])

    def name_to_OHE(label):
        """Converts class name as a string to the correct OHE format

        Args:
            label (str): Name of the class this block has been labelled as

        Returns:
            Block label in OHE format

        """
        for j in range(n_classes):
            if label == classes[j]:
                return identity[j]

    # Creates CLASS column as a duplicate of LABEL column to hold class names
    labels['CLASS'] = labels['LABEL']

    # Converts the labels in LABEL to OHE
    labels['LABEL'] = labels['CLASS'].apply(name_to_OHE)

    return labels, n_classes, classes, identity


def balance_data(data, classes, verbose=0):
    """Balances data into equally sized class sub-populations

    Args:
        data (DataFrame): DataFrame containing all the data with labels
        classes ([str]): List of all class names
        verbose (int): Setting for level of output and analysis

    Returns:
        new_data (DataFrame): DataFrame with balanced sized class sub-populations
    """

    # Plot distribution of class sub-populations before balancing
    if verbose == 1:
        plot_subpopulations(data['CLASS'])

    # Find distribution of class sub-populations within data
    modes = Counter(data['CLASS']).most_common()

    # Find the smallest class
    min_size = modes[len(modes) - 1][1]

    # Create a dict to hold all the DataFrames of each class
    dataframes = {}

    # Splits data apart into different DataFrames for each class
    for classification in classes:
        # Creates a DataFrame of just 1 class
        class_df = data[data['CLASS'] == classification].reset_index(drop=True)

        class_size = len(class_df)

        # Randomly selects the names of images to remove from class_df of number of difference in sizes
        names = [class_df['NAME'][i] for i in random.sample(range(class_size), class_size - min_size)]

        # Adds down-sampled DataFrame of class to dict of all class_df
        dataframes[classification] = class_df[~class_df['NAME'].isin(names)]

    # Plot distribution of class sub-populations after balancing
    new_data = pd.concat(dataframes)

    # Plot distribution of class sub-populations after balancing
    if verbose == 1:
        plot_subpopulations(new_data['CLASS'])

    return new_data


def reorder(images):
    return images.reshape((len(images), n_channels, image_length))


def split_data(data, train_frac, val_test_frac=None, verbose=0):
    """Splits data into training, validation and testing data

    Args:
        data (DataFrame): Table of images with filenames and labels
        train_frac (float): Fraction of images desired for training. Remainder for validation and testing
        val_test_frac (float): Fraction of remaining images for validation. Remainder for test.
                               If None, remaining images from train split are assumed just for validation and no test.
        verbose (int): Setting for level of output and analysis

    Returns:
        train_images ([[[float]]]): All training images
        val_images ([[[float]]]): All validation images
        test_images ([[[float]]]): All testing images
        train_labels ([[int]]): All accompanying training labels
        val_labels ([[int]]): All accompanying testing labels
        test_labels ([[int]]): All accompanying testing labels

    """

    # Finds number of images to select for training
    train_n = int(len(data.index) * train_frac)

    # Fixes seed number so results are replicable
    random.seed(42)

    names = data['NAME'].tolist()

    if verbose is 1:
        print('Length of names: %s' % len(names))

    if len(names) != len(set(names)):
        print('ERROR: Duplicate names detected in data!')

    train_names = []

    # Randomly selects the desired number of training images
    for i in random.sample(range(0, len(names)), train_n):
        train_names.append(names[i])

    if verbose is 1:
        print('Length of train names: %s' % len(train_names))

    # Takes the difference of lists to find remaining names must be for validation and testing
    val_n_test_names = list(set(names).difference(set(train_names)))

    if verbose is 1:
        print('Length of validation and test names: %s' % len(val_n_test_names))

    val_names = []

    if val_test_frac is None:

        val_names = val_n_test_names

        # Uses these to find those names in data to make cut
        train_images = np.array(data.loc[data['NAME'].isin(train_names)]['IMAGE'].tolist())
        val_images = np.array(data.loc[data['NAME'].isin(val_names)]['IMAGE'].tolist())

        train_labels = np.array(data.loc[data['NAME'].isin(train_names)]['LABEL'].tolist())
        val_labels = np.array(data.loc[data['NAME'].isin(val_names)]['LABEL'].tolist())

        if verbose is 1:
            print('Length of train images: %s' % len(train_images))
            print('Length of train labels: %s' % len(train_labels))
            print('Length of validation images: %s' % len(val_images))
            print('Length of validation labels: %s' % len(val_labels))

        return reorder(train_images), reorder(val_images), train_labels, val_labels

    else:
        # Finds the number of images to randomly select as validation split from test
        val_n = int(len(data.index) * val_test_frac)

        # Randomly selects the desired number of validation images from the remaining names
        for i in random.sample(range(0, len(val_n_test_names)), val_n):
            val_names.append(val_n_test_names[i])

        test_names = list(set(val_n_test_names).difference(set(val_names)))

        if verbose is 1:
            print('Length of test names: %s' % len(test_names))

        # Uses these to find those names in data to make cut
        train_images = np.array(data.loc[data['NAME'].isin(train_names)]['IMAGE'].tolist())
        val_images = np.array(data.loc[data['NAME'].isin(val_names)]['IMAGE'].tolist())
        test_images = np.array(data.loc[data['NAME'].isin(test_names)]['IMAGE'].tolist())

        train_labels = np.array(data.loc[data['NAME'].isin(train_names)]['LABEL'].tolist())
        val_labels = np.array(data.loc[data['NAME'].isin(val_names)]['LABEL'].tolist())
        test_labels = np.array(data.loc[data['NAME'].isin(test_names)]['LABEL'].tolist())

        if verbose is 1:
            print('Length of train images: %s' % len(train_images))
            print('Length of train labels: %s' % len(train_labels))
            print('Length of validation images: %s' % len(val_images))
            print('Length of validation labels: %s' % len(val_labels))
            print('Length of test images: %s' % len(test_images))
            print('Length of test labels: %s' % len(test_labels))

        return reorder(train_images), reorder(val_images), reorder(test_images), train_labels, val_labels, test_labels


def set_optimiser(optimiser):
    """Creates a tf.keras.optimizer based on parameters given

    Args:
        optimiser ((str, float) / (str, float, float)): Parameters of optimiser
        (name, learning_rate, momentum (optional))

    Returns:
        optimiser (tf.keras.optimizer): The desired optimiser object
        optimiser_name (str): ID code for optimiser for file naming

    """

    if optimiser[0] is 'RMSprop':
        return tf.keras.optimizers.RMSprop(learning_rate=optimiser[1]), '%s_%sL' % (optimiser[0], optimiser[1])
    if optimiser[0] is 'Adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=optimiser[1]), '%s_%sL' % (optimiser[0], optimiser[1])
    if optimiser[0] is 'Adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=optimiser[1]), '%s_%sL' % (optimiser[0], optimiser[1])
    if optimiser[0] is 'Adamax':
        return tf.keras.optimizers.Adamax(learning_rate=optimiser[1]), '%s_%sL' % (optimiser[0], optimiser[1])
    if optimiser[0] is 'Nadam':
        return tf.keras.optimizers.Nadam(learning_rate=optimiser[1]), '%s_%sL' % (optimiser[0], optimiser[1])
    if optimiser[0] is 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=optimiser[1]), '%s_%sL' % (optimiser[0], optimiser[1])
    if optimiser[0] is 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=optimiser[1], momentum=optimiser[2]), \
               '%s_%sL_%sm' % (optimiser[0], optimiser[1], optimiser[2])
    else:
        return


def plot_subpopulations(class_labels):
    """Creates a pie chart of the distribution of the classes within the data

    Args:
        class_labels ([int]): List of predicted classifications from model, in form of class numbers

    Returns:
        None
    """

    # Finds the distribution of the classes within the data
    modes = Counter(class_labels).most_common()

    # List to hold the name and percentage distribution of each class in the data as str
    classes = []

    # List to hold the total counts of each class
    counts = []

    # Finds total number of images to normalise data
    n_images = len(class_labels)

    # For each class, find the percentage of data that is that class and the total counts for that class
    for label in modes:
        classes.append('%s (%s)' % (label[0], (label[1] / n_images)))
        counts.append(label[1])

    # Plot a pie chart of the data distribution amongst the classes with labels of class name and percentage size
    plt.pie(counts, labels=classes)

    # Show plot for review
    plt.show()


def multi_head_CNN(train_images, train_labels, val_images, val_labels, test_images, test_labels,
                   verbose=1, epochs=50, batch_size=32, in_filt=8):
    """Fits a multi-head CNN using provided hyper-parameters and data

    Args:
        train_images ([[[float]]]): Images for training
        train_labels ([[int]]): Accompanying labels for training images
        val_images ([[[float]]]): Images for validation
        val_labels ([[int]]): Accompanying labels for validation images
        test_images ([[[float]]]): Images for testing model post-fitting
        test_labels ([[int]]): Accompanying labels for testing images
        verbose (int): Setting for level of output and analysis
        epochs (int): Number of epochs of training
        batch_size (int): Number of images in each batch for network input
        in_filt (int): Number of filters in the initial CNN layer. Used as baseline number for subsequent layers

    Returns:
        accuracy (float): Test accuracy of model
    """

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
    history = model.fit([train_images, train_images, train_images], train_labels,
                        validation_data=([val_images, val_images, val_images], val_labels),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose)

    # evaluate model
    _, accuracy = model.evaluate([test_images, test_images, test_images], test_labels, batch_size=batch_size, verbose=0)

    return accuracy


def sequential_CNN(train_images, train_labels, val_images, val_labels, test_images, test_labels, n_classes,
                   epochs=5, batch_size=32, class_weights=None, in_filt=8, filt_mult=2, kernel=9, n_conv=3,
                   n_dense=3, fn_neurons=32, optimiser='SGD', filename='test.csv', log=False, verbose=0,):
    """Creates a sequential CNN using Keras based on hyper-parameters and data supplied

    Args:
        train_images ([[[float]]]): Images for training
        train_labels ([[int]]): Accompanying labels for training images
        val_images ([[[float]]]): Images for validation
        val_labels ([[int]]): Accompanying labels for validation images
        test_images ([[[float]]]): Images for testing model post-fitting
        test_labels ([[int]]): Accompanying labels for testing images
        n_classes (int): Number of classes in data
        epochs (int): Number of epochs of training
        batch_size (int): Number of images in each batch for network input
        class_weights:
        in_filt (int): Number of filters in the initial CNN layer. Used as baseline number for subsequent layers
        filt_mult (int): Factor by which to increase the number of filters in each sucessive layer
        kernel (int): Size of kernel in CNN layers
        n_conv (int): Number of CNN layers (except initial layer)
        n_dense (int): Number of dense layers (except classification layer)
        fn_neurons (): Number of neurons in the final dense layer. Used as baseline number for preceding layers
        optimiser (str, tensorflow.keras.optimizer): Optimiser to use
        filename (str): Name of log file to output History of model to
        log (bool): Whether to output history to file
        verbose (int): Setting for level of output and analysis

    Returns:
        history (keras.History): Object holding the history of the model fitting
        model (keras.Model): The produced Keras model

    """

    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv1D(filters=in_filt, kernel_size=9, activation='relu', batch_size=batch_size,
                            input_shape=(image_length, n_channels)))
    model.add(layers.MaxPooling1D(2, strides=filt_mult))

    # Add convolutional layers
    for i in range(n_conv):
        model.add(layers.Conv1D(in_filt * pow(filt_mult, i + 1), kernel, activation='relu'))
        model.add(layers.MaxPooling1D(2, strides=filt_mult))

    # Build detection layers
    model.add(layers.Flatten())
    for i in range(n_dense):
        model.add(layers.Dense(fn_neurons * pow(filt_mult, n_dense - i), activation='relu'))

    # Add classification layer
    model.add(layers.Dense(n_classes, activation='softmax'))

    if verbose is 2:
        model.summary()

    # Define algorithms
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Creates CSVLogger to output history of model fitting to
    log = callbacks.CSVLogger(filename=filename, append=True, separator=',')

    # Train and test model
    history = model.fit(train_images, train_labels,
                        class_weight=class_weights,
                        epochs=epochs,
                        validation_data=(val_images, val_labels),
                        callbacks=[log])

    # Test model using test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy: %s' % test_acc)

    return history, model


def plot_history(history, filename, show=True, save=False):
    """Plot history of model train and testing

    Args:
        history (keras.History): Object holding the history of the model fitting
        filename (str): Name of file to save plot to
        show (bool): Whether to show plot
        save (bool): Whether to save plot to file

    Returns:
         None
    """

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    # Shows and/or saves plot
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
        plt.close()


def OHE_to_class(ohe_labels, n_classes, classes):
    """Converts OHE labels to string class names

    Args:
        ohe_labels ([[int]]): List of class labels in OHE format
        n_classes (int): Number of classes in data
        classes ([str]): List of all class names

    Returns:
        class_labels ([str]): List of class name labels
    """

    class_labels = []

    for i in range(len(ohe_labels)):
        for j in range(n_classes):
            if ohe_labels[i] == j:
                class_labels.append(classes[j])

    return class_labels


def plot_predictions(model, test_images, batch_size, n_classes, classes):
    """Produces a pie chart of the predictions made by a model

    Args:
        model (keras.Model):
        test_images ([[[float]]]): Images for testing model post-fitting
        batch_size (int): Number of images in each batch for network input
        n_classes (int): Number of classes in data
        classes ([str]): List of all class names

    Returns:
        None
    """

    # Uses model to make predictions on the images supplied
    pred_labels = model.predict_classes(test_images, batch_size=batch_size)

    # Converts these labels from OHE to class names
    class_labels = OHE_to_class(pred_labels, n_classes, classes)

    # Plots distribution of these class names in the data as a pie chart
    plot_subpopulations(class_labels)


def make_confusion_matrix(model, test_images, test_labels, batch_size, classes, filename, show=True, save=False):
    """Creates a heat-map of the confusion matrix of the given model

    Args:
        model (keras.Model):
        test_images ([[[float]]]): Images for testing model post-fitting
        test_labels ([[int]]): Accompanying labels for testing images
        batch_size (int): Number of images in each batch for network input
        classes ([str]): List of all class names
        filename (str): Name of file to save plot to
        show (bool): Whether to show plot
        save (bool): Whether to save plot to file

    Returns:
        None
    """

    # Uses model to make predictions on the images supplied
    pred_labels = model.predict_classes(test_images, batch_size=batch_size)

    # Creates the confusion matrix based on these predictions and the corresponding ground truth labels
    conf_matrix = tf.math.confusion_matrix(labels=np.argmax(test_labels, axis=1), predictions=pred_labels).numpy()

    # Normalises confusion matrix
    conf_matrix_norm = np.around(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

    # Converts confusion matrix to Pandas.DataFrame
    con_mat_df = pd.DataFrame(conf_matrix_norm, index=classes, columns=classes)

    # Plots figure
    plt.figure(figsize=(9, 9))
    sns.heatmap(con_mat_df, annot=True, square=True, cmap=plt.cm.get_cmap('Blues'))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Shows and/or saves plot
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
        plt.close()


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    print('***************************** ACES ********************************************')
    model_type = 'sequential'
    epochs = 40
    verbose = 1
    batch_size = 32

    in_filters = [16]
    filt_mult = [2]

    fn_neurons = [32]

    kernels = [9]

    n_conv = [3]
    n_dense = [3]

    optimisers = [('Adagrad', 0.01), ('Adamax', 0.01), ('Nadam', 2e-5), ('SGD', 0.01, 0.5)]

    print('\nLOAD IMAGES')
    # Load in images
    data = load_images('Voyager1_Blocks/', 20000)

    print('\nLOAD LABELS')
    # Load in accompanying labels into separate randomly ordered DataFrame
    labels, n_classes, classes, identity = load_labels('VOY1_Block_Labels.csv')

    # Merges data and labels together
    data = pd.merge(data, labels, on='NAME')

    # Deletes un-needed variable
    del labels

    print('\nBALANCING DATA')
    data = balance_data(data, classes, verbose=0)

    print('\nSPLIT DATA INTO TRAIN AND VALIDATION')
    # Split images into test and train
    train_images, val_images, train_labels, val_labels = split_data(data, 0.7)

    print('\nLOAD IN TEST DATA')
    test_data_df = load_images('Cassini_Rev20_Blocks/', load_all=True)
    test_labels_df, x, y, z = load_labels('Cassini_Block_Labels/Cassini_Rev20_Block_Labels.csv',
                                          classes=classes, n_classes=n_classes, identity=identity)

    test_data = pd.merge(test_data_df, test_labels_df, on='NAME')

    # Deletes un-needed variables
    del test_data_df, test_labels_df, x, y, z

    test_images = reorder(np.array(test_data['IMAGE'].tolist()))
    test_labels = np.array(test_data['LABEL'].tolist())

    # Corrects shape of images
    train_images = np.swapaxes(train_images, 1, 2)
    val_images = np.swapaxes(val_images, 1, 2)
    test_images = np.swapaxes(test_images, 1, 2)

    if model_type is 'multi-head':
        print('Test accuracy: %s' % multi_head_CNN(train_images, train_labels, val_images, val_labels,
                                                   test_images, test_labels, epochs=epochs, batch_size=batch_size,
                                                   in_filt=32, verbose=verbose))

    if model_type is 'sequential':
        i = 0  # Logs model number
        for a in kernels:
            for b in in_filters:
                for c in n_conv:
                    for d in n_dense:
                        for e in optimisers:
                            for f in filt_mult:
                                for g in fn_neurons:
                                    i = i + 1

                                    # Determines optimiser
                                    optimiser, optimiser_name = set_optimiser(e)

                                    # Unique model ID to use for logging and output
                                    model_name = '%sM_%sK_%sF_%sC_%sD_%s_%sfm_%sd' % (i, a, b, c, d, optimiser_name, f, g)

                                    print('\nMODEL NUMBER: %s' % i)
                                    print('Kernel: %s' % a)
                                    print('Initial filters: %s' % b)
                                    print('Number of convolutional layers: %s' % c)
                                    print('Number of dense layers: %s' % d)
                                    print('Optimiser: %s' % e[0])
                                    print('Learning rate: %s' % e[1])

                                    # Performs model fitting with given hyper-parameters
                                    history, model = sequential_CNN(train_images, train_labels, val_images, val_labels,
                                                                    test_images, test_labels, n_classes, epochs=epochs,
                                                                    batch_size=batch_size, in_filt=b, filt_mult=f,
                                                                    kernel=a, n_conv=c, n_dense=d, optimiser=optimiser,
                                                                    fn_neurons=g, verbose=verbose,
                                                                    filename='Logs/%s.csv' % model_name, log=True)

                                    # Plots the history of the model fitting and saves to file
                                    plot_history(history, 'ROCs/%s-ROC.png' % model_name, show=False, save=True)

                                    # Creates the confusion matrix of the model and saves to file
                                    make_confusion_matrix(model, test_images, test_labels, batch_size, classes,
                                                          'Confusion_Matrices/%s-CM.png' % model_name, show=False,
                                                          save=True)


if __name__ == '__main__':
    main()
