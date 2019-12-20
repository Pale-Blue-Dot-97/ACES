"""Script run simple CNN on Voyager 2 magnetometer data for binary classification

TODO:
    * Split data into train and test
    * Construct model

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================

#from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import glob
import random


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_images(path):
    """Loads in images and their names from file

    Args:
        path (str): Path to folder containing images

    Returns:
        images ([[float]]): 3D array of all the 2D images
        names ([str]): List of names of each image without path and .png

    """

    images = []
    names = []

    for name in glob.glob('%s*.png' % path):
        # Normalize pixel values to be between 0 and 1
        images.append([np.array(Image.open(name).getdata()) / 255.0])
        names.append(name.replace('Blocks\\', '').replace('.png', ''))

    return images, names


def load_labels():
    """

    Returns:

    """

    labels = pd.read_csv('VOY2_JE_LABELS.csv', names=('NAME', 'LABEL'), dtype=str, header=0)

    def bool_to_binary(label):
        if label == 'True':
            return 1
        if label == 'False':
            return 0

    labels['LABEL'] = labels['LABEL'].apply(bool_to_binary)

    return labels


def split_data(data, labels, n):
    random.seed(42)

    names = data['NAME']

    train_names = []

    for i in range(n):
        train_names.append(random.choice(names))

    test_names = list(set(names).difference(set(train_names)))

    train_images = data.loc[data['NAME'].isin(train_names)]['IMAGE']
    test_images = data.loc[data['NAME'].isin(test_names)]['IMAGE']

    train_labels = labels.loc[labels['NAME'].isin(train_names)]['LABEL']
    test_labels = labels.loc[labels['NAME'].isin(test_names)]['LABEL']

    return train_images, test_images, train_labels, test_labels


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():

    # Load in images
    images, names = load_images('Blocks/')

    # Construct DataFrame matching images to their names
    data = pd.DataFrame()
    data['NAME'] = names
    data['IMAGE'] = images

    # Load in accompanying labels into separate randomly ordered DataFrame
    labels = load_labels()

    # Split images into test and train
    train_images, test_images, train_labels, test_labels = split_data(data, labels, 8000)

    # *********** BROKEN DUE TO INCORRECT SHAPES OF CONV LAYERS! NEED 1D CONV LAYERS *******************************
    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1, 4, 4096), data_format='channels_first'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

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
