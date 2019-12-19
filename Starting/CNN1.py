"""Script run simple CNN on Voyager 2 magnetometer data for binary classification

TODO:
    * Convert boolean labels to binary
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


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_images(path):

    images = []
    names = []

    for name in glob.glob('%s*.png' % path):
        images.append(np.array(Image.open(name).getdata()))
        names.append(name.replace('Blocks\\', '').replace('.png', ''))

    return images, names


def load_labels():

    labels = pd.read_csv('VOY2_JE_LABELS.csv', names=('NAME', 'LABEL'), header=0)
    return labels


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    # Load in images
    images, names = load_images('Blocks/')

    data = pd.DataFrame()
    data['NAME'] = names
    data['IMAGE'] = images

    print(data)

    # Load in accompanying labels
    labels = load_labels()

    print(labels)
    """
    # Split images into test and train

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Build convolutional layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
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
    history = model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

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
    """

if __name__ == '__main__':
    main()
