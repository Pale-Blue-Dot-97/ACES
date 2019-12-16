"""Script to process cleaned and normalised Voyager magnetometer data into 4-channel blocks for use as training and
validation data for neural networks

TODO:
    * Add methods to process data into 4-channel blocks for training
    * Add method to automatically label data

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import time
import sys
import pandas as pd
import numpy as np
from PIL import Image
import pyttsx3 as speech
import webbrowser
import random

# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================


def load_data():
    """Load in cleaned and normalised data from file

    Returns:
        data (DataFrame): Table of all the cleaned and normalised data from file
        norm_data (DataFrame): Table of just the normalised data

    """

    data_names = ['BR', 'BTH', 'BPH', 'BMAG', 'UNIX TIME', 'BR_norm', 'BTH_norm', 'BPH_norm', 'BMAG_norm']

    data = pd.read_csv('VOY2_JE_PROC.csv', names=data_names, dtype=float, header=0)

    norm_data = data.drop(columns=['BR', 'BTH', 'BPH', 'BMAG', 'UNIX TIME'])

    return data, norm_data


def renormalise(data):
    """Re-normalises data from -1<x<1 to 0<x<1

    Args:
        data (DataFrame): Table containing data normalised to -1<x<1

    Returns:
        data (DataFrame): Data re-normalised to 0<x<1

    """

    data = data.copy()

    for i in ['BR_norm', 'BPH_norm', 'BTH_norm', 'BMAG_norm']:
        data[i] = (data[i] + 1.0).multiply(0.5)

    return data


def create_blocks(data):
    """Splits the data into 4096 long blocks as numpy arrays

    Args:
        data (DataFrame): Table containing data to split into blocks

    Returns:
        blocks ([[[float]]]): 3D array containing blocks of 4096 * 4 values

    """

    blocks = []

    # Slices DataFrame into 4096 long blocks
    for i in range(int(len(data['BR_norm']) / 4096.0)):
        block_slice = data[(i-1) * 4096: (i * 4096) - 1]

        block = []

        for j in ['BR_norm', 'BPH_norm', 'BTH_norm', 'BMAG_norm']:
            block.append(np.array(block_slice[j]))

        blocks.append(np.array(block))

    return blocks


def create_random_blocks(data, n):
    """Selects n number of random 4096 long blocks from the data as numpy arrays

        Args:
            data (DataFrame): Table containing data to split into blocks
            n (int): Number of blocks to randomly select

        Returns:
            blocks ([[[float]]]): 3D array containing blocks of 4096 * 4 values

    """

    blocks = []

    # Slices DataFrame into 4096 long blocks
    for i in range(n):
        j = random.choice(np.linspace(0, len(data['BR_norm'] - 4096), 1))
        block_slice = data[j : j + 4096]

        block = []

        for j in ['BR_norm', 'BPH_norm', 'BTH_norm', 'BMAG_norm']:
            block.append(np.array(block_slice[j]))

        blocks.append(np.array(block))

    return blocks


def reverse_data(data):
    """Reverses the order of the DataFrame

    Args:
        data (DataFrame): Table of data

    Returns:
        data(DataFrame): Backwards ordering of data

    """
    data = data.copy()

    return data[::-1].reset_index(drop=True)


def mirror_data(data):
    """Switches sign of data (excluding magnitude and time of course)

        Args:
            data (DataFrame): Table of data

        Returns:
            data(DataFrame): Data mirrored in x-axis

    """
    data = data.copy()

    data['BR_norm'] = data['BR_norm'].multiply(-1)
    data['BTH_norm'] = data['BTH_norm'].multiply(-1)
    data['BPH_norm'] = data['BPH_norm'].multiply(-1)

    return data


def smooth_data(data):
    return


def sharpen_data(data):
    return


def shift_data(data):
    return


def multiply_data(data):
    return


def data_perturb(data, mode):

    # reverse data
    if mode == 'reverse':
        return reverse_data(data)

    # mirror data
    if mode == 'mirror':
        return mirror_data(data)

    # smooth data
    # sharpen data
    # raise data
    # lower data
    # increase data
    # decrease data

    return


def label_data():
    return


def block_to_image(block):
    """Takes a 4096 long block of the data and converts to a greyscale image

    Args:
        block ([[float]]): 2D numpy array of 4 rows of data 4096 points long

    Returns:
        image (Image): A 4096 x 4 greyscale Image

    """
    image = Image.fromarray((block * 255).astype(np.uint8), mode='L')
    return image


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    engine = speech.init()

    engine.say("Loading data")
    engine.runAndWait()

    data, norm_data = load_data()

    engine.say("Re-normalising data")
    engine.runAndWait()

    norm_data = renormalise(norm_data)

    engine.say("Perturbing data")
    engine.runAndWait()
    mir_dat = data_perturb(norm_data, 'mirror')

    engine.say("Creating blocks")
    engine.runAndWait()

    mir_blocks = create_blocks(mir_dat)
    blocks = create_blocks(norm_data)

    engine.say("Converting blocks to images")
    engine.runAndWait()

    image = block_to_image(blocks[10])
    mir_image = block_to_image(mir_blocks[10])

    engine.say("Saving test image")
    engine.runAndWait()

    image.save('test_block.png')
    mir_image.save('mir_test_block.png')

    # Alert bell
    for i in range(1, 3):
        sys.stdout.write('\r\a')
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write('\n')

    engine.say("Finished")
    engine.runAndWait()

    # A little something to cheer up anyone's day
    # List of Star Wars memes
    STWRs_memes = ('https://www.youtube.com/watch?v=QiZNSzWIaLo', 'https://www.youtube.com/watch?v=Sg14jNbBb-8',
                   'https://www.youtube.com/watch?v=lCscYsICvoA', 'https://www.youtube.com/watch?v=sNjWpZmxDgg',
                   'https://www.youtube.com/watch?v=r0zj3Ap74Vw', 'https://www.youtube.com/watch?v=LRXm2zFAmwc',
                   'https://www.youtube.com/watch?v=J0BciHfsU7k')

    # List of The Thick Of It memes
    TTOI_memes = ('https://www.youtube.com/watch?v=KFkJLlU-3GI', 'https://www.youtube.com/watch?v=M9spU_T9Oys',
                  'https://www.youtube.com/watch?v=dP4cKky7WC8', 'https://www.youtube.com/watch?v=YhOUaYzO0dE',
                  'https://www.youtube.com/watch?v=xM8DfCgVWx8', 'https://www.youtube.com/watch?v=28YEH--rX3c')

    # Randomly selects a meme from the list selected
    webbrowser.open(random.choice(STWRs_memes))


if __name__ == '__main__':
    main()
