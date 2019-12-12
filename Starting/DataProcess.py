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


def create_blocks(data):
    """Splits the data into 4096 long blocks as numpy arrays

    Args:
        data (DataFrame): Table containing data to split into blocks

    Returns:
        blocks ([[[float]]]): 3D array containing blocks of 4096 * 4 values

    """

    blocks = []

    for i in range(int(len(data['BR_norm']) / 4096.0)):
        block_slice = data[(i-1) * 4096: (i * 4096) - 1]

        block = []

        for j in ['BR_norm', 'BPH_norm', 'BTH_norm', 'BMAG_norm']:
            block.append(np.array(block_slice[j]))

        blocks.append(np.array(block))

    return blocks


def mirror_data(data):
    """

    Args:
        data (DataFrame): Table of data

    Returns:
        data(DataFrame): Backwards ordering of data

    """
    return data[::-1].reset_index(drop=True)


def smooth_data(data):
    return


def sharpen_data(data):
    return


def shift_data(data):
    return


def multiply_data(data):
    return


def data_perturb(data, mode):

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

    image = Image.fromarray(block, mode='L')
    return image


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    engine = speech.init()

    engine.say("Loading data")
    engine.runAndWait()

    data, norm_data = load_data()

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
    for i in range(1, 4):
        sys.stdout.write('\r\a')
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write('\n')

    engine.say("Finished")
    engine.runAndWait()

    STWRs_memes = ('https://www.youtube.com/watch?v=QiZNSzWIaLo', 'https://www.youtube.com/watch?v=Sg14jNbBb-8',
                   'https://www.youtube.com/watch?v=lCscYsICvoA')

    TTOI_memes = ('https://www.youtube.com/watch?v=KFkJLlU-3GI', 'https://www.youtube.com/watch?v=M9spU_T9Oys',
                  'https://www.youtube.com/watch?v=dP4cKky7WC8', 'https://www.youtube.com/watch?v=YhOUaYzO0dE',
                  'https://www.youtube.com/watch?v=xM8DfCgVWx8', 'https://www.youtube.com/watch?v=28YEH--rX3c')
    webbrowser.open(random.choice(TTOI_memes))


if __name__ == '__main__':
    main()
