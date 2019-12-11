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

    norm_data = data.drop(columns=['BR', 'BTH', 'BPH', 'BMAG'])

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
        block = data[(i-1) * 4096:i * 4096].to_numpy()
        blocks.append(block)

    return blocks


def block_to_image(block):
    """Takes a 4096 long block of the data and converts to a greyscale image

    Args:
        block ([[float]]): 2D numpy array of 4 rows of data 4096 points long

    Returns:
        image (Image): A 4096 x 4 greyscale Image

    """

    image = Image.fromarray(block, mode='1')
    return image


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    engine = speech.init()

    engine.say("Loading data")
    engine.runAndWait()

    data, norm_data = load_data()

    engine.say("Creating blocks")
    engine.runAndWait()

    blocks = create_blocks(norm_data)

    engine.say("Converting blocks to images")
    engine.runAndWait()

    image = block_to_image(blocks[10])

    engine.say("Saving test image")
    engine.runAndWait()

    image.save('test_block.png')

    # Alert bell
    for i in range(1, 4):
        sys.stdout.write('\r\a')
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write('\n')

    engine.say("Finished")
    engine.runAndWait()


if __name__ == '__main__':
    main()
