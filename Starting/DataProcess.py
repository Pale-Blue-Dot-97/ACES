"""Script to process cleaned and normalised Voyager magnetometer data into 4-channel blocks for use as training and
validation data for neural networks

TODO:
    * Fully comment and docstring

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import time
import sys
import pandas as pd
import numpy as np
import Plot2D as laplt
import matplotlib.pyplot as plt
import scipy.signal as sg
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

    norm_data = data.drop(columns=['BR', 'BTH', 'BPH', 'BMAG'])

    norm_data.reset_index(drop=True)
    data.reset_index(drop=True)

    return data, norm_data


def renormalise(data):
    """Re-normalises data from -1<x<1 to 0<x<1

    Args:
        data (DataFrame): Table containing data normalised to -1<x<1

    Returns:
        data (DataFrame): Data re-normalised to 0<x<1

    """

    new_data = data.copy()

    for i in ['BR_norm', 'BPH_norm', 'BTH_norm', 'BMAG_norm']:
        new_data[i] = (data[i] + 1.0).multiply(0.5)

    return new_data.reset_index(drop=True)


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


def create_random_blocks(data, data_columns, n):
    """Selects n number of random 4096 long blocks from the data as numpy arrays

        Args:
            data (DataFrame): Table containing data to split into blocks
            data_columns (list): List of column names containing the data
            n (int): Number of blocks to randomly select

        Returns:
            blocks ([[[float]]]): 3D array containing blocks of 4096 * 4 values

    """
    data = data.copy()
    data.reset_index(drop=True)

    # Sets seed number at 42 to produce same selection of indices every run
    random.seed(42)

    blocks = []

    ran_indices = random.sample(range(len(data[data_columns[0]]) - 4096), n)

    # Slices DataFrame into 4096 long blocks
    for i in ran_indices:
        block_slice = data[i: (i + 4096)]

        # Labels block based on mode of point labels in block slice
        label = block_slice['LABELS'].mode()[0]

        block = []

        for k in data_columns:
            channel = np.array(block_slice['%s' % k])
            block.append(channel)
            if len(channel) != 4096:
                print('%s: %s' % (k, len(channel)))

        # Adds tuple of the first index of the block, and the block
        blocks.append((i, label, np.array(block)))

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

    return data.reset_index(drop=True)


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

    return


def label_data(data, data_columns, kernel_size=4096, pad=250):
    """

    Args:
        data (DataFrame): Table of data to be labelled
        data_columns ([str]): List of the names of columns containing data
        kernel_size (int): Size of the kernel for scipy.signal.find_peaks() to use
        pad (int): Number of points either side of a detection to include as `interesting'

    Returns:
        labelled_data (DataFrame): data with the addition of column `LABELS' with labels of True/ False for each point

    """

    labelled_data = data.copy()

    interesting = []

    # Performs a rolling average along each column
    roll = data.rolling(window=kernel_size, win_type=None, min_periods=1)

    for i in data_columns:
        # Finds the peaks in the rolling mean to identify interesting spots
        peaks = sg.find_peaks(x=np.abs(roll[i].mean().tolist()), width=200)[0]

        # Adds the padding about each peak in mean found
        for j in peaks:
            interesting += range(j - pad, j + pad)

    # Eliminate duplicates in interesting
    interesting = np.array(list(dict.fromkeys(interesting)))

    # Removes out-of-bound indexes from interesting
    interesting = interesting[0 <= interesting]
    interesting = interesting[interesting < len(data[data_columns[0]])]

    # Creates new column in DataFrame to hold True for interesting points and False if not
    labelled_data['LABELS'] = float('NaN')
    labelled_data['LABELS'][interesting] = True
    labelled_data['LABELS'].fillna(False, inplace=True)

    return labelled_data.reset_index(drop=True)


def blocks_to_images(blocks, name):
    """Converts each block in a series to 8-bit greyscale png images and saves to file

    Args:
        blocks: Series of blocks of data
        name: Name of the series to identify images with

    Returns:
        None
    """

    for block in blocks:
        Image.fromarray((block[2] * 255).astype(np.uint8), mode='L')\
            .save('Blocks/%s_%s_%s.png' % (block[0], name, block[1]))

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


def labels_to_file(all_blocks, all_names):

    names = []
    labels = []

    for i in range(4):
        for block in all_blocks[i]:
            names.append('%s_%s_%s' % (block[0], all_names[i], block[1]))
            labels.append(block[1])

    data = pd.DataFrame()
    data['NAME'] = names
    data['LABEL'] = labels
    data.to_csv('VOY2_JE_LABELS.csv')

    return


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    engine = speech.init()
    speech_on = False
    memes_on = False

    n = 10000  # Number of blocks to create for each data perturbation

    data_columns = ['BR_norm', 'BTH_norm', 'BPH_norm', 'BMAG_norm']

    print('*************************** WELCOME TO DATAPROCESS *************************************')

    print('\nLOADING DATA')
    if speech_on:
        engine.say("Loading data")
        engine.runAndWait()
    data, norm_data = load_data()

    print('\nRE-NORMALISING DATA')
    if speech_on:
        engine.say("Re-normalising data")
        engine.runAndWait()
    re_norm_data = renormalise(norm_data)

    print('\nPERTURBING DATA:')
    if speech_on:
        engine.say("Perturbing data")
        engine.runAndWait()

    print('\t-MIRRORING DATA')
    if speech_on:
        engine.say("Mirroring data")
        engine.runAndWait()
    mir_dat = data_perturb(re_norm_data, 'mirror')

    print('\t-REVERSING DATA')
    if speech_on:
        engine.say("Reversing data")
        engine.runAndWait()
    rev_dat = data_perturb(re_norm_data, 'reverse')

    print('\t-MIRRORING AND REVERSING DATA')
    if speech_on:
        engine.say("Mirroring and reversing data")
        engine.runAndWait()
    mir_rev_dat = data_perturb(mir_dat, 'reverse')

    print('\nLABELLING DATA:')
    if speech_on:
        engine.say("Labelling data")
        engine.runAndWait()

    print('\t-STANDARD DATA')
    if speech_on:
        engine.say("Standard data")
        engine.runAndWait()
    stan_data = label_data(norm_data, data_columns)

    print('\t-MIRRORED DATA')
    if speech_on:
        engine.say("Mirrored data")
        engine.runAndWait()
    mir_dat = label_data(mir_dat, data_columns)

    print('\t-REVERSED DATA')
    if speech_on:
        engine.say("Reversed data")
        engine.runAndWait()
    rev_dat = label_data(rev_dat, data_columns)

    print('\t-MIRRORED AND REVERSED DATA')
    if speech_on:
        engine.say("Mirrored and reversed data")
        engine.runAndWait()
    mir_rev_dat = label_data(mir_rev_dat, data_columns)

    print('\nCREATING RANDOMISED BLOCKS:')
    if speech_on:
        engine.say("Creating randomised blocks")
        engine.runAndWait()

    print('\t-STANDARD DATA')
    if speech_on:
        engine.say("Standard data")
        engine.runAndWait()
    blocks = create_random_blocks(stan_data, data_columns, n)

    print('\t-MIRRORED DATA')
    if speech_on:
        engine.say("Mirrored data")
        engine.runAndWait()
    mir_blocks = create_random_blocks(mir_dat, data_columns, n)

    print('\t-REVERSED DATA')
    if speech_on:
        engine.say("Reversed data")
        engine.runAndWait()
    rev_blocks = create_random_blocks(rev_dat, data_columns, n)

    print('\t-MIRRORED AND REVERSED DATA')
    if speech_on:
        engine.say("Mirrored and reversed data")
        engine.runAndWait()
    mir_rev_blocks = create_random_blocks(mir_rev_dat, data_columns, n)

    print('\nCONVERTING BLOCKS TO IMAGES:')
    if speech_on:
        engine.say("Converting blocks to images")
        engine.runAndWait()

    print('\t-STANDARD DATA')
    if speech_on:
        engine.say("Standard data")
        engine.runAndWait()
    blocks_to_images(blocks, 'OG')

    print('\t-MIRRORED DATA')
    if speech_on:
        engine.say("Mirrored data")
        engine.runAndWait()
    blocks_to_images(mir_blocks, 'MIR')

    print('\t-REVERSED DATA')
    if speech_on:
        engine.say("Reversed data")
        engine.runAndWait()
    blocks_to_images(rev_blocks, 'REV')

    print('\t-MIRRORED AND REVERSED DATA')
    if speech_on:
        engine.say("Mirrored and reversed data")
        engine.runAndWait()
    blocks_to_images(mir_rev_blocks, 'MIR_REV')

    print('\nEXPORTING LABELS TO FILE')
    if speech_on:
        engine.say("EXPORTING LABELS TO FILE")
        engine.runAndWait()
    labels_to_file((blocks, mir_blocks, rev_blocks, mir_rev_blocks), ('OG', 'MIR', 'REV', 'MIR_REV'))

    if speech_on:
        # Alert bell
        for i in range(1, 3):
            sys.stdout.write('\r\a')
            sys.stdout.flush()
            time.sleep(0.5)
        sys.stdout.write('\n')
    
    print('\nFINISHED')
    if speech_on:
        engine.say("Finished")
        engine.runAndWait()

    if memes_on:
        # A little something to cheer up anyone's day
        # List of Star Wars memes
        STWRs_memes = ('https://www.youtube.com/watch?v=v_YozYt8l-g', 'https://www.youtube.com/watch?v=Sg14jNbBb-8',
                       'https://www.youtube.com/watch?v=lCscYsICvoA', 'https://www.youtube.com/watch?v=sNjWpZmxDgg',
                       'https://www.youtube.com/watch?v=r0zj3Ap74Vw', 'https://www.youtube.com/watch?v=LRXm2zFAmwc',
                       'https://www.youtube.com/watch?v=J0BciHfsU7k')

        # List of The Thick Of It memes
        TTOI_memes = ('https://www.youtube.com/watch?v=KFkJLlU-3GI', 'https://www.youtube.com/watch?v=M9spU_T9Oys',
                      'https://www.youtube.com/watch?v=dP4cKky7WC8', 'https://www.youtube.com/watch?v=YhOUaYzO0dE',
                      'https://www.youtube.com/watch?v=xM8DfCgVWx8', 'https://www.youtube.com/watch?v=28YEH--rX3c',
                      'https://www.youtube.com/watch?v=pF3a-DQdDJI')

        # Randomly selects a meme from the list selected
        webbrowser.open(random.choice(TTOI_memes))


if __name__ == '__main__':
    main()
