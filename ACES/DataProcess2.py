"""Script to process cleaned and normalised Voyager magnetometer data into 4-channel blocks for use as training and
validation data for neural networks using manually constructed labels

TODO:

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pandas as pd
import numpy as np
from PIL import Image
import random
from collections import Counter
from Labeller import load_labels
import sys


# =====================================================================================================================
#                                                     GLOBAL
# =====================================================================================================================
# Fraction of a block to be threshold to reach for block to labelled as such
threshold_fraction = 0.5

# Number of blocks to create for each data perturbation
n = 1500

# Length of each block
block_length = 2048

data_columns = ['BR', 'BTH', 'BPH', 'BMAG']


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def renormalise(data):
    """Re-normalises data from -1<x<1 to 0<x<1

    Args:
        data (DataFrame): Table containing data normalised to -1<x<1

    Returns:
        data (DataFrame): Data re-normalised to 0<x<1

    """

    new_data = data.copy()

    for i in data_columns:
        new_data[i] = (data[i] + 1.0).multiply(0.5)

    return new_data.reset_index(drop=True)


def create_random_blocks(data):
    """Selects n number of random 4096 long blocks from the data as numpy arrays

        Args:
            data (DataFrame): Table containing data to split into blocks

        Returns:
            blocks ([[[float]]]): 3D array containing blocks of 4096 * 4 values

    """
    data = data.copy()
    data.reset_index(drop=True)

    # Threshold number of interesting points in block to be considered interesting
    thres = int(threshold_fraction * block_length)

    # Sets seed number at 42 to produce same selection of indices every run
    random.seed(42)

    blocks = []

    ran_indices = random.sample(range(len(data[data_columns[0]]) - block_length), n)

    # Slices DataFrame into blocks
    for i in ran_indices:
        block_slice = data[i: (i + block_length)]

        # Assume block label is False initially
        label = False

        # Finds 2 most common labels of the block
        mode = Counter(block_slice['LABELS']).most_common()

        if len(mode) > 1:
            # If mode is not False, mode must be a classification
            if mode[0][0] is not False:
                # If more than the threshold value of the block is the mode, label block as that mode
                if mode[0][1] > thres:
                    label = mode[0][0]

            # If mode is False, the 2nd mode may be a classification that reaches threshold
            if mode[0][0] is False:
                # The 2nd mode must be a classification label. If it reaches threshold, label block as such
                if mode[1][1] > thres:
                    label = mode[1][0]
                # Else, label block as False
                else:
                    label = False
        else:
            label = mode[0][0]

        block = []

        for k in data_columns:
            channel = np.array(block_slice['%s' % k])
            block.append(channel)
            if len(channel) != block_length:
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

    data['BR'] = data['BR'].multiply(-1)
    data['BTH'] = data['BTH'].multiply(-1)
    data['BPH'] = data['BPH'].multiply(-1)

    return data.reset_index(drop=True)


def data_perturb(data, mode):

    # reverse data
    if mode == 'reverse':
        return reverse_data(data)

    # mirror data
    if mode == 'mirror':
        return mirror_data(data)

    return


def blocks_to_images(blocks, name, block_dir):
    """Converts each block in a series to 8-bit greyscale png images and saves to file

    Args:
        blocks: Series of blocks of data
        name (str): Name of the series to identify images with
        block_dir (str): Name of directory to save images to

    Returns:
        None
    """

    for block in blocks:
        Image.fromarray((block[2] * 65535).astype(np.uint16), mode='I;16')\
            .save('%s/%s_%s.png' % (block_dir, block[0], name))

    return


def labels_to_file(all_blocks, all_names, voy_num):
    """

    Args:
        all_blocks ():
        all_names ([str]): List of names of data pertubations
        voy_num (str): 1 or 2 to identify which Voyager mission this is

    Returns:

    """

    names = []
    labels = []

    for i in range(len(all_names)):
        for block in all_blocks[i]:
            names.append('%s_%s' % (block[0], all_names[i]))
            labels.append(block[1])

    data = pd.DataFrame()
    data['NAME'] = names
    data['LABEL'] = labels
    data.to_csv('Voyager%s/VOY%s_Block_Labels.csv' % (voy_num, voy_num))

    return


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    print('*************************** WELCOME TO DATAPROCESS2 *************************************')

    event = sys.argv[1]
    block_dir = 'Voyager%s_Blocks' % event
    perturb_names = ('VOY%s_OG' % event, 'VOY%s_MIR' % event, 'VOY%s_REV' % event, 'VOY%s_MIR_REV' % event)

    print('\nLOADING DATA')
    data, classes = load_labels('Voyager%s/VOY%s_data.csv' % (event, event),
                                'Voyager%s/VOY%s_Labels.csv' % (event, event))

    print('\nRE-NORMALISING DATA')
    stan_data = renormalise(data)

    print('\nPERTURBING DATA:')

    print('\t-MIRRORING DATA')
    mir_dat = data_perturb(stan_data, 'mirror')

    print('\t-REVERSING DATA')
    rev_dat = data_perturb(stan_data, 'reverse')

    print('\t-MIRRORING AND REVERSING DATA')
    mir_rev_dat = data_perturb(mir_dat, 'reverse')

    print('\nCREATING RANDOMISED BLOCKS:')

    print('\t-STANDARD DATA')
    blocks = create_random_blocks(stan_data)

    print('\t-MIRRORED DATA')
    mir_blocks = create_random_blocks(mir_dat)

    print('\t-REVERSED DATA')
    rev_blocks = create_random_blocks(rev_dat)

    print('\t-MIRRORED AND REVERSED DATA')
    mir_rev_blocks = create_random_blocks(mir_rev_dat)

    print('\nCONVERTING BLOCKS TO IMAGES:')

    print('\t-STANDARD DATA')
    blocks_to_images(blocks, perturb_names[0], block_dir)

    print('\t-MIRRORED DATA')
    blocks_to_images(mir_blocks, perturb_names[1], block_dir)

    print('\t-REVERSED DATA')
    blocks_to_images(rev_blocks, perturb_names[2], block_dir)

    print('\t-MIRRORED AND REVERSED DATA')
    blocks_to_images(mir_rev_blocks, perturb_names[3], block_dir)

    print('\nEXPORTING LABELS TO FILE')
    labels_to_file((blocks, mir_blocks, rev_blocks, mir_rev_blocks), perturb_names, event)
    
    print('\nFINISHED')


if __name__ == '__main__':
    main()
