"""Script to process cleaned and normalised Voyager magnetometer data into 4-channel blocks for use as training and
validation data for neural networks using manually constructed labels

TODO:
    * Implement manual labels into block creation
    * Fully comment and docstring

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


def create_random_blocks(data, data_columns, n, block_length, thres_frac=0.2):
    """Selects n number of random 4096 long blocks from the data as numpy arrays

        Args:
            data (DataFrame): Table containing data to split into blocks
            data_columns (list): List of column names containing the data
            n (int): Number of blocks to randomly select
            block_length (int): Length of each block
            thres_frac (float): Fraction of block_length above which it can be considered interesting

        Returns:
            blocks ([[[float]]]): 3D array containing blocks of 4096 * 4 values

    """
    data = data.copy()
    data.reset_index(drop=True)

    # Threshold number of interesting points in block to be considered interesting
    thres = int(thres_frac * block_length)

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

    data['BR_norm'] = data['BR_norm'].multiply(-1)
    data['BTH_norm'] = data['BTH_norm'].multiply(-1)
    data['BPH_norm'] = data['BPH_norm'].multiply(-1)

    return data.reset_index(drop=True)


def data_perturb(data, mode):

    # reverse data
    if mode == 'reverse':
        return reverse_data(data)

    # mirror data
    if mode == 'mirror':
        return mirror_data(data)

    return


def blocks_to_images(blocks, name):
    """Converts each block in a series to 8-bit greyscale png images and saves to file

    Args:
        blocks: Series of blocks of data
        name: Name of the series to identify images with

    Returns:
        None
    """

    for block in blocks:
        Image.fromarray((block[2] * 255).astype(np.uint8), mode='L').save('Blocks/%s_%s.png' % (block[0], name))

    return


def block_to_image(block):
    """Takes a n_length long block of the data and converts to a greyscale image

    Args:
        block ([[float]]): 2D numpy array of 4 rows of data n_length points long

    Returns:
        image (Image): A n_length x n_channel greyscale Image

    """
    image = Image.fromarray((block * 255).astype(np.uint8), mode='L')
    return image


def labels_to_file(all_blocks, all_names):

    names = []
    labels = []

    for i in range(len(all_names)):
        for block in all_blocks[i]:
            names.append('%s_%s' % (block[0], all_names[i]))
            labels.append(block[1])

    data = pd.DataFrame()
    data['NAME'] = names
    data['LABEL'] = labels
    data.to_csv('Block_Labels.csv')

    return


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    n = 10000  # Number of blocks to create for each data perturbation
    block_length = 1024

    data_columns = ['BR_norm', 'BTH_norm', 'BPH_norm', 'BMAG_norm']

    print('*************************** WELCOME TO DATAPROCESS2 *************************************')

    print('\nLOADING DATA')
    data = load_labels()

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
    blocks = create_random_blocks(stan_data, data_columns, n, block_length)

    print('\t-MIRRORED DATA')
    mir_blocks = create_random_blocks(mir_dat, data_columns, n, block_length)

    print('\t-REVERSED DATA')
    rev_blocks = create_random_blocks(rev_dat, data_columns, n, block_length)

    print('\t-MIRRORED AND REVERSED DATA')
    mir_rev_blocks = create_random_blocks(mir_rev_dat, data_columns, n, block_length)

    print('\nCONVERTING BLOCKS TO IMAGES:')

    print('\t-STANDARD DATA')
    blocks_to_images(blocks, 'OG')

    print('\t-MIRRORED DATA')
    blocks_to_images(mir_blocks, 'MIR')

    print('\t-REVERSED DATA')
    blocks_to_images(rev_blocks, 'REV')

    print('\t-MIRRORED AND REVERSED DATA')
    blocks_to_images(mir_rev_blocks, 'MIR_REV')

    print('\nEXPORTING LABELS TO FILE')
    labels_to_file((blocks, mir_blocks, rev_blocks, mir_rev_blocks), ('OG', 'MIR', 'REV', 'MIR_REV'))
    
    print('\nFINISHED')


if __name__ == '__main__':
    main()
