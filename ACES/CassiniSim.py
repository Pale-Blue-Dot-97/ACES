"""Script to simulate the recording of data by Cassini in near real-time as a test for neural networks

TODO:
    * Re-work for Cassini data in ordered blocks rather than randomised blocks
    * Streamline code

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pandas as pd
import numpy as np
from PIL import Image
import sys
from collections import Counter
from Labeller import load_labels


# =====================================================================================================================
#                                                     GLOBAL
# =====================================================================================================================
# Fraction of a block to be threshold to reach for block to labelled as such
threshold_fraction = 0.5

# Length of each block
block_length = 4096

data_columns = ['BR', 'BTH', 'BPH', 'BMAG']

block_labels_path = 'Cassini_Block_Labels'

data_path = 'Processed_Cassini_Data'

labels_path = 'Cassini_Labels'


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


def create_blocks(data, stride=2):
    """Slices dataset into blocks of image_length in order, at strides apart

        Args:
            data (DataFrame): Table containing data to split into blocks

        Returns:
            blocks ([[[float]]]): 3D array containing blocks of image_length * n_channels values

    """
    data = data.copy()
    data.reset_index(drop=True)

    # Threshold number of points of a class in a block for whole block to be classified as such
    thres = int(threshold_fraction * block_length)

    blocks = []

    indices = range(int(len(data) * stride / block_length))

    # Slices DataFrame into blocks
    for i in indices:
        block_slice = data[(i-1) * int(block_length/stride): (i * block_length) - 1]

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

        # Adds tuple of the block number, label of the block, and the block itself
        blocks.append((i, label, np.array(block)))

    return blocks


def blocks_to_images(blocks, block_folder):
    """Converts each block in a series to 8-bit greyscale png images and saves to file

    Args:
        blocks: Series of blocks of data

    Returns:
        None
    """

    for block in blocks:
        Image.fromarray((block[2] * 255).astype(np.uint8), mode='L').save('%s/%s.png' % (block_folder, block[0]))

    return


def labels_to_file(blocks, filename):

    names = []
    labels = []

    for block in blocks:
        names.append('%s' % block[0])
        labels.append(block[1])

    data = pd.DataFrame()
    data['NAME'] = names
    data['LABEL'] = labels
    data.to_csv('%s/%s' % (block_labels_path, filename))


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    print('*************************** WELCOME TO CASSINISIM *************************************')

    rev_num = sys.argv[1]

    print('\nLOADING DATA')
    data, classes = load_labels('%s/CASSINI_Rev%s_PROC.csv' % (data_path, rev_num),
                                '%s/Cassini_Labels_Rev%s' % (labels_path, rev_num))

    print('\nRE-NORMALISING DATA')
    data = renormalise(data)

    print('\nCREATING BLOCKS:')
    blocks = create_blocks(data, stride=2)

    print('\nCONVERTING BLOCKS TO IMAGES:')
    blocks_to_images(blocks, 'Cassini_Rev%s_Blocks' % rev_num)

    print('\nEXPORTING LABELS TO FILE')
    labels_to_file(blocks, 'Cassini_Rev%s_Block_Labels.csv' % rev_num)
    
    print('\nFINISHED')


if __name__ == '__main__':
    main()
