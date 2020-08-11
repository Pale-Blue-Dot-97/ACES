"""Script to load Cassini magnetometer data in from file and clean, interpolate and work on
and process ready for use as test data for ACES.py

TODO:

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from PIL import Image
import random
from collections import Counter
from Labeller import label_data

# =====================================================================================================================
#                                                     GLOBAL
# =====================================================================================================================
data_path = 'Ulysses'
block_path = 'Ulysses_Blocks'

# Fraction of a block to be threshold to reach for block to labelled as such
threshold_fraction = 0.5

# Number of blocks to create for each data perturbation
n = 250

# Length of each block
block_length = 2048

data_columns = ['BR', 'BTH', 'BPH', 'BMAG']

perturb_names = ('ULY_OG', 'ULY_MIR', 'ULY_REV', 'ULY_MIR_REV')


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_data():
    """Load in data from PDS archive. Column headers come from the LBL meta data

    Returns:
        data (DataFrame): Table of the magnetometer data time-series
        data_columns ([str]): List of the header names of columns containing important data
        position (DataFrame): Table of the position of the spacecraft in R, LON, LAT

    """

    data_names = ['TIME', 'BR', 'BTH', 'BPH', 'BMAG']

    data = pd.read_table('%s/FGM38_40.TAB' % data_path, delim_whitespace=True, names=data_names, na_values=-9999.999)

    print('Number of NaNs: %d' % data.isnull().sum().sum())

    # Removes any 'NaNs' from the DataFrame
    for i in data_columns:
        data.drop(data[data.isnull()[i]].index, inplace=True)
        data.reset_index(inplace=True, drop=True)

    print('Size of raw data: %d' % len(data))

    position_names = ['TIME', 'R', 'LAT', 'LON', 'LOCTIME']

    position = pd.read_table('%s/SPK28_45.TAB' % data_path, delim_whitespace=True, names=position_names)
    position.drop(columns=['LAT', 'LON', 'LOCTIME'], inplace=True)

    return data, position


def extract_time(data):
    """Extracts the time since UNIX Epoch from time-stamp strings

    Args:
        data (DataFrame): Data containing time-stamp column to extract from

    Returns:
        new_data (DataFrame): Data with column containing times since UNIX Epoch
    """

    def get_UNIX_time(row):
        dt = datetime.datetime.strptime(row['TIME'], '%Y-%m-%dT%H:%M:%S.%fZ')
        return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

    new_data = data.copy()

    print('\nExtracting UNIX time from time-stamps')

    new_data['UNIX TIME'] = new_data.apply(get_UNIX_time, axis=1)

    new_data.drop(columns=['TIME'], inplace=True)

    # Resets index after indices have been dropped to avoid key errors
    new_data.reset_index(drop=True)

    return new_data


def match_positions(positions, data):
    """Matches positional data to match time intervals of main data

    Args:
        positions (DataFrame): Positions of the spacecraft
        data (DataFrame): Magnetometer data

    Returns:
        new_data (DataFrame): data with the addition of the positions interpolated to match 1.0s

    """

    positions = extract_time(positions)

    new_data = data.merge(positions, how='left', left_on='UNIX TIME', right_on='UNIX TIME')

    return new_data


def plot_data(data):
    # Create Matplotlib datetime64 type date-time column from UNIX time
    data['DATETIME'] = pd.to_datetime(data['UNIX TIME'], unit='s')

    # Re-index data to date-time
    data.index = data['DATETIME']
    del data['DATETIME']

    data.plot(y=['BR', 'BTH', 'BPH', 'BMAG'], kind='line')

    plt.legend(['BR', 'BTH', 'BPH', 'BMAG'], loc='upper right')
    plt.show()


def pow_normalise(data, a=4.0e5, b=200.0, c=35.0):
    """Normalise data using power series of position

    Args:
        data (DataFrame): DataFrame containing data to be normalised with contained position
        a (float): Scalar in power series
        b (float): Scalar in power series
        c (float): Scalar in power series

    Returns:
        norm_data (DataFrame): Dataframe with new columns with normalised data

    """

    def power_series_norm(x, r):
        """

        Args:
            x (float, Array-like): Values to be normalised
            r (float, Array-like): Distance to planet to normalise with

        Returns:
            Normalised data
        """

        return x / (a * np.power(r, -3) + b * np.power(r, -2) + c * np.power(r, -1))

    norm_data = data.copy()

    # More complex polynomial approach
    print('\nApplying power series normalisation to data')
    norm_data['BR'] = power_series_norm(data['BR'], data['R'])
    norm_data['BTH'] = power_series_norm(data['BTH'], data['R'])
    norm_data['BPH'] = power_series_norm(data['BPH'], data['R'])
    norm_data['BMAG'] = power_series_norm(data['BMAG'], data['R'])

    return norm_data


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


def labels_to_file(all_blocks, all_names):
    """

    Args:
        all_blocks ():
        all_names ([str]): List of names of data pertubations

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
    data.to_csv('%s/ULYS_Block_Labels.csv' % data_path)

    return


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():

    print("\nLoading data")
    data, position = load_data()

    print("\nExtracting time stamps")
    data = extract_time(data)

    print("\nMatching data")
    data = match_positions(position, data)

    print("\nNormalising data")
    norm_data = pow_normalise(data, a=5.0e5, b=5.0e4, c=220.0)

    print('\nWRITING DATA TO FILE')
    norm_data.drop(columns=['R'], inplace=True)
    norm_data.reset_index(drop=True)
    norm_data.to_csv('%s/Ulysses_PROC.csv' % data_path)

    # Create Matplotlib datetime64 type date-time column from UNIX time
    norm_data['DATETIME'] = pd.to_datetime(norm_data['UNIX TIME'], unit='s')

    # Re-index data to date-time
    norm_data.index = norm_data['DATETIME']
    del norm_data['DATETIME']

    norm_data.plot(y=['BR', 'BTH', 'BPH', 'BMAG'], kind='line')

    plt.legend(['BR', 'BTH', 'BPH', 'BMAG'], loc='upper right')
    plt.show()

    del data, norm_data

    print('\nLOADING DATA')
    data, classes = label_data('%s/Ulysses_PROC.csv' % data_path, '%s/Ulysses_Labels.csv' % data_path,
                               resample='2S', mode='up')

    print(data)

    data.plot(y=['BR', 'BTH', 'BPH', 'BMAG'], kind='line')

    plt.legend(['BR', 'BTH', 'BPH', 'BMAG'], loc='upper right')
    plt.show()

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
    blocks_to_images(blocks, perturb_names[0], block_path)

    print('\t-MIRRORED DATA')
    blocks_to_images(mir_blocks, perturb_names[1], block_path)

    print('\t-REVERSED DATA')
    blocks_to_images(rev_blocks, perturb_names[2], block_path)

    print('\t-MIRRORED AND REVERSED DATA')
    blocks_to_images(mir_rev_blocks, perturb_names[3], block_path)

    print('\nEXPORTING LABELS TO FILE')
    labels_to_file((blocks, mir_blocks, rev_blocks, mir_rev_blocks), perturb_names)

    print('\nFINISHED')


if __name__ == '__main__':
    main()
