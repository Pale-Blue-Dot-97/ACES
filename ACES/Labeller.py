"""Script to manually label data for neural network classification

TODO:
    * Generalise for all ACES magnetometer data

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import sys

# =====================================================================================================================
#                                                     GLOBAL
# =====================================================================================================================
pd.plotting.register_matplotlib_converters()

# Header names in the labels file
header = ['CLASS', 'START', 'STOP']

# List of variables
variables = ['BR', 'BTH', 'BPH', 'BMAG']


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_data(filename, resample=None, mode=None):
    """Load in cleaned and normalised data from file

    Args:
        filename (str): Name of file to be loaded
        resample (str): Resampling frequency
        mode (str): Mode of resampling. Either up or down

    Returns:
        data (DataFrame): Table of all the cleaned and normalised data from file
        norm_data (DataFrame): Table of just the normalised data

    """

    data_names = ['BR', 'BTH', 'BPH', 'BMAG', 'UNIX TIME']

    data = pd.read_csv(filename, names=data_names, dtype=float, header=0)

    # Create Matplotlib datetime64 type date-time column from UNIX time
    data['DATETIME'] = pd.to_datetime(data['UNIX TIME'], unit='s')

    # Re-index data to date-time
    data.index = data['DATETIME']
    del data['DATETIME']

    if resample is not None:
        if mode is 'up':
            new_data = data.resample('%s' % resample).interpolate(method='time', order=2)
            return new_data

        if mode is 'down':
            new_data = data.resample('%s' % resample).mean()
            return new_data

    if resample is None:
        return data


def label_data(data_filename, labels_filename, resample=None, mode=None):
    """

    Args:
        data_filename (str): Name of file containing data
        labels_filename (str): Name of file containing labels
        resample (str): Resampling frequency
        mode (str): Mode of resampling. Either up or down

    Returns:
        labelled_data (DataFrame): Labelled data
        classes ([str]): List of class names found in labels file

    """

    # Loads in data
    data = load_data(data_filename, resample, mode)

    # Loads the start and endpoints of the labelled regions of the data
    labels = pd.read_csv(labels_filename, names=header, dtype=str, header=0, sep=',', index_col='CLASS')

    # Converts strings to datetime64 dtype
    labels['START'] = pd.to_datetime(labels['START'])
    labels['STOP'] = pd.to_datetime(labels['STOP'])

    # Creates new DataFrame with an additional column for the class labels
    labelled_data = data.copy()
    labelled_data['LABELS'] = float('NaN')

    # Dict to hold all the individual class DataFrame slices
    LABELS = {}

    # Finds class names from Labels
    classes = [item[0] for item in Counter(labels.index).most_common()]

    # Slices labels into separate DataFrames for each classification
    for classification in classes:
        LABELS[classification] = labels.loc[classification].reset_index(drop=True)

    for classification in classes:
        label_match_list = []

        class_df = LABELS[classification]

        for i in range(len(class_df)):
            # Start and endpoints for a label range
            start = class_df['START'][i]
            stop = class_df['STOP'][i]

            # Finds all data points that lie in that datetime range and adds to list
            label_match_list = label_match_list + labelled_data[start:stop].index.tolist()

        # Classifies all data points that lie in this classified event ranges
        labelled_data['LABELS'][label_match_list] = classification

    # Labels any remaining points as False
    labelled_data['LABELS'].fillna(False, inplace=True)

    plot_labelled_data(data, labels, classes)

    return labelled_data, classes


def plot_labelled_data(data, labels, classes):

    classes.append('False')

    # Plot using inbuilt Pandas function
    var_handles = data.plot(y=variables, kind='line', alpha=0.7)

    handles_dict = {}

    for i in range(len(variables)):
        handles_dict[variables[i]] = var_handles.lines[i]

    colours = {}

    cmap = plt.get_cmap('Set2')

    for i in range(len(classes)):
        colours[classes[i]] = cmap(i)

    for i in range(len(labels)):
        start = labels['START'][i]
        stop = labels['STOP'][i]
        classification = labels.index[i]
        handle = plt.axvspan(start, stop, alpha=0.5, color=colours[classification])

        if classification not in handles_dict:
            handles_dict[classification] = handle

    handles = handles_dict.values()
    leg_labels = handles_dict.keys()

    plt.legend(handles=handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(leg_labels))
    plt.xlabel('UTC')
    plt.ylabel('B')
    plt.show()


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    print('LOADING DATA AND LABELS')
    label_data(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
