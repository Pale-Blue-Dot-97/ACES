"""Script to manually label Voyager 1 Jupiter flyby magnetometer data for neural network training

TODO:
    * Import Voyager 1 data and plot against data index and datetime
    * Manually define CSCs and output to file for data labelling
    * Add further science classifications

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
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


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    # Load all data and normalised data from file in Pandas.DataFrame form
    data, norm_data = load_data()

    # Plot B_r against index
    plt.plot(norm_data['BR_norm'])
    plt.show()


if __name__ == '__main__':
    main()
