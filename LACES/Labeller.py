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
from datetime import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib.dates as dates
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

    data = pd.read_csv('VOY2_JE_PROC.csv', names=data_names, dtype=float, header=0).drop(columns=['BR', 'BTH', 'BPH', 'BMAG'])

    data.reset_index(drop=True)

    return data


def unix_to_datetime(times):
    times.to_list()

    datetimes = []

    for i in times:
        datetimes.append(datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S'))

    print(datetimes[0])

    #plotable = dates.date2num(datetimes[0])
    #print(plotable)

    return datetimes


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    # Load all data and normalised data from file in Pandas.DataFrame form
    data = load_data()

    data['DATETIME'] = pd.to_datetime(unix_to_datetime(data['UNIX TIME']))

    data.index = data['DATETIME']
    del data['DATETIME']

    ax = data.plot(y='BR_norm', kind='line')
    ax2 = ax.twiny()
    ax2.set_xticks(range(0, len(data['BR_norm']), 100000))

    plt.show()


if __name__ == '__main__':
    pd.plotting.register_matplotlib_converters()
    main()
