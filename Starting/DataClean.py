"""DataClean.py

Script to load Voyager magnetometer data in from file and clean, interpolate and work on

TODO:
    * Fix time-stamps with 60s
    * Convert to using DataLoad to load data to pandas DataFrame
    * Compute variances between points to clean non-physical data
    * Use variances to calculate transmission priorities/ compression or fitting levels
"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal as sg
import datetime
import Plot2D as laplt
import MultiFig as mf

# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================


def load_data():
    """Load in data from PDS archive. Column headers come from the LBL meta data
    """

    names = ['TIME', 'SCLK', 'MAG_ID', 'BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH',
             'RMS_BPH', 'NUM_PTS']

    data = pd.read_csv('S3_1_92S.TAB', names=names, na_values=-9999.999)

    data_columns = ['BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH', 'RMS_BPH',
                    'NUM_PTS']

    return data, data_columns


def calc_variances(data, data_column, peak_indices, kernel, thres):
    delete = []
    for i in peak_indices:
        if i > kernel:
            window = np.array(data[data_column][int(i-((kernel-1)/2)):int(i+((kernel-1)/2))])

            """
            rolls_for = [window]
            rolls_back = [window]
    
            for j in range((kernel-1)/2):
                rolls_for.append(np.roll(window, j))
                rolls_back.append(np.roll(window, -j))
    
            for j in range(len(rolls_for)):
                for k in range(len(window)):
                    if np.abs(rolls_for[j][1][k] - rolls_for[j-1][1][k]) > thres:
            """
            for j in range(len(window)-1):
                if np.abs(window[j] - window[j+1]) > thres:
                    for k in np.arange(i-((kernel-1)/2), i+((kernel-1)/2), 1):
                        print('k: %d' % k)
                        delete.append(k)

    print('Delete:')
    print(delete)
    return delete


def find_dodgy_data(data, data_columns, kernel, thres):
    """

    Args:
        data:
        data_columns:
        kernel:
        thres:

    Returns:

    """
    data = data.copy()
    deleted = []

    #kernels = np.arange(kernel, kernel + 20, 2)

    for i in data_columns:
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        max_var = thres * np.abs(data_max - data_min)

        loc_max = sg.argrelmax(np.array(data[i]), order=kernel)
        loc_min = sg.argrelmin(np.array(data[i]), order=kernel)

        delete = calc_variances(data, i, loc_max[0], kernel, max_var) + calc_variances(data, i, loc_min[0], kernel, max_var)

        print('Returned Deletes')
        print(delete)

        for j in delete:
            if j not in deleted:
                deleted.append(j)

        """
        for j in loc_max[0]:
            try:
                #print(np.abs(data[i][j] - data[i][j+1]))
                if np.abs(data[i][j] - data[i][j+1]) > max_var or np.abs(data[i][j] - data[i][j-1]) > max_var:
                    if j not in deleted:
                        deleted.append(j)
            except KeyError:
                print('Key Error in accessing entry %d' % j)

        for j in loc_min[0]:
            try:
                if np.abs(data[i][j]-data[i][j+1]) > max_var or np.abs(data[i][j] - data[i][j-1]) > max_var:
                    if j not in deleted:
                        deleted.append(j)
            except KeyError:
                print('Key Error in accessing entry %d' % j)
        """

    print('\nNow Deleting non-physical points')
    print('This make take some time, please be patient!')

    print(deleted)

    indexes_to_keep = set(range(data.shape[0])) - set(deleted)
    cleaned_data = data.take(list(indexes_to_keep))

    print('\nTotal deleted points: %s' % len(deleted))

    return cleaned_data


def medfilt_data(data, data_columns, kernel_size):
    """Applies a median filter to the data, varying kernel size from given to 20 increments more

    Args:
        data (DataFrame):
        data_columns (array-like):
        kernel_size (int):

    Returns:
        Median filtered copy of 'data'

    """

    data = data.copy()

    cleaned_data_arrays = []

    for i in data_columns:
        print('Filtering %s' % i)
        filtered = []
        for j in np.arange(kernel_size, kernel_size + 20, 2):
            for k in [1, 2, 3]:
                filtered = sg.medfilt(data[i], j)

        cleaned_data_arrays.append(filtered)
    new_data = {'UNIX TIME': data['UNIX TIME']}

    for i in range(len(data_columns)):
        new_data.update({data_columns[i]: cleaned_data_arrays[i]})

    cleaned_df = pd.DataFrame(new_data)

    return cleaned_df


def clean_data(data, data_columns, kernel_size):
    data = data.copy()
    deleted = []
    for i in data_columns:
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        max_var = 0.1 * np.abs(data_max - data_min)

        noise = sg.find_peaks(data[i], threshold=max_var, width=(1, 5), wlen=kernel_size)

        print(noise[0])

        for j in noise[0]:
            if j not in deleted:
                print('Deleted %d' % j)
                deleted.append(j)
                data.drop(data.index[j], axis=0, inplace=True)

    return data, deleted


def main():
    data, data_columns = load_data()

    time = []

    print('\nExtracting UNIX time from time-stamps')

    skipped = []

    for i in range(len(data['TIME'])):
        j = data['TIME'][i]

        try:
            dt = datetime.datetime.strptime(j, '%Y-%m-%dT%H:%M:%S.%f')
            t = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
            time.append(t)

        except ValueError:
            print('Exception in datestamp: Removing row %d' % i)
            skipped.append(i)

    print('\nTotal number of exceptions: %s' % len(skipped))

    print('\nNow removing erroneous times')
    for i in skipped:
        data.drop(data.index[i], axis=0, inplace=True)

    # Adds UNIX time to DataFrame
    data['UNIX TIME'] = time

    # Resets index after indices have been dropped to avoid key errors
    data.reset_index(drop=True)

    raw_data = data.copy()

    print('Size of raw data: %d' % len(raw_data))

    print('\nFirst removing non-physical data via local extrema')

    cleaned_data = find_dodgy_data(data, data_columns, 5, 0.1)

    print('Size of cleaned data: %d' % len(cleaned_data))

    cldt = cleaned_data.copy()

    print('\nCleaning data via median filter')

    med_data = medfilt_data(cleaned_data, data_columns, 5)

    print('Size of filtered data: %d' % len(med_data))

    print('\nCREATING FIGURE')

    laplt.create_figure(y=[med_data['BR'], raw_data['BR'], cldt['BR']],
                        x=[med_data['UNIX TIME'], raw_data['UNIX TIME'], cldt['UNIX TIME']], LWID=[1],
                        figure_name='raw_vs_filtered.png', COLOURS=['r', 'b', 'g'], POINTSTYLES=['-'],
                        DATALABELS=['Filtered Data', 'Raw Data', 'Cleaned Data'], x_label='UNIX Time (ms)',
                        y_label='B_r (nT)', axis_range=[time[0], time[len(time) - 1], -1000, 1000])


if __name__ == '__main__':
    main()
