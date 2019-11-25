"""DataClean.py

Script to load Voyager magnetometer data in from file and clean, interpolate and work on

TODO:
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
import scipy.interpolate
import scipy.signal as sg
import datetime
import Plot2D as laplt
import MultiFig as mf
import numba


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


def extract_time(data):
    new_data = data.copy()
    times = []

    print('\nExtracting UNIX time from time-stamps')

    err = []

    for i in range(len(new_data['TIME'])):
        j = new_data['TIME'][i]

        try:
            dt = datetime.datetime.strptime(j, '%Y-%m-%dT%H:%M:%S.%f')
            t = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
            times.append(t)

        except ValueError:
            print('Exception in timestamp extraction in row %d' % i)
            print('Handling exception')
            err.append(i)

            # Handling exception by splitting into date and time components
            # then finding Unix time and adding back together
            stamp = list(j)

            date = ""
            date = date.join(stamp[:10])

            hr = ""
            hr = hr.join(stamp[11:13])

            mn = ""
            mn = mn.join(stamp[14:16])

            ss = ""
            ss = ss.join(stamp[17:])

            date_time = datetime.datetime.strptime(date, '%Y-%m-%d')

            time_time = datetime.timedelta(hours=int(hr), minutes=int(mn), seconds=float(ss))

            dt = date_time + time_time
            t = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
            times.append(t)

    print('\nTotal number of exceptions: %s' % len(err))

    # Adds UNIX time to DataFrame
    new_data['UNIX TIME'] = times

    # Resets index after indices have been dropped to avoid key errors
    new_data.reset_index(drop=True)

    return new_data, times


def calc_variances(data, data_column, peak_indices, kernel, thres):
    delete = []
    for i in peak_indices:
        if i > kernel:
            window = np.array(data[data_column][int(i - ((kernel - 1) / 2)):int(i + ((kernel - 1) / 2))])

            for j in range(len(window) - 1):
                if np.abs(window[j] - window[j + 1]) > thres:
                    for k in np.arange(i - ((kernel - 1) / 2), i + ((kernel - 1) / 2), 1):
                        delete.append(k)

    return delete


def find_dodgy_data(data, data_columns, det_kernel, thres_kernel, thres):
    """Function to find non-physical data points and remove them

    Args:
        data (DataFrame): Data to be cleaned of non-physical data
        data_columns ([str]): List of the heading names of columns containing data in the DataFrame
        det_kernel ([int]): Kernel size for the detection of local maxima/ minima. Must be a positive odd integer
        thres_kernel ([int]): List of kernel sizes to pass over data (must be an integer odd number)
        thres (float): Fraction of the global min-max range to use as the threshold to determine if a point
                       is non-physical

    Returns:


    """
    data = data.copy()
    deleted = []

    for i in data_columns:
        print('Cleaning %s' % i)
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        max_var = thres * np.abs(data_max - data_min)

        loc_max = []
        loc_min = []

        for j in det_kernel:
            maxima = sg.argrelmax(np.array(data[i]), order=j)
            minima = sg.argrelmin(np.array(data[i]), order=j)

            for k in maxima[0]:
                if k not in loc_max:
                    loc_max.append(k)

            for k in minima[0]:
                if k not in loc_min:
                    loc_min.append(k)

        for j in thres_kernel:
            print('Kernel pass: %d' % j)
            delete = calc_variances(data, i, loc_max, j, max_var) + calc_variances(data, i, loc_min, j, max_var)
            for k in delete:
                if k not in deleted:
                    deleted.append(k)

    print('\nNow Deleting non-physical points')
    print('This make take some time, please be patient!')

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

    data, time = extract_time(data)

    raw_data = data.copy()

    print('Size of raw data: %d' % len(raw_data))

    print('\nFirst removing non-physical data via local extrema')

    cleaned_data = find_dodgy_data(data, ['BR', 'BTH', 'BPH', 'BMAG'], (3, 5, 10, 15), (3, 5, 7, 9, 11, 19), 0.01)

    print('Size of cleaned data: %d' % len(cleaned_data))

    cldt = cleaned_data.copy()

    print('\nCleaning data via median filter')

    #med_data = medfilt_data(cleaned_data,  ['BR', 'BTH', 'BPH'], 5)

    #print('Size of filtered data: %d' % len(med_data))

    print('\nCREATING FIGURE')

    laplt.create_figure(y=[raw_data['BR'], cldt['BR']],
                        x=[raw_data['UNIX TIME'], cldt['UNIX TIME']], LWID=[1],
                        figure_name='raw_vs_filtered.png', COLOURS=['r', 'b', 'g'], POINTSTYLES=['-'],
                        DATALABELS=['Raw Data', 'Cleaned Data'], x_label='UNIX Time (ms)',
                        y_label='B_r (nT)', axis_range=[time[0], time[len(time) - 1], -1000, 1000])

    mf.create_grid(y=[[raw_data['BR'], cldt['BR']], [raw_data['BTH'], cldt['BTH']], [raw_data['BPH'], cldt['BPH']],
                      [raw_data['BMAG'], cldt['BMAG']]],
                   x=[[raw_data['UNIX TIME'], cldt['UNIX TIME']], [raw_data['UNIX TIME'], cldt['UNIX TIME']],
                      [raw_data['UNIX TIME'], cldt['UNIX TIME']], [raw_data['UNIX TIME'], cldt['UNIX TIME']]],
                   shape=[[1, 2],
                          [3, 4]],
                   LWID=[[0.5]], figure_name='GridPlot.png', COLOURS=[['b', 'g']], POINTSTYLES=[['-']],
                   DATALABELS=[['Raw Data', 'Cleaned Data']], x_label='UNIX Time (ms)', y_label='B_r (nT)',
                   axis_range=[time[0], time[len(time) - 1], -1000, 1000])


if __name__ == '__main__':
    main()
