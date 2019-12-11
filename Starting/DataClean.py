"""Script to load Voyager magnetometer data in from file and clean, interpolate and work on
and process ready for training

TODO:

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import time
import sys
import pandas as pd
import numpy as np
import scipy.interpolate as ip
import scipy.signal as sg
import datetime
import Plot2D as laplt
import MultiFig as mf
import pyttsx3 as speech

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

    data_names = ['TIME', 'SCLK', 'MAG_ID', 'BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR',
                  'RMS_BTH', 'RMS_BPH', 'NUM_PTS']

    data = pd.read_csv('S3_1_92S.TAB', names=data_names, na_values=-9999.999)

    data.drop(columns=['SCLK', 'MAG_ID', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH', 'RMS_BPH', 'NUM_PTS'],
              inplace=True)

    data_columns = ['BR', 'BTH', 'BPH', 'BMAG']

    print('Number of NaNs: %d' % data.isnull().sum().sum())

    # Removes any 'NaNs' from the DataFrame
    for i in data_columns:
        data.drop(data[data.isnull()[i]].index, inplace=True)
        data.reset_index(inplace=True, drop=True)

    position_names = ['TIME', 'R', 'LAT', 'LON', 'LOCTIME']

    position = pd.read_table('SPICE062_071.TAB', delim_whitespace=True, names=position_names, na_values=-999.999)

    return data, data_columns, position


def extract_time(data):
    """Extracts the time since UNIX Epoch from time-stamp strings

    Args:
        data (DataFrame): Data containing time-stamp column to extract from

    Returns:
        new_data (DataFrame): Data with column containing times since UNIX Epoch
        times ([float]): Array identical to new column added to data
    """

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


def interpolate_positions(positions, data):
    """Interpolates positional data to match time intervals of main data

    Args:
        positions (DataFrame): Positions of the spacecraft
        data (DataFrame): Magnetometer data

    Returns:
        new_data (DataFrame): data with the addition of the positions dataframe interpolated to match 1.92s intervals

    """

    new_data = data.copy()

    new_stamps = []

    for stamp in np.array(positions['TIME']):
        stamp_list = list(stamp)
        stamp_list.remove('Z')
        new_stamp = ""
        for i in stamp_list:
            new_stamp += i
        new_stamps.append(new_stamp)

    positions['TIME'] = new_stamps

    positions, time = extract_time(positions)

    print('\nInterpolating positional data to match time intervals of meter data')

    R = ip.interp1d(x=time, y=positions['R'], bounds_error=False, fill_value='extrapolate')
    LAT = ip.interp1d(x=time, y=positions['LAT'], bounds_error=False, fill_value='extrapolate')
    LON = ip.interp1d(x=time, y=positions['LON'], bounds_error=False, fill_value='extrapolate')

    new_data['R'] = R(data['UNIX TIME'])
    new_data['LAT'] = LAT(data['UNIX TIME'])
    new_data['LON'] = LON(data['UNIX TIME'])

    return new_data


def calc_variances(data_column, peak_indices, kernel, threshold, deleted):
    """Calculates if any of the differences between points within the kernels about the local extrema supplied cross
    a threshold. Deletes window if this is True

    Args:
        data_column ([float]): Column of data to examine
        peak_indices ([int]): Indices of local extrema identified
        kernel (int): Width of window centered on local extrema to investigate.
                      Must be a positive integer
        threshold (float): Fraction of the global min-max range to use as the threshold to determine if a point
                      is non-physical
        deleted ([int]): Current array of indexes of points to be deleted

    Returns:
        deleted ([int]): Updated array of indexes of points to be deleted

    """

    half_win = (kernel - 1) / 2

    for i in peak_indices[kernel:]:
        window = data_column[int(i - half_win):int(i + half_win)]

        for j in range(len(window) - 1):
            if np.abs(window[j] - window[j + 1]) > threshold:
                for k in np.arange(i - half_win, i + half_win, 1):
                    if k not in deleted:
                        deleted.append(k)

    return deleted


def find_dodgy_data(data, data_columns, columns_to_clean, det_kernel, thres_kernel, threshold):
    """Function to find non-physical data points and remove them

    Args:
        data (DataFrame): Data to be cleaned of non-physical data
        data_columns ([str]): List of the heading names of columns containing data in the DataFrame
        columns_to_clean ([str]): List of the names of columns of data to clean
        det_kernel (int): Kernel size for the detection of local maxima/ minima. Must be a positive odd integer
        thres_kernel ([int]): List of kernel sizes to pass over data (must be an integer odd number)
        threshold (float): Fraction of the global min-max range to use as the threshold to determine if a point
                       is non-physical

    Returns:
        cleaned_data (DataFrame): The cleaned data

    """

    data = data.copy()
    deleted = []

    loc_max = []
    loc_min = []

    print('\nFinding all local minima and maxima')

    for i in data_columns:
        maxima = sg.argrelmax(np.array(data[i]), order=det_kernel)
        minima = sg.argrelmin(np.array(data[i]), order=det_kernel)

        loc_max.extend(maxima[0])
        loc_min.extend(minima[0])

    loc_min = list(dict.fromkeys(loc_min))
    loc_max = list(dict.fromkeys(loc_max))

    for i in columns_to_clean:
        print('\nCleaning %s' % i)
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        max_var = threshold * np.abs(data_max - data_min)

        for j in thres_kernel:
            print('Kernel pass: %d' % j)
            deleted = calc_variances(np.array(data[i]), loc_max, j, max_var, deleted) \
                      + calc_variances(np.array(data[i]), loc_min, j, max_var, deleted)

    deleted = list(dict.fromkeys(deleted))

    print('\nNow deleting non-physical points')

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


def dipole(x, r, a):
    """Normalises x using simple r^-3 dipole assumption

    Args:
        x (float, Array-like): Values to be normalised
        r (float, Array-like): Distance to planet to normalise with
        a (float): Normalisation parameter

    Returns:
        Normalised data
    """

    return x / (a * np.power(r, -3))


def power_series_norm(x, r, a, b, c):
    """

    Args:
        x (float, Array-like): Values to be normalised
        r (float, Array-like): Distance to planet to normalise with
        a (float): Normalisation parameter for r^-3 term
        b (float): Normalisation parameter for r^-2 term
        c (float): Normalisation parameter for r^-1 term

    Returns:
        Normalised data
    """

    return x / (a * np.power(r, -3) + b * np.power(r, -2) + c * np.power(r, -1))


def dipole_normalise(data, a=6.0e5):
    """Normalise data using power series of position

    Args:
        data (DataFrame): DataFrame containing data to be normalised with contained position
        a (float): Scalar in power series

    Returns:
        norm_data (DataFrame): Dataframe with new columns with normalised data

    """

    norm_data = data.copy()

    # Simple dipole normalisation using r^-3
    print('\nApplying simple dipole normalisation to data')
    norm_data['BR_norm'] = dipole(data['BR'], data['R'], a)
    norm_data['BTH_norm'] = dipole(data['BTH'], data['R'], a)
    norm_data['BPH_norm'] = dipole(data['BPH'], data['R'], a)
    norm_data['BMAG_norm'] = dipole(data['BMAG'], data['R'], a)

    return norm_data


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

    norm_data = data.copy()

    # More complex polynomial approach
    print('\nApplying power series normalisation to data')
    norm_data['BR_norm'] = power_series_norm(data['BR'], data['R'], a, b, c)
    norm_data['BTH_norm'] = power_series_norm(data['BTH'], data['R'], a, b, c)
    norm_data['BPH_norm'] = power_series_norm(data['BPH'], data['R'], a, b, c)
    norm_data['BMAG_norm'] = power_series_norm(data['BMAG'], data['R'], a, b, c)

    return norm_data


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    engine = speech.init()

    engine.say("Loading data")
    engine.runAndWait()

    data, data_columns, position = load_data()

    engine.say("Extracting time stamps")
    engine.runAndWait()

    data, times = extract_time(data)

    engine.say("Interpolating data")
    engine.runAndWait()

    data = interpolate_positions(position, data)

    raw_data = data.copy()

    print('Size of raw data: %d' % len(raw_data))

    print('\nFirst removing non-physical data via local extrema')
    engine.say("Removing non-physical data")
    engine.runAndWait()

    cleaned_data = find_dodgy_data(data, data_columns, ['BR', 'BTH', 'BPH', 'BMAG'], 5, (3, 5, 9, 15), 0.01)

    print('Size of cleaned data: %d' % len(cleaned_data))

    cldt = cleaned_data.copy()

    engine.say("Normalising data")
    engine.runAndWait()

    norm_data = pow_normalise(cldt, a=6.0e5, b=5.0e4, c=400.0)

    engine.say("Writing data to file")
    engine.runAndWait()

    print('\nWRITING DATA TO FILE')
    norm_data.drop(columns=['TIME', 'R', 'LAT', 'LON'], inplace=True)
    norm_data.to_csv('VOY2_JE_PROC.csv')

    #print('\nCREATING FIGURE')

    # Alert bell
    for i in range(1, 4):
        sys.stdout.write('\r\a')
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write('\n')

    engine.say("Finished")
    engine.runAndWait()

    """
    mf.create_grid(y=[[norm_data['BR_norm']], [norm_data['BTH_norm']], [norm_data['BPH_norm']], [norm_data['BMAG_norm']]],
                   x=[[norm_data['UNIX TIME']], [norm_data['UNIX TIME']], [norm_data['UNIX TIME']],
                      [norm_data['UNIX TIME']]], LWID=[[0.5]], figure_name='NormData.png', COLOURS=[['b']],
                   POINTSTYLES=[['-']], DATALABELS=[['B_R'], ['B_TH'], ['B_PH'], ['B']], x_label='UNIX Time (ms)',
                   y_label='', axis_range=[times[0], times[len(times) - 1], -1, 1], shape=[[1, 2],
                                                                                           [3, 4]]
                   )
    """


if __name__ == '__main__':
    main()

