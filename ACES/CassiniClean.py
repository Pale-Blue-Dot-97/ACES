"""Script to load Cassini magnetometer data in from file and clean, interpolate and work on
and process ready for use as test data for ACES.py

TODO:

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pandas as pd
import numpy as np
import scipy.interpolate as ip
import datetime
import sys
import matplotlib.pyplot as plt

# =====================================================================================================================
#                                                     GLOBAL
# =====================================================================================================================
raw_data_path = 'Raw_Cassini_Data'
pos_path = 'Cassini_Trajectory'
proc_data_path = 'Processed_Cassini_Data'


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_data(data_name, pos_name):
    """Load in data from PDS archive. Column headers come from the LBL meta data

    Args:
        data_name (str): Name of file containing data
        pos_name (str): Name of file containing corresponding position data

    Returns:
        data (DataFrame): Table of the magnetometer data time-series
        data_columns ([str]): List of the header names of columns containing important data
        position (DataFrame): Table of the position of the spacecraft in R, LON, LAT

    """

    data_names = ['TIME', 'BR', 'BTH', 'BPH', 'BMAG', 'NUM_PTS']

    data = pd.read_table('%s/%s' % (raw_data_path, data_name), delim_whitespace=True, names=data_names,
                         na_values=99999.999)

    data.drop(columns=['NUM_PTS'], inplace=True)

    data_columns = ['BR', 'BTH', 'BPH', 'BMAG']

    print('Number of NaNs: %d' % data.isnull().sum().sum())

    # Removes any 'NaNs' from the DataFrame
    for i in data_columns:
        data.drop(data[data.isnull()[i]].index, inplace=True)
        data.reset_index(inplace=True, drop=True)

    position_names = ['YEAR', 'DOY', 'HR', 'MIN', 'SEC', 'X (Rs)', 'Y (Rs)', 'Z (Rs)', 'R', 'X (km/s)', 'Y (km/s)',
                      'Z (km/s)', 'Vmag (km/s)']

    position = pd.read_table('%s/%s' % (pos_path, pos_name), delim_whitespace=True,
                             names=position_names)
    position.drop(columns=['X (Rs)', 'Y (Rs)', 'Z (Rs)', 'X (km/s)', 'Y (km/s)', 'Z (km/s)', 'Vmag (km/s)'],
                  inplace=True)

    position = reformat_time(position)

    return data, data_columns, position


def reformat_time(position):
    """Re-formats the separate date and time columns in the telemetry file into PDS format timestamps

    Args:
        position (DataFrame): DataFrame containing Cassini positions with datetime in separate columns

    Returns:
        df (DataFrame): positions with datetime columns merged into datetime in PDS format

    """
    def time_together(row):
        """Takes the separate year, day of year, hour, minute and seconds values in a row and sticks them together
        into a PDS format timestamp

        Args:
            yr (float):
            doy (float):
            hr (float):
            minutes (float):
            sec (float):

        Returns:
            datetime (str): PDS format timestamp
        """

        date_time = datetime.datetime.strptime('%s %s %s %s %s' % (int(row['YEAR']), int(row['DOY']), int(row['HR']),
                                                                   int(row['MIN']), int(row['SEC'])),'%Y %j %H %M %S')

        return date_time.strftime('%Y-%m-%dT%H:%M:%S.%f')

    df = position.copy()

    df['TIME'] = df.apply(time_together, axis=1)

    df.drop(columns=['YEAR', 'DOY', 'HR', 'MIN', 'SEC'], inplace=True)

    return df


def extract_time(data):
    """Extracts the time since UNIX Epoch from time-stamp strings

    Args:
        data (DataFrame): Data containing time-stamp column to extract from

    Returns:
        new_data (DataFrame): Data with column containing times since UNIX Epoch
    """

    def get_UNIX_time(row):
        dt = datetime.datetime.strptime(row['TIME'], '%Y-%m-%dT%H:%M:%S.%f')
        return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

    new_data = data.copy()

    print('\nExtracting UNIX time from time-stamps')

    new_data['UNIX TIME'] = new_data.apply(get_UNIX_time, axis=1)

    # Resets index after indices have been dropped to avoid key errors
    new_data.reset_index(drop=True)

    return new_data


def interpolate_positions(positions, data):
    """Interpolates positional data to match time intervals of main data

    Args:
        positions (DataFrame): Positions of the spacecraft
        data (DataFrame): Magnetometer data

    Returns:
        new_data (DataFrame): data with the addition of the positions dataframe interpolated to match 1.92s intervals

    """

    new_data = data.copy()

    positions = extract_time(positions)

    print('\nInterpolating positional data to match time intervals of meter data')

    R = ip.interp1d(x=positions['UNIX TIME'], y=positions['R'], bounds_error=False, fill_value='extrapolate')

    new_data['R'] = R(data['UNIX TIME'])

    return new_data


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


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():

    print("\nLoading data")
    data, data_columns, position = load_data(sys.argv[1], sys.argv[2])

    print("\nExtracting time stamps")
    data = extract_time(data)

    print("\nInterpolating data")
    data = interpolate_positions(position, data)

    print("\nNormalising data")
    norm_data = pow_normalise(data, a=4.5e4, b=0.6e3, c=400.0)

    print('\nWRITING DATA TO FILE')
    norm_data.drop(columns=['TIME', 'R'], inplace=True)
    norm_data.reset_index(drop=True)
    norm_data.to_csv('%s/%s' % (proc_data_path, sys.argv[3]))

    # Create Matplotlib datetime64 type date-time column from UNIX time
    norm_data['DATETIME'] = pd.to_datetime(norm_data['UNIX TIME'], unit='s')

    # Re-index data to date-time
    norm_data.index = norm_data['DATETIME']
    del norm_data['DATETIME']

    norm_data.plot(y=['BR', 'BTH', 'BPH', 'BMAG'], kind='line')

    plt.legend(['BR', 'BTH', 'BPH', 'BMAG'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
