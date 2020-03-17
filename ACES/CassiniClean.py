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
import matplotlib.pyplot as plt

# =====================================================================================================================
#                                                     GLOBAL
# =====================================================================================================================
folder = 'Cassini Data'


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

    data_names = ['TIME', 'BR', 'BTH', 'BPH', 'BMAG', 'NUM_PTS']

    data = pd.read_table('%s/06005_06036_20_FGM_KRTP_1S.DAT' % folder, delim_whitespace=True, names=data_names,
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

    position = pd.read_table('%s/Cassini_POS_2006-01-05-2006-02-05_300S.DAT' % folder, delim_whitespace=True,
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
    def time_together(yr, doy, hr, minutes, sec):
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

        date_time = datetime.datetime.strptime('%s %s %s %s %s' % (int(yr), int(doy), int(hr), int(minutes), int(sec)),
                                               '%Y %j %H %M %S')

        return date_time.strftime('%Y-%m-%dT%H:%M:%S.%f')

    df = position.copy()

    df['TIME'] = float('NaN')

    df['TIME'] = df.apply(lambda row: time_together(row['YEAR'], row['DOY'], row['HR'], row['MIN'], row['SEC']), axis=1)

    df.drop(columns=['YEAR', 'DOY', 'HR', 'MIN', 'SEC'], inplace=True)

    return df


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

    positions, time = extract_time(positions)

    print('\nInterpolating positional data to match time intervals of meter data')

    R = ip.interp1d(x=time, y=positions['R'], bounds_error=False, fill_value='extrapolate')

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
    norm_data['BR_norm'] = power_series_norm(data['BR'], data['R'])
    norm_data['BTH_norm'] = power_series_norm(data['BTH'], data['R'])
    norm_data['BPH_norm'] = power_series_norm(data['BPH'], data['R'])
    norm_data['BMAG_norm'] = power_series_norm(data['BMAG'], data['R'])

    return norm_data


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():

    print("\nLoading data")
    data, data_columns, position = load_data()

    print("\nExtracting time stamps")
    data, times = extract_time(data)

    print("\nInterpolating data")
    data = interpolate_positions(position, data)

    print("\nNormalising data")
    norm_data = pow_normalise(data, a=4.0e4, b=2.0e3, c=220.0)

    print('\nWRITING DATA TO FILE')
    norm_data.drop(columns=['TIME', 'R'], inplace=True)
    norm_data.reset_index(drop=True)
    norm_data.to_csv('%s/CASSINI_2006_01_PROC.csv' % folder)


if __name__ == '__main__':
    main()
