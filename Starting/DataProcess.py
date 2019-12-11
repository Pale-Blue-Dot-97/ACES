"""Script to process cleaned and normalised Voyager magnetometer data into 4-channel blocks for use as training and
validation data for neural networks

TODO:
    * Add methods to process data into 4-channel blocks for training
    * Add method to automatically label data

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
