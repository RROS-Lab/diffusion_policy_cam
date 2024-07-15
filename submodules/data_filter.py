import pandas as pd
import math
from scipy.signal import savgol_filter
import re

def fps_sampler(data: pd.DataFrame, target_fps:float, input_fps: float = 240.0):
    sample_size = int(input_fps / target_fps)
    _output_data = []
    _output_data = data.iloc[::sample_size]
    return _output_data

def distance(current, next):
    distance = math.sqrt((next[0] - current[0])**2 + (next[1] - current[1])**2 + (next[2] - current[2])**2)
    return distance

## redundant functions
def read_cleaned_data(csv_path, fps: float = 30.0, filter: bool = False, window_size: int = 15, polyorder: int = 3) -> pd.DataFrame:
    file = re.sub(r'\.csv', '_cleaned.csv', csv_path)
    data = pd.read_csv(csv_path)
    _new_data = fps_sampler(data, fps)
    if filter:
        _new_data = apply_savgol_filter(_new_data, window_size, polyorder)
    return _new_data

def apply_savgol_filter(df: pd.DataFrame, window_size: int, polyorder: int) -> pd.DataFrame:
    """
    Applies Savitzky-Golay filter to all columns in the DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame with columns to be smoothed.
    window_size (int): The length of the filter window (i.e., the number of coefficients).
                       window_size must be a positive odd integer.
    polyorder (int): The order of the polynomial used to fit the samples.
                     polyorder must be less than window_size.

    Returns:
    pd.DataFrame: DataFrame with smoothed columns.
    """
    smoothed_df = df.copy()
    for col in df.columns:
        smoothed_df[col] = savgol_filter(df[col], window_size, polyorder)
    return smoothed_df

