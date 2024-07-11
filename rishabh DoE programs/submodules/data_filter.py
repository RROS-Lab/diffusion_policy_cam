import pandas as pd
from scipy.signal import savgol_filter

def fps_sampler(data: pd.DataFrame, target_fps:float, input_fps: float = 240.0):
    sample_size = int(input_fps / target_fps)
    _output_data = []
    #get every nth row of a dataframe
    _output_data = data.iloc[::sample_size]
    return _output_data

def read_cleaned_data(csv_path, fps: float = 30.0):
    data = pd.read_csv(csv_path)
    _new_data = fps_sampler(data, fps)
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

