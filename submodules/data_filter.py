import pandas as pd
import numpy as np
import re
from scipy.signal import savgol_filter
from typing import Union

def fps_sampler(data: pd.DataFrame, target_fps:float, input_fps: float) ->pd.DataFrame:
    """
        Downsample the data to the target_fps.
    """
    sample_size = int(input_fps / target_fps)
    _output_data = []
    _output_data = data.iloc[::sample_size]
    _output_data = _output_data.reset_index(drop=True)
    return _output_data

def distance(current : np.array, next: np.array) -> float:
    """
    Calculate the distance between two points.
    """
    distance = np.linalg.norm(current - current)
    return distance


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
        smoothed_df[col].iloc[1:] = savgol_filter(df[col].iloc[1:], window_size, polyorder)
    return smoothed_df

def axis_transformation(data_frame: pd.DataFrame, transformation: dict) -> pd.DataFrame:
    """
    Args : data frame
    transformation : dict (mapping of the columns to be transformed) like {'x': 'z', 'y': 'x', 'z': 'y'}
    where key is the original column and value is the new column name
    Transforms the data from the motive axis to the Any axis but jsut x,y,z

    """
    data = data_frame.copy()
    colls = data.columns
    data.columns = data.iloc[0]
    new_cols = {}


    # Loop through each column name
    for col in data.columns:
        # Regular expression to match and replace patterns
        match = re.search(r'_(.)$', col)
        if match:
            suffix = match.group(1)
            if suffix.lower() in transformation:
                new_suffix = transformation[suffix.lower()]
                # Preserve the original case of the suffix
                new_suffix = new_suffix.upper() if suffix.isupper() else new_suffix.lower()
            else:
                new_suffix = suffix  # If no match, keep the original suffix
            
            # Create new column name with modified suffix
            new_col = re.sub(r'_(.)$', r'_' + new_suffix, col)
            new_cols[col] = new_col

    # Rename the columns
    data.rename(columns=new_cols, inplace=True)
    data.iloc[0] = data.columns
    data.columns = colls

    return data

def indexer(data_frame: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    """
    Args : data frame
    start_frame : int
    end_frame : int
    Returns the data frame with the rows from start_frame to end_frame
    """
    data = data_frame.copy()
    data = data.iloc[start_frame:end_frame]
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data

def episode_splitter(data_frame: pd.DataFrame, episode_length: Union(list, np.array)) -> dict:
    """
    Args : data frame
    episode_length : int
    Returns a list of data frames with each data frame having the length of episode_length
    """
    data = data_frame.copy()
    episodes = {}
    for i in range(len(episode_length)-1):
        episodes[f'Traj_{i}'](data.iloc[episode_length[i]:episode_length[i+1]])

    return episodes

def episode_combiner(data_frames: dict) -> (list , list):
    """
    Args : dict of data frames
    Returns a single data frame with all the data frames combined
    """
    combined_data = []
    indexes = []
    cumulative_count = 0

    # Loop through each key (assuming data_frames is a dictionary)
    for key in data_frames.keys():
        combined_data.append(data_frames[key].values)  # Append data as numpy array
        cumulative_count += len(data_frames[key])
        # Store cumulative count as index
        indexes.append(cumulative_count) 
    combined_data = np.concatenate(combined_data, axis=0)

    return combined_data, indexes
 