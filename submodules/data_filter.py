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




def apply_savgol_filter(df: pd.DataFrame, window_size: int, polyorder: int, time_frame: bool = True) -> pd.DataFrame:
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
    for index, col in enumerate(df.columns):
        if index == 0 and time_frame:
            continue
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

def episode_splitter(data_frame: pd.DataFrame, episode_length: Union[list, np.array]) -> dict:
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

def episode_combiner(item_data: dict[str: dict[str: list]], item_name: list ) -> tuple[dict[str: list], dict[str: list]]:
    """
    Args : dict of np.arryas
    Returns a single data frame with all the data frames combined and there indexes
    """
    # Initialize a new dictionary to store concatenated values and their indices
    concatenated_dict = {key: [] for key in item_name}

    # Initialize a dictionary to store indices
    index_dict = {key: [] for key in item_name}

    # Iterate through the outer dictionary keys
    for outer_key in item_data.keys():
        # Iterate through the keys 'battery', 'gripper', 'chisel'
        for inner_key in item_name:
            
            # Extend the values for the current inner_key
            concatenated_dict[inner_key].extend(item_data[outer_key][inner_key])
            
            # Store the indices along with the original key
            index_dict[inner_key].append(len(concatenated_dict[inner_key]))

    return concatenated_dict, index_dict
 

def state_to_velocity(data_state_dict: dict[str :np.array], data_time: np.array, ignore_item: list) -> dict[str :list]:
    """
    Args : dict of np.arryas, list of np vaules, ignore_item list
    Returns a single dict with there velocity
    """

    data_velocity_dict = {}

    for key in data_state_dict.keys():
        if key not in ignore_item:
            data_velocity_dict[key] = np.zeros_like(data_state_dict[key])
            for i in range(1, len(data_time)):
                data_velocity_dict[key][i] = (data_state_dict[key][i] - data_state_dict[key][i-1]) / (data_time[i] - data_time[i-1])
            velocity_data = pd.DataFrame(data_velocity_dict[key], columns = [f'{key}_X', f'{key}_Y', f'{key}_Z', f'{key}_x', f'{key}_y', f'{key}_z'])
            filtered_velocity = apply_savgol_filter(velocity_data, window_size = 15, polyorder = 3, time_frame= False)
            data_velocity_dict[key] = filtered_velocity.values
        else:
            data_velocity_dict[key] = data_state_dict[key]

    return data_velocity_dict


def velocity_to_state(data_state_dict: dict[str :np.array], data_time: np.array, ignore_item: list) -> dict[str :list]:
    """
    Args : dict of np.arryas, list of np vaules, ignore_item list
    Returns a single dict with there velocity
    """

    data_velocity_dict = {}

    for key in data_state_dict.keys():
        if key not in ignore_item:
            data_velocity_dict[key] = np.zeros_like(data_state_dict[key])
            for i in range(1, len(data_time)):
                data_velocity_dict[key][i] = (data_state_dict[key][i] - data_state_dict[key][i-1]) / (data_time[i] - data_time[i-1])
            velocity_data = pd.DataFrame(data_velocity_dict[key], columns = [f'{key}_X', f'{key}_Y', f'{key}_Z', f'{key}_x', f'{key}_y', f'{key}_z'])
            filtered_velocity = apply_savgol_filter(velocity_data, window_size = 15, polyorder = 3, time_frame= False)
            data_velocity_dict[key] = filtered_velocity.values
        else:
            data_velocity_dict[key] = data_state_dict[key]

    return data_velocity_dict

def rename_columns_with_prefix(df, prefix):
    return df.rename(columns=lambda x: f"{prefix}_{x}")


def trim_lists_in_dicts(dicts):
    # Step 1: Determine the minimum number of lists across all dictionaries
    min_len = min(len(v) for d in dicts for v in d.values())

    # Step 2: Trim the lists in each dictionary to this minimum number
    trimmed_dicts = []
    for d in dicts:
        trimmed_dict = {}
        for k, v in d.items():
            trimmed_dict[k] = v[:min_len]
        trimmed_dicts.append(trimmed_dict)
    
    return trimmed_dicts