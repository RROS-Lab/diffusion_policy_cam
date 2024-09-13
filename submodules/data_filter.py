import pandas as pd
import numpy as np
import re
from scipy.signal import savgol_filter
from typing import Union
# import submodules.data_filter as cfp
import moveit_motion.diffusion_policy_cam.submodules.cleaned_file_parser as cfp
import moveit_motion.diffusion_policy_cam.submodules.robolink_addon as rma

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
    distance = np.linalg.norm(next - current)
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

def rename_rb_columns_with_prefix(df, prefix):
    return df.rename(columns=lambda x: f"{prefix.split('_')[-1]}_{x}")


def rename_ms_columns_with_prefix(df, prefix):
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

def find_closest_number(target, arr):
    closest = min(arr, key=lambda x: abs(x - target))
    return closest

def find_index(number, arr):
    arr = np.array(arr)
    indices = np.where(arr == number)[0]
    return indices[0] if indices.size > 0 else -1


def stop_rotation_index(path, target_fps):
    
    data = cfp.DataParser.from_quat_file(file_path = path, target_fps= target_fps, filter=False, window_size=5, polyorder=3)
    _times = data.get_time()
    battery = data.get_rigid_TxyzRxyz()['battery']
    W_T_bat = battery

    W_T_bat0 = battery[0]
    bat0_T_bat = np.apply_along_axis(rma.BxyzRxyz_wrt_AxyzRxyz, 1, W_T_bat, W_T_bat0)
    yaw = bat0_T_bat[:,5]
    yaw = savgol_filter(yaw, 100, 3)

    yaw = np.rad2deg(yaw)
    yaw_dot = rma.first_derivative(yaw, _times)

    # Find the end of rotation
    thresehold = 1
    final_yaw = yaw[-1]
    stop_index = np.argmax(np.abs(yaw_dot[::-1]) > thresehold)
    
    return stop_index, final_yaw


from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore


def filter_outliers(waypoints, threshold=2.0):
    # Calculate z-score for each column (X, Y, Z, rx, ry, rz)
    z_scores = np.abs(zscore(waypoints, axis=0))
    
    # Filter out points where any dimension's z-score exceeds threshold
    filtered_waypoints = waypoints[(z_scores < threshold).all(axis=1)]
    
    return filtered_waypoints

def filter_outliers_verbose(waypoints, threshold=2.0):
    """
    Filters outliers from the waypoints based on z-score and prints the removed points with explanation.
    
    Parameters:
    - waypoints: np.array of shape (n, 6), where n is the number of waypoints, and 6 represents [X, Y, Z, rx, ry, rz].
    - threshold: Z-score threshold for detecting outliers.
    
    Returns:
    - filtered_waypoints: np.array of waypoints with outliers removed.
    - removed_waypoints: np.array of waypoints that were removed.
    """
    waypoints = np.array(waypoints)
    z_scores = np.abs(zscore(waypoints, axis=0))
    
    # Create mask for waypoints that are not outliers
    mask = (z_scores < threshold).all(axis=1)
    
    # Separate filtered and removed waypoints
    filtered_waypoints = waypoints[mask]
    removed_waypoints = waypoints[~mask]
    
    # Verbose logging of removed waypoints
    for i, (point, z_point) in enumerate(zip(removed_waypoints, z_scores[~mask])):
        reason = []
        labels = ['X', 'Y', 'Z', 'rx', 'ry', 'rz']
        for dim, z, label in zip(point, z_point, labels):
            if z >= threshold:
                reason.append(f"{label}: z-score = {z:.2f}")
        print(f"Removed waypoint {i + 1}: {point} | Reason: {'; '.join(reason)}")
    
    return filtered_waypoints

def smooth_waypoints_gaussian(waypoints, sigma=1.0):
    # Apply Gaussian smoothing along each column (X, Y, Z, rx, ry, rz)
    smoothed_waypoints = gaussian_filter1d(waypoints, sigma=sigma, axis=0)
    return smoothed_waypoints

def smooth_waypoints_sg(waypoints, window_length=7, polyorder=3):
    """
    Applies Savitzky-Golay filter to smooth waypoints.
    
    Parameters:
    - waypoints: np.array of shape (n, 6), where n is the number of waypoints, 
                 and 6 represents [X, Y, Z, rx, ry, rz].
    - window_length: Odd integer representing the size of the window. 
    - polyorder: Degree of the polynomial to fit in each window.
    
    Returns:
    - Smoothed waypoints of the same shape as input.
    """
    # Apply SG filter for each column (X, Y, Z, rx, ry, rz)
    smoothed_waypoints = savgol_filter(waypoints, window_length=window_length, polyorder=polyorder, axis=0)
    return smoothed_waypoints