import pandas as pd
import regex as re


def fps_sampler(data: pd.DataFrame, target_fps:float, input_fps: float = 240.0):
    sample_size = int(input_fps / target_fps)
    _output_data = []
    #get every nth row of a dataframe
    _output_data = data.iloc[::sample_size]
    return _output_data


def extract_motive_data(csv_path:str,
                 filter_keys: dict[str, str],
                 start_frame: int, end_frame: int, 
                 fps: float = 30.0):
    '''
    Extract and filter motive data from a CSV file.

    Args:
    csv_path (str): Path to the CSV file.
    filter_keys (dict): Dictionary to filter and rename columns. Keys are new names, values are original column names.
    start_frame (int): Start frame for slicing the data.
    end_frame (int): End frame for slicing the data.
    fps (float): Target frames per second for downsampling. Default is 30.0.

    Returns:
    pd.DataFrame: Filtered and downsampled DataFrame.
    ----------------
    filter_keys = {
    'GR': 'GRIPPER_2_Rotation', 
    'GP': 'GRIPPER_2_Position',
    'SR': 'diff_scooper_2_2_Rotation',
    'SP': 'diff_scooper_2_2_Position',
    'BxR': 'box3_Rotation',
    'BxP': 'box3_Position',
    'BuR': 'bucket_SC_Rotation',
    'BuP': 'bucket_SC_Position'}

    '''
    _data  = pd.read_csv(csv_path)
    _data = _data.reset_index(drop=False)
    _data = _data.drop(index =0)
    _data = _data.drop(index =2)
    _data = _data.reset_index(drop=True)

    row1 = _data.iloc[0]
    row2 = _data.iloc[1]
    row3 = _data.iloc[2]

    combined_values = []
    for a, b, c in zip(row1, row2, row3):
        combined_values.append(str(a) + '_' + str(b) + '_' + str(c))

    _data.columns = combined_values
    _data = _data.drop(index =0)
    _data = _data.drop(index =1)
    _data = _data.drop(index =2)
    _data = _data.drop(_data.columns[:2], axis=1)
    # print(result_dict)
    _data = _data.iloc[start_frame:end_frame]
    _data = _data.dropna()
    _data = _data.reset_index(drop=True)
    _data = fps_sampler(_data, target_fps=fps, input_fps=240.0)

    useful_data = {}
    for new_name, original_name in filter_keys.items():
        if original_name in _data.columns:
            useful_data[new_name] = _data[original_name]

    return pd.DataFrame(useful_data)


def motive_csv_cleaner(csv_path:str):
    data = pd.read_csv(csv_path)
    data = data.reset_index(drop=False)
    data = data.drop(index =0)
    data = data.drop(index =2)
    data = data.reset_index(drop=True)

    row1 = data.iloc[0]
    row2 = data.iloc[1]
    row3 = data.iloc[2]

    combined_values = []
    columns_to_drop = [] 
    list = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1']
    i = -1 

    for index, (a, b, c) in enumerate(zip(row1, row2, row3)):
        if str(b) + '_' + str(c) in ('Rotation_X', 'Rotation_Y', 'Rotation_Z', 'Rotation_W'):
            if str(a).startswith('Unlabeled'):
                i += 1
                if i <= 29:
                    combined_values.append(list[int(i / 3)] + '_' + c.lower())  # Use list comprehension for efficiency
                else:
                    columns_to_drop.append(index)
            elif 'Marker' in str(a):
                columns_to_drop.append(index)
            elif 'RigidBody' in str(a):
                combined_values.append('battery_' + c.lower())
            else:
                combined_values.append(str(a).split("_")[0] + '_' + c.lower())
        else:
            if str(a).startswith('Unlabeled'):
                i += 1
                if i <= 29:
                    combined_values.append(list[int(i / 3)] + '_' + c)
                else:
                    columns_to_drop.append(index)
            elif 'Marker' in str(a):
                columns_to_drop.append(index)
            elif 'RigidBody' in str(a):
                combined_values.append('battery_' + c)
            else:
                combined_values.append(str(a).split("_")[0] + '_' + c)

    columns_to_drop.append(0)
    columns_to_drop.append(1)
    data = data.drop(data.columns[columns_to_drop], axis=1)
    data.columns = combined_values[2:]
    data = data.drop(index =0)
    data = data.drop(index =1)
    data = data.drop(index =2)
    data = data.dropna()
    data = data.reset_index(drop=True)
    data.to_csv("test.csv", index=False)
