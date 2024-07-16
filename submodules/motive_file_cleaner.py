import pandas as pd
import regex as re
import numpy as np
from submodules import data_filter as df
import os

def motive_handover_task_cleaner(csv_path:str,
                 start_frame: int, end_frame: int) -> None:
    '''
    csv_path: str
    start_frame: int
    end_frame: int

    create a new csv file with the cleaned data
    '''
    filter_keys = {
                'GR': 'GRIPPER_2_Rotation',
                'GP': 'GRIPPER_2_Position',
                'SR': 'diff_scooper_2_2_Rotation',
                'SP': 'diff_scooper_2_2_Position',
                'BxR': 'box3_Rotation',
                'BxP': 'box3_Position',
                'BuR': 'bucket_SC_Rotation',
                'BuP': 'bucket_SC_Position'
            }
    diff = []
    mins = []
    value_to_indexes = {}
    indexes_in_list2 = []

    _data  = pd.read_csv(csv_path)
    file = re.sub(r'\.csv', '_cleaned.csv', csv_path)
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

    useful_data = {}
    for new_name, original_name in filter_keys.items():
        print(new_name)
        print(original_name)
        print(_data.columns)
        if original_name in _data.columns:
            useful_data[new_name] = _data[original_name]

    print(useful_data.keys())

    for i in range(min(len(useful_data[GR]), len(useful_data[BxR]))):
        diff_s = df.distance(useful_data[GR][i], useful_data[BxR][i])
        diff.append(diff_s)

    GA = np.full_like(diff, -1)
    min_values = sorted(set(diff))

    for index, value in enumerate(min_values):
        if value < 0.01:
            mins.append(value)
            
    # Populate the dictionary with list2 values and their indexes
    for index, value in enumerate(diff):
        if value in value_to_indexes:
            value_to_indexes[value].append(index)
        else:
            value_to_indexes[value] = [index]

    # Find indexes in list2 corresponding to values in list1
    for value in mins:
        if value in value_to_indexes:
            indexes_in_list2.extend(value_to_indexes[value])

    for i in range (min(indexes_in_list2), max(indexes_in_list2)+1):
        GA[i] = 1

    useful_data['ON/OFF'] = GA

    new_data = pd.DataFrame(useful_data)
    new_data.to_csv(f'{file}', index=False)



def motive_chizel_task_cleaner(csv_path:str, save_path:str) -> None:
    '''
    csv_path: str
    start_frame: int
    end_frame: int

    create a new csv file with the cleaned data in robodk frame

    '''
    dict_of_lists = {}
    data = pd.read_csv(csv_path , header = 1)
    name = csv_path.split('/')[-1]
    file = re.sub(r'\.csv', '_cleaned.csv', name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, file)
    data = data.drop(index =1)
    data = data.reset_index(drop=True)

    row1 = data.iloc[0]
    row2 = data.iloc[1]
    row3 = data.iloc[2]

    colums_val = []

    for index, (val , val1) in enumerate(zip(row1, row2)):
        if str(val).startswith('Unlabeled'):
            colums_val.append(index)
        if str(val).startswith('RigidBody') and 'Marker'  not in str(val) and 'Rotation' not in str(val1):
            colums_val.append(index)
            
    unlabled_data = data.iloc[:,colums_val]

    for idx in range(0, len(unlabled_data.columns), 3):
        col_name = unlabled_data.iloc[0, idx]
        if col_name.startswith('RigidBody'):
            x = float(unlabled_data.iloc[150, idx])
            y = float(unlabled_data.iloc[150, idx + 1])
            z = float(unlabled_data.iloc[150, idx + 2])
            battery_coo = [x, y, z]

        if col_name.startswith('Unlabeled'):
            x = float(unlabled_data.iloc[150, idx])
            y = float(unlabled_data.iloc[150, idx + 1])
            z = float(unlabled_data.iloc[150, idx + 2])
            point = [x, y, z]
            dict_of_lists[col_name] = point
            
    filtered_dict = {key: [value for value in values if np.isfinite(value)] for key, values in dict_of_lists.items() if any(np.isfinite(value) for value in values)}

    P = np.array(battery_coo)

    vec = {}
    for key in filtered_dict.keys():
        vector_ab = P - filtered_dict[key]
        vec[key] = np.round(vector_ab[0], 2), np.round(vector_ab[2], 2)

    common_values = {
        'A1': (-0.16, -0.08),
        'A2': (-0.16, 0.0),
        'A3': (-0.16, 0.08),
        'B1': (-0.0, -0.08),
        'B2': (-0.0, 0.0),
        'B3': (-0.0, 0.08),
        'C1': (0.15, -0.08),
        'C2': (0.15, 0.0),
        'C3': (0.15, 0.07)}

    matching_keys = {}

    for key, value in vec.items():
        for common_key, common_value in common_values.items():
            if value == common_value:
                matching_keys[common_key] = key
                break

    combined_values = []
    object_values = []
    columns_to_drop = [] 

    for index, (a, b, c) in enumerate(zip(row1, row2, row3)):
        # print(str(b) + '_' + str(c))
        if str(b) + '_' + str(c) in ('Rotation_X', 'Rotation_Y', 'Rotation_Z', 'Rotation_W'):
            if str(a) in matching_keys.values():
                object_values.append('Marker')
                combined_values.append(next((key for key, value in matching_keys.items() if value == str(a)), None) + '_' + c)
            elif'Marker' in str(a):
                columns_to_drop.append(index)
            elif 'RigidBody' in str(a):
                object_values.append('RigidBody')
                combined_values.append('battery_' + c.lower())
            else:
                object_values.append('Tool')
                combined_values.append(str(a).split("_")[0] + '_' + c.lower())
        else:
            if str(a) in matching_keys.values():
                object_values.append('Marker')
                combined_values.append(next((key for key, value in matching_keys.items() if value == str(a)), None) + '_' + c)
            elif 'Marker' in str(a) or 'Active' in str(a):
                columns_to_drop.append(index)
            elif 'RigidBody' in str(a):
                object_values.append('RigidBody')
                combined_values.append('battery_' + c)
            elif str(a) not in matching_keys.values() and str(a).startswith('Unlabeled'):
                columns_to_drop.append(index)
            else:
                object_values.append('Tool')
                combined_values.append(str(a).split("_")[0] + '_' + c)
                
    columns_to_drop.append(0)
    columns_to_drop.append(1)
    data = data.drop(data.columns[columns_to_drop], axis=1)
    data.columns = object_values[2:]
    new_row = pd.DataFrame([combined_values[2:]], columns=data.columns)
    data = pd.concat([data.iloc[:0], new_row, data.iloc[0:]], ignore_index=True)
    data = data.drop(index =1)
    data = data.drop(index =2)
    data = data.drop(index =3)
    data = data.reset_index(drop=True)
    data.to_csv(f'{file_path}', index=False)
