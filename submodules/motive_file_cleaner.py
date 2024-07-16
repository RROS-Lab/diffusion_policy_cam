import pandas as pd
import regex as re
import numpy as np
import submodules.data_filter as _df
import os


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
                object_values.append('RigidBody')
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
                object_values.append('RigidBody')
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
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = _df.axis_transformation(data, {'x': 'z', 'y': 'x', 'z': 'y'})
    data.to_csv(f'{file_path}', index=False)
