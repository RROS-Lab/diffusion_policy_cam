import pandas as pd
import regex as re
import numpy as np
import submodules.robomath_addon as rma
import os


def motive_chizel_task_cleaner(csv_path:str, save_path:str) -> None:
    '''
    csv_path: str
        path to the csv file
    save_path: str
        path to save the cleaned csv file

    create a new csv file with the cleaned data in robodk frame

    '''
    dict_of_lists = {}
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

    RigidBody = { 'battery', 'chisel', 'gripper'}
    Marker = { 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3'}
    combined_set = RigidBody.union(Marker)

    _params = {
        'RigidBody': {'len':7,
                    'dof': ['X', 'Y', 'Z', 'w', 'x', 'y', 'z']},
        'Marker': {'len':3,
                    'dof': ['X', 'Y', 'Z']}
    }


    name = csv_path.split('/')[-1]
    file = re.sub(r'\.csv', '_cleaned.csv', name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, file)

    _data = pd.read_csv(csv_path)
    name , rate = _data.columns[6], _data.columns[7]
    _data = _data.reset_index(drop=False)
    indices_drop = [0, 2]
    _data = _data.drop(indices_drop)
    _data = _data.reset_index(drop=True)

    row1 = _data.iloc[0]
    row2 = _data.iloc[1]
    row3 = _data.iloc[2]

    colums_val = []
    for index, (val , val1) in enumerate(zip(row1, row2)):
        if str(val).startswith('Unlabeled'):
            colums_val.append(index)
        if str(val).startswith('RigidBody') and 'Marker'  not in str(val) and 'Rotation' not in str(val1):
            colums_val.append(index)

            
    unlabled_data = _data.iloc[:,colums_val]
    for idx in range(0, len(unlabled_data.columns), 3):
        col_name = unlabled_data.iloc[0, idx]
        if col_name.startswith('RigidBody'):
            battery_coo = [float(unlabled_data.iloc[150, idx]), float(unlabled_data.iloc[150, idx + 1]), float(unlabled_data.iloc[150, idx + 2])]
        if col_name.startswith('Unlabeled'):
            dict_of_lists[col_name]  = [float(unlabled_data.iloc[150, idx]), float(unlabled_data.iloc[150, idx + 1]), float(unlabled_data.iloc[150, idx + 2])]
            
    filtered_dict = {key: [value for value in values if np.isfinite(value)] for key, values in dict_of_lists.items() if any(np.isfinite(value) for value in values)}
    matching_keys = {}
    for key in filtered_dict.keys():
        vector_ab = np.round(np.array(battery_coo) - filtered_dict[key], 2)
        for common_key, common_value in common_values.items():
            if (vector_ab[0], vector_ab[2]) == common_value:
                matching_keys[common_key] = key
                break

    combined_values = []
    columns_to_drop = [] 
    for index, (a, b, c) in enumerate(zip(row1, row2, row3)):
        if str(b) + '_' + str(c) in ('Rotation_X', 'Rotation_Y', 'Rotation_Z', 'Rotation_W'):

            if str(a) in matching_keys.values():
                combined_values.append(next((key for key, value in matching_keys.items() if value == str(a)), None) + '_' + c)
            elif'Marker' in str(a):
                columns_to_drop.append(index)
            elif 'RigidBody' in str(a):
                combined_values.append('battery_' + c.lower())
            else:
                combined_values.append(str(a).split("_")[0] + '_' + c.lower())
        else:
            if str(a) in matching_keys.values():
                combined_values.append(next((key for key, value in matching_keys.items() if value == str(a)), None) + '_' + c)
            elif 'Marker' in str(a) or 'Active' in str(a):
                columns_to_drop.append(index)
            elif 'RigidBody' in str(a):
                combined_values.append('battery_' + c)
            elif str(a) not in matching_keys.values() and str(a).startswith('Unlabeled'):
                columns_to_drop.append(index)
            else:
                if 'Time' in str(c):
                    combined_values.append('Time')
                else:
                    combined_values.append(str(a).split("_")[0] + '_' + c)
                
    columns_to_drop.append(0)
    _data = _data.drop(_data.columns[columns_to_drop], axis=1)
    _data.columns = combined_values[1:]
    _data = _data.drop([0, 1, 2, 3])
    _data = _data.dropna()
    _data = _data.reset_index(drop=True)


    _SUP_HEADER_ROW = (['Time_stamp']+["RigidBody"] * len(RigidBody) * _params['RigidBody']['len'] + ["Marker"] * len(Marker) * _params['Marker']['len'])
    _rb_col_names = [f"{rb}_{axis}" for rb in RigidBody for axis in _params['RigidBody']['dof']]
    _mk_col_names = [f"{mk}_{axis}" for mk in Marker for axis in _params['Marker']['dof']]
    _HEADER_ROW = ['Time'] +_rb_col_names + _mk_col_names
    _data = _data.reindex(columns=_HEADER_ROW)
    state_dict = {_rb: _params['RigidBody']['len'] * i + 1 for i, _rb in enumerate(RigidBody, start=1)}
    add_row = pd.DataFrame([pd.Series([name, rate] + [np.nan] * (len(_data.columns) - 2), index=_data.columns, dtype=str)], columns=_data.columns)
    item_row = pd.DataFrame(np.reshape(_HEADER_ROW, (1, -1)), columns=_data.columns)

    for rb in combined_set:
        rb_columns = [col for col in _data.columns if col.startswith(rb)]
        sorted_columns = sorted(rb_columns, key=lambda x: x.split('_')[1])
        if len(rb_columns) == 3:
            _data[sorted_columns] = np.apply_along_axis(rma.motive_2_robodk_marker, 1, _data[sorted_columns].values.astype(float))
        else:
            _data[sorted_columns] = np.apply_along_axis(rma.motive_2_robodk_rigidbody, 1, _data[sorted_columns].values.astype(float))

    _data = pd.concat([_data.iloc[:0], add_row, item_row, _data.iloc[0:]], ignore_index=True)
    _data = _data.reset_index(drop=True)
    # add_col = pd.DataFrame(np.reshape(np.full_like(_data.iloc[:,0], -1, dtype=int), (-1, 1)), columns=['RigidBody'])
    _data.columns = _SUP_HEADER_ROW
    offset = 0
    for key , vals in state_dict.items():
        add_col = pd.DataFrame(np.reshape(([np.nan] + [key + '_state'] + [-1] * (len(_data) - 2)), (-1, 1)), columns=['RigidBody'])
        _data.insert(loc=vals + offset ,column = 'RigidBody', value=add_col, allow_duplicates=True)
        offset += 1

    _data.to_csv(f'{file_path}', index=False)
