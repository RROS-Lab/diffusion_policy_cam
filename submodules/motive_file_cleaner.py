import pandas as pd
import regex as re
import numpy as np
import submodules.robomath_addon as rma
import submodules.robomath as rm
import submodules.data_filter as _df
import os


def compare_values(val1, val2, tol):
    return all((np.abs(a - b) <= t).all() for a, b, t in zip(val1, val2, tol))


def find_similar_values_across_all(dicts, tolerance):
    similar_keys = {}

    for key1, val1 in dicts[0].items():
        for key2, val2 in dicts[1].items():
            if compare_values(val1, val2, tolerance):
                for key3, val3 in dicts[2].items():
                    if compare_values(val1, val3, tolerance):
                        similar_keys[tuple(val1)] = (key1, key2, key3)
    
    return similar_keys


def filter_MOIs(markers_dict: dict, MOIs: dict) -> dict:
    filtered_markers = {}

    for name, pos in zip(MOIs["names"], MOIs["pos"]):
        tolerance = MOIs["tolerance"]
        for marker_name, marker_pos in markers_dict.items():
            if _is_mk_within_box(marker_pos, pos, tolerance):
                filtered_markers[name] = marker_name
                
    return filtered_markers


def _get_marker_wrt_item(marker: dict, item_XYZwxyz: list) -> dict:
    unlabel_vector = {}

    for name, XYZ in marker.items():
        # unlabel_vector[name] = np.round(rma.Vxyz_wrt_TxyzQwxyz(XYZ, item_XYZwxyz), 2)
        unlabel_vector[name] = rma.Vxyz_wrt_TxyzQwxyz(XYZ, item_XYZwxyz)
        
    return unlabel_vector


def _get_data_dictionary(csv_path: str, RigidBody_OI: list) -> dict:

    data, FPS = _pre_process(csv_path)

    row1_Name = data.iloc[0,:].values # Name of Object of Interest
    row2_Rot_Pos = data.iloc[1,:].values # Rotation or Position
    row3_TxyzQwxyz = data.iloc[2].values # X/Y/Z/w/x/y/z

    items_dict = _get_items_of_interest(row1_Name, row2_Rot_Pos, row3_TxyzQwxyz, RigidBody_OI)
    data = data.drop([0,1,2]).reset_index(drop=True)
    DOI_dict = _get_data_of_interest(data, items_dict)

    # Collect all invalid row indices
    invalid_rows = set()
    for _rb_name in DOI_dict['rb']:
        invalid_rows.update(_get_invalid_rows(DOI_dict['rb'][_rb_name]))
        
        
    # Drop invalid rows from all DataFrames
    for key in DOI_dict['rb']:
        DOI_dict['rb'][key] = DOI_dict['rb'][key].drop(invalid_rows).reset_index(drop=True)

    for key in DOI_dict['mk']:
        DOI_dict['mk'][key] = DOI_dict['mk'][key].drop(invalid_rows).reset_index(drop=True)

    DOI_dict['times'] = DOI_dict['times'].drop(invalid_rows).reset_index(drop=True)
    
    return DOI_dict, FPS


def _pre_process(csv_path:str) -> tuple[pd.DataFrame, str]:
    _data = pd.read_csv(csv_path)
    FPS = _data.columns[7]
    _data = _data.reset_index(drop=False)
    indices_drop = [0, 2]
    _data = _data.drop(indices_drop).reset_index(drop=True)
    return _data, FPS 


def _get_items_of_interest(Names:np.ndarray, Rot_Pos:np.ndarray, TxyzQwxyz:np.ndarray, RigidBody_OI: list) -> dict:
    _clms = {"rb": {}, "mk": {}, "times":1} # {"rb": {"name": {"X":,... "w":,..}}, "mk": {"name": {"X":, "Y":, "Z": }}}

    for index, (name , rot_or_pos, XYZwzyz) in enumerate(zip(Names, Rot_Pos, TxyzQwxyz)):
        name = str(name); rot_or_pos = str(rot_or_pos)
        # get all rigid body data if name starts with RigidBody_OI and not a marker
        _type = None
        if name.startswith('Unlabeled'): 
            _type = 'mk'
        if any(name.startswith(rb) for rb in RigidBody_OI) and 'Marker' not in name:
            _type = 'rb'
        if _type is None:
            continue
        
        if _clms[_type].get(name) is None:
            _clms[_type][name] = {}
        if _type == 'rb' and rot_or_pos == 'Rotation':
            XYZwzyz = XYZwzyz.lower()
            
        _clms[_type][name][XYZwzyz] = index
    return _clms


def _get_data_of_interest(_data: pd.DataFrame, _clms: dict) -> dict:
    _DOI = {"rb": {}, "mk": {}, "times": None}

    rb_order = ['X', 'Y', 'Z', 'w', 'x', 'y', 'z']
    mk_order = ['X', 'Y', 'Z']
    
    times_data = _data.iloc[:, _clms["times"]].to_frame()
    times_data.columns = ['Time']
    _DOI["times"] = times_data
    
    # Process rigid bodies
    for rb_name, indices in _clms["rb"].items():
        sorted_indices = [indices[key] for key in rb_order]
        rb_data = _data.iloc[:, sorted_indices]
        #convert rb_data to float
        rb_data = rb_data.astype(float)
        rb_data.columns = [key for key in rb_order]
        _DOI["rb"][rb_name.split('_')[0]] = rb_data
    
    # Process markers
    for mk_name, indices in _clms["mk"].items():
        sorted_indices = [indices[key] for key in mk_order]
        mk_data = _data.iloc[:, sorted_indices]
        #convert mk_data to float
        mk_data = mk_data.astype(float)
        mk_data.columns = [key for key in mk_order]
        _DOI["mk"][mk_name] = mk_data

    return _DOI

def _get_invalid_rows(data_frame: pd.DataFrame) -> pd.Index:
    """
    Identifies rows with NaN, Inf, or empty values in the DataFrame.
    """
    invalid_rows = data_frame.isnull().any(axis=1) | data_frame.isin([np.inf, -np.inf]).any(axis=1) | data_frame.isin([""]).any(axis=1)
    return data_frame[invalid_rows].index

def _is_mk_within_box(marker: tuple[3], center: tuple[3], tolerance: tuple[3]) -> bool:
    '''This function is used to check if the marker is within the box'''
    return all(abs(marker[i] - center[i]) <= tolerance[i] for i in range(3))

def _get_marker_limit(dir_path: str, RigidBody_OI: list, Body_type: str, Body_OI: str, REF_FRAME: int, tolerance: list, marker_label: list, cross_ref_limit: int) -> dict:
    '''
    This function is used to get the marker limit for the chisel task
    '''
    dicts = []
    for index, file in enumerate(os.listdir(dir_path)):
        if file.endswith('.csv') and index < cross_ref_limit:
            csv_path = os.path.join(dir_path, file)

            DOI_dict, _ = _get_data_dictionary(csv_path, RigidBody_OI)

            mk_to_filter_dict = {name: DOI_dict['mk'][name].iloc[REF_FRAME].dropna().values 
                                for name in DOI_dict['mk'] 
                                if not DOI_dict['mk'][name].iloc[REF_FRAME].isna().any()}
            
            item_pos = DOI_dict[Body_type][Body_OI].iloc[REF_FRAME].dropna().values
            markers_vectors = _get_marker_wrt_item(mk_to_filter_dict, item_pos)
            
            dicts.append(markers_vectors)
        else:
            break
    
    similar_keys = find_similar_values_across_all(dicts, tolerance)
    
    marker_limit = {"names": marker_label, "pos": [key for key in similar_keys.keys()], "tolerance": tolerance}
    
    return marker_limit


def _get_cleaned_dataframe(DOI_dict: dict, FPS:int ,RigidBody_OI: list, Marker_OI: list, _params: dict) -> pd.DataFrame:
    '''
    This function is used to get the cleaned dataframe
    Changing to robodk frame First time for the rigid body and markers'''
    
    df = pd.concat(
        [DOI_dict["times"]] +
        [_df.rename_columns_with_prefix(DOI_dict["rb"][name], name) for name in DOI_dict["rb"]] +
        [_df.rename_columns_with_prefix(DOI_dict["mk"][name], name) for name in DOI_dict["mk"]],
        axis=1
    )
    _SUP_HEADER_ROW = (['Time_stamp']+["RigidBody"] * len(RigidBody_OI) * _params['RigidBody']['len'] + ["Marker"] * len(Marker_OI) * _params['Marker']['len'])
    _HEADER_ROW = df.columns
    add_row = pd.DataFrame([pd.Series(["FPS", FPS] + [0] * (len(df.columns) - 2), index=df.columns, dtype=str)], columns=df.columns)

    ######## ADDING THE ITEM ROW TO THE DATAFRAME ########
    item_row = pd.DataFrame(np.reshape(_HEADER_ROW, (1, -1)), columns=df.columns)
    combined_set = RigidBody_OI + Marker_OI

    for rb in combined_set:
        rb_columns = [col for col in df.columns if col.startswith(rb)]
        if len(rb_columns) == 3:
            df[rb_columns] = np.apply_along_axis(rma.motive_2_robodk_marker, 1, df[rb_columns].values.astype(float))
        else:
            df[rb_columns] = np.apply_along_axis(rma.motive_2_robodk_rigidbody, 1, df[rb_columns].values.astype(float))
            
    df = pd.concat([df.iloc[:0], add_row, item_row, df.iloc[0:]], ignore_index=True) ## add_row is the FPS row and item_row is the header row
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.columns = _SUP_HEADER_ROW

    return df

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
####### MAIN FUNCTION BELLOW ##########################################################

def motive_chizel_task_cleaner(csv_path:str, save_path:str, RigidBody_OI: list, Marker_OI: list, _params: dict, REF_FRAME: int, B_MOIs: dict) -> None:
    '''
    dir_path: str
        path to the csv files
    save_path: str
        path to save the cleaned csv file

    create a new csv file with the cleaned data in robodk frame

    '''
    save_file = re.sub(r'\.csv', '_cleaned.csv', csv_path)
    file_path = os.path.join(save_path, save_file)

    DOI_dict, FPS = _get_data_dictionary(csv_path, RigidBody_OI)

    mk_to_filter_dict = {name: DOI_dict['mk'][name].iloc[REF_FRAME].dropna().values 
                        for name in DOI_dict['mk'] 
                        if not DOI_dict['mk'][name].iloc[REF_FRAME].isna().any()}
    
    battery_pos = DOI_dict['rb']['battery'].iloc[REF_FRAME].dropna().values
    sheet_markers_vectors = _get_marker_wrt_item(mk_to_filter_dict, battery_pos)
    gripper_pos = DOI_dict['rb']['gripper'].iloc[REF_FRAME].dropna().values
    gripper_marker_vectors = _get_marker_wrt_item(mk_to_filter_dict, gripper_pos)
    
    # filter_labels = filter_MOIs(sheet_markers_vectors, B_MOIs) | filter_MOIs(gripper_marker_vectors, G_MOIs)
    filter_labels = filter_MOIs(sheet_markers_vectors, B_MOIs)
    
    if len(filter_labels) == len(Marker_OI):
        print(f"File: {save_file} has all markers")
    elif len(filter_labels) > len(Marker_OI):
        print(f"File: {save_file} has extra markers")
        print("Stopping execution......................")
        return  # Exit the function
    else:
        print(f"File: {save_file} has missing markers")
        print("Stopping execution......................")
        return  # Exit the function
    
    # Create a set of keys to remove
    keys_to_remove = set(DOI_dict['mk'].keys()) - set(filter_labels.values())

    # Update the dictionary with new keys and remove extra keys in one go
    for new_key, old_key in filter_labels.items():
        if old_key in DOI_dict['mk']:
            # print(f"Old key: {old_key}, New key: {new_key}")
            DOI_dict['mk'][new_key] = DOI_dict['mk'].pop(old_key)

    # Remove the extra keys
    for key in keys_to_remove:
        # print(f"Removing extra key: {key}")
        DOI_dict['mk'].pop(key)
    print("File path: ", save_file)
    _data = _get_cleaned_dataframe(DOI_dict, FPS, RigidBody_OI, Marker_OI, _params)

    _data.to_csv(f'{file_path}', index=False)
        

# if __name__ == "__main__":

#     cross_ref_limit = 3
#     Body_type = 'rb'
#     tolerance_sheet = [0.02, 0.02, 0.02]
#     tolerance_gripper = [0.005, 0.06, 0.005]

#     _params = {
#         'RigidBody': {'len':7,
#                     'dof': ['X', 'Y', 'Z', 'w', 'x', 'y', 'z']},
#         'Marker': {'len':3,
#                     'dof': ['X', 'Y', 'Z']}
#     }


#     Marker_OI = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5']
#     gripper_marker_name = ['GS']
#     RigidBody_OI = ['battery', 'chisel', 'gripper']
#     REF_FRAME = 100


#     dir_path = '/home/cam/Documents/scratch/diffusion_policy_cam/diffusion_pipline/data_chisel_task/raw_traj/'
#     save_path = '/home/cam/Documents/scratch/diffusion_policy_cam/diffusion_pipline/data_chisel_task/new_cleaned_old/'

#     B_MOIs = _get_marker_limit(dir_path, RigidBody_OI ,Body_type, 'battery', REF_FRAME, tolerance_sheet, Marker_OI, cross_ref_limit)
#     G_MOIs = _get_marker_limit(dir_path, RigidBody_OI ,Body_type, 'gripper', REF_FRAME, tolerance_gripper, gripper_marker_name, cross_ref_limit)

    