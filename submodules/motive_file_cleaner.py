import pandas as pd
import regex as re
import numpy as np
from itertools import combinations
import submodules.robomath_addon as rma
import submodules.robomath as rm
import submodules.data_filter as _df
import os
import string

def compare_values(val1, val2, tol):
    '''
    This function is used to compare values
    input = val1: list of values
            val2: list of values
            tol: list of tolerance
    output = boolean
    '''
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

# 
def generate_names(clusters: list) -> list:
    '''
    Function to generate cluster names
    input = clusters: list of clusters
    output = cluster_names: list of tuples containing cluster names and points
    '''
    cluster_names = []
    alphabet = string.ascii_uppercase  # Letters for naming clusters
    for cluster_index, cluster in enumerate(clusters):
        letter = alphabet[cluster_index % len(alphabet)]  # Cycle through letters if there are more clusters than letters
        for point_index, point in enumerate(cluster):
            name = f"{letter}{point_index + 1}"  # Naming each point in the cluster
            cluster_names.append((name, point))
    return cluster_names

def filter_clusters(coordinates: list, tolerance: float) -> dict:
    '''
    Function to filter clusters based on a tolerance
    input = clusters: list of clusters
    output = filtered_clusters: dictionary of filtered clusters
    '''
    clusters = []
    while coordinates:
        current_point = coordinates.pop(0)  # Remove and get the first element
        cluster = [current_point]
        remaining_points = []
        for point in coordinates:
            if abs(current_point[2] - point[2]) <= tolerance:
                cluster.append(point)
            else:
                remaining_points.append(point)
        clusters.append(cluster)
        coordinates = remaining_points  # Update coordinates to remaining points
            
    clusters = [sorted(cluster, key=lambda x: x[0]) for cluster in clusters]
    clusters = sorted(clusters, key=lambda cluster: np.mean([point[2] for point in cluster]))
        
    labeled_points = generate_names(clusters)
    
    return labeled_points
    

# def find_similar_values_across_all(dicts, tolerance):
#     '''
#     This function finds similar values across all dictionaries.
#     input = dicts: list of dictionaries
#             tolerance: list of tolerances
#     output = dictionary of similar values
#     '''
#     # Convert dictionaries to lists of values for easier comparison
#     values_lists = [list(d.values()) for d in dicts]

#     # Initialize a dictionary to store similar values
#     similar_keys = {}

#     # Compare values from the first dictionary with values from others
#     for val1 in values_lists[0]:
#         if all(any(compare_values(val1, val2, tolerance) for val2 in values_list) for values_list in values_lists[1:]):
#             # Store similar values and their corresponding keys
#             keys = tuple(d.get(tuple(val1)) for d in dicts)
#             similar_keys[tuple(val1)] = keys
    
#     return similar_keys

def filter_MOIs(markers_dict: dict, MOIs: dict) -> dict:
    '''
    This function is used to filter the markers of interest
    input = markers_dict: dictionary of markers
            MOIs: dictionary of markers of interest
    output = filtered_markers: dictionary of filtered markers
    '''
    
    filtered_markers = {}

    for name, pos in zip(MOIs["names"], MOIs["pos"]):
        tolerance = MOIs["tolerance"]
        for marker_name, marker_pos in markers_dict.items():
            if _is_mk_within_box(marker_pos, pos, tolerance):
                filtered_markers[name] = marker_name
                
    return filtered_markers


def _get_marker_wrt_item(marker: dict, item_XYZwxyz: list) -> dict:
    '''
    This function is used to get the marker wrt item
    input = marker: dictionary of markers
            item_XYZwxyz: list of item position
    output = unlabel_vector: dictionary of markers wrt item
    '''
    
    unlabel_vector = {}

    for name, XYZ in marker.items():
        # unlabel_vector[name] = np.round(rma.Vxyz_wrt_TxyzQwxyz(XYZ, item_XYZwxyz), 2)
        unlabel_vector[name] = rma.Vxyz_wrt_TxyzQwxyz(XYZ, item_XYZwxyz)
        
    return unlabel_vector


def _get_data_dictionary(csv_path: str, RigidBody_OI: list) -> dict:
    '''
    This function is used to get the data dictionary
    input = csv_path: str
            RigidBody_OI: list of rigid bodies
    output = DOI_dict: dictionary of data of interest
            FPS: int'''

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
    '''
    This function is used to pre-process the data
    input = csv_path: str
    output = _data: pandas dataframe
            FPS: str
    '''
    
    _data = pd.read_csv(csv_path)
    FPS = _data.columns[7]
    _data = _data.reset_index(drop=False)
    indices_drop = [0, 2]
    _data = _data.drop(indices_drop).reset_index(drop=True)
    return _data, FPS 




def _get_items_of_interest(Names:np.ndarray, Rot_Pos:np.ndarray, TxyzQwxyz:np.ndarray, RigidBody_OI: list) -> dict:
    '''
    This function is used to get the items of interest
    input = Names: of the items
            Rot_Pos: Rotation or Position information
            TxyzQwxyz: Rotation and Position values
            RigidBody_OI: list of rigid bodies
    output = _clms: dictionary of columns
    '''
    
    
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
    '''
    This function is used to get the data of interest
    input = _data: pandas dataframe
            _clms: dictionary of columns
    output = _DOI: dictionary of data of interest
    '''
    
    
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

def _get_marker_dict(DOI_dict: dict, REF_FRAME: int) -> dict:
    '''
    This function is used to get the marker dictionary
    '''
    
    mk_to_filter_dict = {name: DOI_dict['mk'][name].iloc[REF_FRAME].dropna().values 
                    for name in DOI_dict['mk'] 
                    if not DOI_dict['mk'][name].iloc[REF_FRAME].isna().any()}
    
    return mk_to_filter_dict


def _get_V_wrt_Item_per_file(dir_path: str, RigidBody_OI: list, Body_type: str, Body_OI: str, REF_FRAME: int, cross_ref_limit: int) -> list:
    '''
    This function is used to get the vector per file
    '''
    dicts = []
    files_used = []
    for index, file in enumerate(os.listdir(dir_path)):
        if file.endswith('.csv') and index < cross_ref_limit:
            files_used.append(file)
            
            csv_path = os.path.join(dir_path, file)

            DOI_dict, _ = _get_data_dictionary(csv_path, RigidBody_OI)

            mk_to_filter_dict = _get_marker_dict(DOI_dict, REF_FRAME)
            
            item_pos = DOI_dict[Body_type][Body_OI].iloc[REF_FRAME].dropna().values
            
            markers_vectors = _get_marker_wrt_item(mk_to_filter_dict, item_pos)
            
            dicts.append(markers_vectors)
        else:
            break
        
    print(f"Files used: {files_used}")
    return dicts

def _get_sheet_marker_limit(dir_path: str, RigidBody_OI: list, Body_type: str, Body_OI: str, 
                      REF_FRAME: int, tolerance: list, cross_ref_limit: int) -> dict:
    '''
    This function is used to get the marker limit for the chisel task
    '''
    dicts = _get_V_wrt_Item_per_file(dir_path, RigidBody_OI, Body_type, Body_OI, REF_FRAME, cross_ref_limit)
    
    similar_keys = find_similar_values_across_all(dicts, tolerance)
    similar_vectors = [key for key in similar_keys.keys()]
    labeled_points = filter_clusters(similar_vectors, 0.0095)
    
    marker_limit = {"names": [label_info[0] for label_info in labeled_points], "pos": [label_info[1] for label_info in labeled_points], "tolerance": tolerance}
    
    return marker_limit


def _get_object_marker_limit(dir_path: str, RigidBody_OI: list, Body_type: str, Body_OI: str, 
                      REF_FRAME: int, tolerance: list, marker_label: list, cross_ref_limit: int) -> dict:
    '''
    This function is used to get the marker limit of objects for the chisel task
    '''
    dicts = _get_V_wrt_Item_per_file(dir_path, RigidBody_OI, Body_type, Body_OI, REF_FRAME, cross_ref_limit)
    
    similar_keys = find_similar_values_across_all(dicts, tolerance)
    
    marker_limit = {"names": marker_label, "pos": [key for key in similar_keys.keys()], "tolerance": tolerance}
    
    return marker_limit

def _process_markers(DOI_dict, MOIs, REF_FRAME, save_file, Marker_OI):
    # Construct mk_to_filter_dict
    mk_to_filter_dict = _get_marker_dict(DOI_dict, REF_FRAME)
    
    # Initialize a set to hold all concatenated filter labels
    filter_labels_combined = {}

    # Process each type of marker ('battery', 'gripper', etc.)
    for item_key, MOI_set in MOIs.items():
        pos = DOI_dict['rb'][item_key].iloc[REF_FRAME].dropna().values
        marker_vectors = _get_marker_wrt_item(mk_to_filter_dict, pos)
        filter_labels = filter_MOIs(marker_vectors, MOI_set)
        
        # Update the combined dictionary with filter labels
        for key, labels in filter_labels.items():
            if key in filter_labels_combined:
                filter_labels_combined[key].update(labels)
            else:
                filter_labels_combined[key] = labels
        
    # Check results for current item
    if len(filter_labels_combined) == len(Marker_OI):
        print(f"File: {save_file} has all {item_key} markers")
    elif len(filter_labels_combined) > len(Marker_OI):
        print(f"File: {save_file} has extra {item_key} markers")
        print("Stopping execution......................")
        return  # Exit the function
    else:
        print(f"File: {save_file} has missing {item_key} markers")
        print("Stopping execution......................")
        return  # Exit the function
    
    return filter_labels_combined
    
    
def _get_cleaned_dataframe(DOI_dict: dict, FPS:int ,RigidBody_OI: list, Marker_OI: list, _params: dict) -> pd.DataFrame:
    '''
    This function is used to get the cleaned dataframe
    Changing to robodk frame First time for the rigid body and markers'''
    
    df = pd.concat(
        [DOI_dict["times"]] +
        [_df.rename_columns_with_prefix(DOI_dict["rb"][name], name) for name in DOI_dict["rb"]] +
        [_df.rename_columns_with_prefix(DOI_dict["mk"][name], name) for name in DOI_dict["mk"] if name in Marker_OI],
        axis=1
    )
    _SUP_HEADER_ROW = (['Time_stamp']+["RigidBody"] * len(RigidBody_OI) * _params['RigidBody']['len'] + ["Marker"] * len(Marker_OI) * _params['Marker']['len'])
    
    # # Filter columns
    # filtered_columns_rigid = [col for col in df.columns if any(keyword in col for keyword in RigidBody_OI)]
    # # Create new DataFrame with filtered columns
    # rigid_data = df[filtered_columns_rigid]
    # # Step 1: Identify rows with NaN values
    # nan_rows_rigid = rigid_data.isna().any(axis=1)
    # # Step 2: Get indexes of rows with NaN values
    # indexes_with_nan = nan_rows_rigid[nan_rows_rigid].index.to_list()
    # df = df.drop(indexes_with_nan)
    # df = df.reset_index(drop=True)

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
    # len_df = len(df)
    df = df.dropna()
    df.columns = _SUP_HEADER_ROW
    df = df.reset_index(drop=True)
    
    # if len_df - len(df_no_NAN) < 960:
    #     print("NAN values found and removed")
    #     df_no_NAN.columns = _SUP_HEADER_ROW
    #     return df_no_NAN
    
    return df

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
####### MAIN FUNCTION BELLOW ##########################################################

def motive_chizel_task_cleaner(csv_path:str, save_path:str, OI:dict, _params: dict, REF_FRAME: int, MOIs: dict) -> None:
    '''
    dir_path: str
        path to the csv files
    save_path: str
        path to save the cleaned csv file

    create a new csv file with the cleaned data in robodk frame

    '''
    save_file = re.sub(r'\.csv', '_cleaned.csv', csv_path)

    DOI_dict, FPS = _get_data_dictionary(csv_path, OI['RigidBody'])

    filter_labels = _process_markers(DOI_dict, MOIs, REF_FRAME, save_file, OI['Marker'])
    
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
    
    _data = _get_cleaned_dataframe(DOI_dict, FPS, OI['RigidBody'], OI['Marker'], _params)
    
    if _data is not False:
        _data.to_csv(f'{save_path}', index=False)
        

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

    