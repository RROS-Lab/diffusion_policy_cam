import pandas as pd
import regex as re
import numpy as np
import os


# Range of x,y,z values of markers of interest
z = 0.07
# markers of interest wrt to world frame
REF_FRAME = 150

B_MOIs = {"names": ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3'],
         "pos": [ (-0.16, -0.08), 
                  (-0.16, 0.0), 
                  (-0.16, 0.08),
                  (-0.0, -0.08),
                  (-0.0, 0.0),
                  (-0.0, 0.08),
                  (0.15, -0.08),
                  (0.15, 0.0),
                  (0.15, 0.07)],
        "tolerance": [0.01, 0.01, 0.01]}
        
G_MOIs = {"names": ['GS'], 
          "pos" : (0.0, 0.05, 0.01),
          "tlnc": {"x": 0.005, "y": 0.06, "z": 0.005}}


RigidBody_OI = { 'RigidBody', 'chisel', 'gripper'} # RIGID BODY OF INTEREST #oldname: RigidBody

def filter_W_MOIs(markers_dict: dict, W_MOIs: dict) -> dict:
    filtered_markers = {}

    for name, pos in zip(W_MOIs["names"], W_MOIs["pos"]):
        tolerance = W_MOIs["tolerance"]
        for marker_name, marker_pos in markers_dict.items():
            if _is_mk_within_box(marker_pos, pos, tolerance):
                filtered_markers[name] = marker_pos
                
    return filtered_markers


def _pre_process(csv_path:str) -> tuple[pd.DataFrame, str]:
    _data = pd.read_csv(csv_path)
    FPS = _data.columns[9]
    _data = _data.reset_index(drop=False)
    indices_drop = [0, 2]
    _data = _data.drop(indices_drop).reset_index(drop=True)
    return _data, FPS 

def _get_items_of_interest(Names:np.ndarray, Rot_Pos:np.ndarray, TxyzQwxyz:np.ndarray) -> dict:
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
        _DOI["rb"][rb_name] = rb_data
    
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


def main(csv_path:str, save_path:str) -> None:
    '''
    This function is used to clean the motive file for the chizel task
    '''
    data, FPS = _pre_process(csv_path)

    row1_Name = data.iloc[0,:].values # Name of Object of Interest
    row2_Rot_Pos = data.iloc[1,:].values # Rotation or Position
    row3_TxyzQwxyz = data.iloc[2].values # X/Y/Z/w/x/y/z

    items_dict = _get_items_of_interest(row1_Name, row2_Rot_Pos, row3_TxyzQwxyz)
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

    
    mk_to_filter_dict = {name: DOI_dict['mk'][name].iloc[REF_FRAME].values for name in DOI_dict['mk']}
    
    

    filtered_markers = filter_W_MOIs(mk_to_filter_dict, B_MOIs)

    print()
    # # Continue with any additional processing and saving of cleaned data
    # cleaned_data = pd.concat([DOI_dict["times"]] +
    #                          [DOI_dict["rb"][name] for name in DOI_dict["rb"]] +
    #                          [DOI_dict["mk"][name] for name in DOI_dict["mk"]], axis=1)
    
    # cleaned_data.to_csv("cleaned_data.csv", index=False)




    #filter and assign the markers of interest based on W_MOIs
    #if the marker is within the tolerance of the W_MOIs then assign the name of the marker to the W_MOIs




    

    # Filter columns
    # filtered_columns = [col for col in _data.columns if any(keyword in col for keyword in RigidBody)]
    # # Create new DataFrame with filtered columns
    # rigid_data = _data[filtered_columns]
    # # Step 1: Identify rows with NaN values
    # nan_rows = rigid_data.isna().any(axis=1)
    # # Step 2: Get indexes of rows with NaN values
    # indexes_with_nan = nan_rows[nan_rows].index.to_list()
    # _data = _data.drop(indexes_with_nan)
    # _data = _data.reset_index(drop=True)

    #########################################################################
    ####### NEED TO ADD INTERPOLATION FUNCTIONALITY HERE FOR MARKERS ########
    #########################################################################




    #########################################################################

    '''Changing to robodk frame First time for the rigid body and markers'''
    '''Addding frame information, time information and sorting according to X,Y,Z,w,x,y,z with respect to robodk frame'''
    _SUP_HEADER_ROW = (['Time_stamp']+["RigidBody"] * len(RigidBody) * _params['RigidBody']['len'] + ["Marker"] * len(Marker) * _params['Marker']['len'])
    _rb_col_names = [f"{rb}_{axis}" for rb in RigidBody for axis in _params['RigidBody']['dof']]
    _mk_col_names = [f"{mk}_{axis}" for mk in Marker for axis in _params['Marker']['dof']]
    _HEADER_ROW = ['Time'] +_rb_col_names + _mk_col_names
    _data = _data.reindex(columns=_HEADER_ROW)
    state_dict = {_rb: _params['RigidBody']['len'] * i + 1 for i, _rb in enumerate(RigidBody, start=1)}
    add_row = pd.DataFrame([pd.Series(["FPS", rate] + [np.nan] * (len(_data.columns) - 2), index=_data.columns, dtype=str)], columns=_data.columns)


    ######## ADDING THE ITEM ROW TO THE DATAFRAME ########
    item_row = pd.DataFrame(np.reshape(_HEADER_ROW, (1, -1)), columns=_data.columns)
    


    ############ CAhanging to robodk frame for the rigid body and markers
    for rb in combined_set:
        rb_columns = [col for col in _data.columns if col.startswith(rb)]
        sorted_columns = sorted(rb_columns, key=lambda x: x.split('_')[1])
        if len(rb_columns) == 3:
            _data[sorted_columns] = np.apply_along_axis(rma.motive_2_robodk_marker, 1, _data[sorted_columns].values.astype(float))
        else:
            _data[sorted_columns] = np.apply_along_axis(rma.motive_2_robodk_rigidbody, 1, _data[sorted_columns].values.astype(float))

    _data = pd.concat([_data.iloc[:0], add_row, item_row, _data.iloc[0:]], ignore_index=True) ## add_row is the FPS row and item_row is the header row
    _data = _data.reset_index(drop=True)
    _data.columns = _SUP_HEADER_ROW


    offset = 0
    ######## ADDING THE STATE INFORMATION TO THE DATAFRAME ########
    for key , vals in state_dict.items():
        add_col = pd.DataFrame(np.reshape(([np.nan] + [key + '_state'] + [-1] * (len(_data) - 2)), (-1, 1)), columns=['RigidBody'])
        _data.insert(loc=vals + offset ,column = 'RigidBody', value=add_col, allow_duplicates=True)
        offset += 1

    _data.to_csv(f'{file_path}', index=False)


if __name__ == '__main__':
    main('diffusion_pipline/data_chisel_task/test_128_raw.csv', 
         'cleaned_data.csv')