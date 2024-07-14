import pandas as pd
import submodules.robomath_addon as rma
import numpy as np

def fps_sampler(data: pd.DataFrame, target_fps:float, input_fps: float = 240.0):
    sample_size = int(input_fps / target_fps)
    _output_data = []
    #get every nth row of a dataframe
    _output_data = data.iloc[::sample_size]
    return _output_data


def parse_cleaned_data(file_path: str):
    pass

def extract_motive_data(file_path: str):
    pass


#TODO - temp code just to fix the mistake. However, this code should be removed.
import os
def extract_data_chisel(data_path: str):
    index = []
    CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        file_data = extract_data_temp(file_path)
        CXYZ.extend(file_data[0]); Cwxyz.extend(file_data[1])
        GXYZ.extend(file_data[2]); Gwxyz.extend(file_data[3])
        BXYZ.extend(file_data[4]); Bwxyz.extend(file_data[5])
        A1XYZ.extend(file_data[6]); A2XYZ.extend(file_data[7]); A3XYZ.extend(file_data[8])
        B1XYZ.extend(file_data[9]); B2XYZ.extend(file_data[10]); B3XYZ.extend(file_data[11])
        C1XYZ.extend(file_data[12]); C2XYZ.extend(file_data[13]); C3XYZ.extend(file_data[14])
        index.append(len(CXYZ))

    return CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ, index
    

def extract_data_temp(file_path: str):
    data = pd.read_csv(file_path)
    rigid_bodies = ["chisel", "gripper", "battery"]
    markers = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    rb_TxyzQwxyz = {}
    mk_TxyzQwxyz = {}

    for rb in rigid_bodies:
        _rb_data = data.filter(like=rb)
        _rb_X = _rb_data.filter(like='_X')
        _rb_Y = _rb_data.filter(like='_Y')
        _rb_Z = _rb_data.filter(like='_Z')
        _rb_w = _rb_data.filter(like='_w')
        _rb_x = _rb_data.filter(like='_x')
        _rb_y = _rb_data.filter(like='_y')
        _rb_z = _rb_data.filter(like='_z')
        #combine the data
        _rb_XYZ_wxyz = pd.concat([_rb_X, _rb_Y, _rb_Z, _rb_w, _rb_x, _rb_y, _rb_z], axis=1)
        rb_TxyzQwxyz[rb] = _rb_XYZ_wxyz
    
    for mk in markers:
        _mk_data = data.filter(like=mk)
        _mk_X = _mk_data.filter(like='_X')
        _mk_Y = _mk_data.filter(like='_Y')
        _mk_Z = _mk_data.filter(like='_Z')
        _mk_XYZ = pd.concat([_mk_X, _mk_Y, _mk_Z], axis=1)
        mk_TxyzQwxyz[mk] = _mk_XYZ

    #convert the data to numpy arrays
    for rb in rigid_bodies:
        _rb_np_array = rb_TxyzQwxyz[rb].to_numpy()
        _rb_np_array = np.apply_along_axis(rma.motive_2_robodk_rigidbody, 1, _rb_np_array)
        rb_TxyzQwxyz[rb] = _rb_np_array
    
    for mk in markers:
        _mk_np_array = mk_TxyzQwxyz[mk].to_numpy()
        _mk_np_array = np.apply_along_axis(rma.motive_2_robodk_marker, 1, _mk_np_array)
        mk_TxyzQwxyz[mk] = _mk_np_array

    
    _CXYZ = rb_TxyzQwxyz["chisel"][:, 0:3].tolist()
    _Cwxyz = rb_TxyzQwxyz["chisel"][:, 3:].tolist()
    _GXYZ = rb_TxyzQwxyz["gripper"][:, 0:3].tolist()
    _Gwxyz = rb_TxyzQwxyz["gripper"][:, 3:].tolist()
    _BXYZ = rb_TxyzQwxyz["battery"][:, 0:3].tolist()
    _Bwxyz = rb_TxyzQwxyz["battery"][:, 3:].tolist()
    _A1XYZ = mk_TxyzQwxyz["A1"].tolist()
    _A2XYZ = mk_TxyzQwxyz["A2"].tolist()
    _A3XYZ = mk_TxyzQwxyz["A3"].tolist()
    _B1XYZ = mk_TxyzQwxyz["B1"].tolist()
    _B2XYZ = mk_TxyzQwxyz["B2"].tolist()
    _B3XYZ = mk_TxyzQwxyz["B3"].tolist()
    _C1XYZ = mk_TxyzQwxyz["C1"].tolist()
    _C2XYZ = mk_TxyzQwxyz["C2"].tolist()
    _C3XYZ = mk_TxyzQwxyz["C3"].tolist()

    return [_CXYZ, _Cwxyz, _GXYZ, _Gwxyz, _BXYZ, _Bwxyz, _A1XYZ, _A2XYZ, _A3XYZ, _B1XYZ, _B2XYZ, _B3XYZ, _C1XYZ, _C2XYZ, _C3XYZ]

# def extract_data(file_path: str):
#     # CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

#     # headers = ['chisel_x', 'chisel_y', 'chisel_z', 'chisel_w', 'chisel_X', 'chisel_Y',
#     #         'chisel_Z', 'gripper_x', 'gripper_y', 'gripper_z', 'gripper_w', 'gripper_X',
#     #         'gripper_Y', 'gripper_Z', 'battery_x', 'battery_y', 'battery_z', 'battery_w',
#     #         'battery_X', 'battery_Y', 'battery_Z', 'A1_X', 'A1_Y', 'A1_Z', 'A2_X', 'A2_Y',
#     #         'A2_Z', 'A3_X', 'A3_Y', 'A3_Z', 'B1_X', 'B1_Y', 'B1_Z', 'B2_X', 'B2_Y', 'B2_Z',
#     #         'B3_X', 'B3_Y', 'B3_Z', 'C1_X', 'C1_Y', 'C1_Z', 'C2_X', 'C2_Y', 'C2_Z',
#     #         'C3_X', 'C3_Y', 'C3_Z']


#     data = pd.read_csv(file_path)

#     pd.concat()
    
#     return None
    # return CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ

