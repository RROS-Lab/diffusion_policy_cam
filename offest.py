from submodules import cleaned_file_parser as cfp
import os
import numpy as np
import re
import csv


def add_marker_offset(position: np.array ,X: int,Y: int,Z: int)-> np.array:
    position = position + np.array([X,Y,Z])
    return position

# def add_rigidbody_offset(position: np.array ,X: int,Y: int,Z: int, W: int = 0, x:int= 0, y:int= 0, z:int = 0)-> np.array:
def add_rigidbody_offset(save_type: str, position: np.array ,X: int,Y: int,Z: int, w:int = 0, x:int= 0, y:int= 0, z:int = 0)-> np.array:

    if save_type == 'EULER':
        position = position + np.array([X,Y,Z,x,y,z])
    if save_type == 'QUAT':
        position = position + np.array({X,Y,Z,w,x,y,z})

    return position

save_path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/test_eular_offset/'

path_dir = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cleaned_traj/'

X_offset = 0.1
Y_offset = 0.1
Z_offset = 0.05

for index, file in enumerate(os.listdir(path_dir)):
    if index % 3 == 0:
        X_offset = np.random.uniform(-0.3, 0.3)
        Y_offset = np.random.uniform(-0.3, 0.3)
        Z_offset = np.random.uniform(-0.05, 0.05)

    
    file_name = re.sub(r'\.csv', f'_x_{X_offset}_y_{Y_offset}_z_{Z_offset}_offset.csv', file)
    
    file_path = os.path.join(save_path, file_name)

    path_name = path_dir + file

    data = cfp.DataParser.from_quat_file(file_path = path_name, target_fps= 120, filter=False, window_size=5, polyorder=3)

    save_type='EULER'
    # add first rows
    _params = {
        'QUAT': {'len':7,
                    'dof': ['X', 'Y', 'Z', 'w', 'x', 'y', 'z'],
                    '__gettr__': data.get_rigid_TxyzQwxyz },
        'EULER': {'len':6,
                    'dof': ['X', 'Y', 'Z', 'x', 'y', 'z'],
                    '__gettr__': data.get_rigid_TxyzRxyz}
    }
    
    _SUP_HEADER_ROW = (['Time_stamp']+["RigidBody"] * len(data.rigid_bodies) * _params[save_type]['len'] + ["Marker"] * len(data.markers) * 3)
    _FPS_ROW = ["FPS", data.fps] + [0.0]*(len(_SUP_HEADER_ROW) - 2)
    _rb_col_names = [f"{rb}_{axis}" for rb in data.rigid_bodies for axis in _params[save_type]['dof']]
    _mk_col_names = [f"{mk}_{axis}" for mk in data.markers for axis in ['X', 'Y', 'Z']]
    _HEADER_ROW = ['Time']+_rb_col_names + _mk_col_names

    _dict_data_rigid = _params[save_type]['__gettr__']()
    _dict_data_marker = data.get_marker_Txyz()
    _dict_data_time = data.get_time()
    _dict_data_time = _dict_data_time.reshape((len(_dict_data_time), 1))

    # concatenate all the data into a single array for _dict_data_rigid
    _transformed_data_rigid = np.concatenate([add_rigidbody_offset(_dict_data_rigid[rb], X_offset, Y_offset, Z_offset) for rb in data.rigid_bodies], axis=1)
    _transformed_data_marker = np.concatenate([add_marker_offset(save_type,_dict_data_marker[mk], X_offset, Y_offset, Z_offset) for mk in data.markers], axis=1)
    _transformed_data = np.concatenate([_dict_data_time, _transformed_data_rigid, _transformed_data_marker], axis=1)

    # save as csv file with SUP_HEADER_ROW, FPS_ROW, HEADER_ROW, and _transformed_data
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(_SUP_HEADER_ROW)
        writer.writerow(_FPS_ROW)
        writer.writerow(_HEADER_ROW)
        writer.writerows(_transformed_data)
