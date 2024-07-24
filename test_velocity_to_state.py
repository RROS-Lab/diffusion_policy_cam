import submodules.cleaned_file_parser as cfp
import os
import pandas as pd
import numpy as np
import submodules.data_filter as _df
import csv
import re
import warnings
warnings.filterwarnings("ignore")

target_fps = 120.0

action_item = ['chisel', 'gripper']
obs_item = ['battery']
# add first rows
_params = {
    'QUAT': {'len':7,
                'dof': ['X', 'Y', 'Z', 'w', 'x', 'y', 'z']},
    'EULER': {'len':6,
                'dof': ['X', 'Y', 'Z', 'x', 'y', 'z']},
    'Vel': {'len':6,
                'dof': ['Xv', 'Yv', 'Zv', 'xv', 'yv', 'zv']}
}

save_path = '/home/cam/Documents/scratch/diffusion_policy_cam/diffusion_pipline/data_chisel_task/converted_state_offset_intepolated_traj_test/'

path_dir = '/home/cam/Documents/scratch/diffusion_policy_cam/diffusion_pipline/data_chisel_task/vel_offset_intepolated_traj_test/'

for file in os.listdir(path_dir):
    if file.endswith(".csv"):
        combined_list = []
        save_type='EULER'
        
        
        # get start postion and time per step from the orignal state file
        file_data_n = re.sub(r'\_velocity.csv', f'.csv', file)
        # give path to the cleaned test file that has XYZ as positions
        file_data_name = '/home/cam/Documents/scratch/diffusion_policy_cam/diffusion_pipline/data_chisel_task/offset_interpolated_traj_test/' + file_data_n
        data_info = cfp.DataParser.from_euler_file(file_path = file_data_name, target_fps=target_fps, filter=True, window_size=15, polyorder=3)
        time_per_step = 1/data_info.fps
        start_pos = {obj : data_info.get_rigid_TxyzRxyz(item = [obj])[obj][0] for obj in action_item}

        
        # Load velocity data
        file_name = re.sub(r'\.csv', f'_state.csv', file)
        file_path = os.path.join(save_path, file_name)
        path_name = path_dir + file
        dict_of_df_rigid = {}
        dict_of_df_marker = {}
        name = []


        data = cfp.DataParser.from_euler_file(file_path = path_name, target_fps=target_fps, filter=True, window_size=15, polyorder=3)
        dict_of_df_marker = data.get_marker_Txyz()
        data_time = data.get_time().astype(float)
        data_vel_dict = data.get_rigid_TxyzRxyz()
        marker_name = data.markers
        
        initial_vel = np.zeros(6)
        data_state_dict = {}
        for key in data_vel_dict.keys():
            if key != 'battery':
                data_state_dict[key] = np.zeros_like(data_vel_dict[key])
                for i in range(len(data_time)):
                    data_state_dict[key][i] = start_pos[key] + (initial_vel * time_per_step)
                    start_pos[key] = data_state_dict[key][i]
                    initial_vel = data_vel_dict[key][i]
            else:
                data_state_dict[key] = data_vel_dict[key]

        dict_of_df_rigid = data_state_dict

        
        _SUP_HEADER_ROW = (['Time_stamp']+["RigidBody"] * len(data.rigid_bodies) * _params[save_type]['len'] + ["Marker"] * len(data.markers) * 3)
        _FPS_ROW = ["FPS", data.fps] + [0.0]*(len(_SUP_HEADER_ROW) - 2)
        _rb_col_names = [f"{rb}_{axis}" for rb in action_item for axis in _params[save_type]['dof']]
        _rbb_col_names = [f"{rb}_{axis}" for rb in obs_item for axis in _params[save_type]['dof']]
        _mk_col_names = [f"{mk}_{axis}" for mk in data.markers for axis in ['X', 'Y', 'Z']]
        _HEADER_ROW = ['Time'] +_rb_col_names  + _rbb_col_names + _mk_col_names
        _dict_data_time = data_time.reshape((len(data_time), 1))

        # concatenate all the data into a single array for _dict_data_rigid
        _transformed_data_rigid = np.concatenate([dict_of_df_rigid[rb] for rb in action_item], axis=1)
        _transformed_data_battery = np.concatenate([dict_of_df_rigid[rb] for rb in obs_item], axis=1)
        _transformed_data_marker = np.concatenate([dict_of_df_marker[mk] for mk in data.markers], axis=1)
        _transformed_data = np.concatenate([_dict_data_time ,_transformed_data_rigid, _transformed_data_battery, _transformed_data_marker], axis=1)

        # save as csv file with SUP_HEADER_ROW, FPS_ROW, HEADER_ROW, and _transformed_data
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(_SUP_HEADER_ROW)
            writer.writerow(_FPS_ROW)
            writer.writerow(_HEADER_ROW)
            writer.writerows(_transformed_data)