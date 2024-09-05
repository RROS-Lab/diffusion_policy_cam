import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import submodules.cleaned_file_parser as cfp
import submodules.ati_file_parser as afp
import submodules.data_filter as _df
import numpy as np
import csv
import re

def save_2_csv(state_data, force_data, file_path):
    # add first rows
    save_type = 'EULER'
    _params = {
        'QUAT': {'len':7,
                    'dof': ['X', 'Y', 'Z', 'w', 'x', 'y', 'z'] },
        'EULER': {'len':6,
                    'dof': ['X', 'Y', 'Z', 'x', 'y', 'z']},
        'FORCE': {'len':6,
                    'dof': ['FX', 'FY', 'FZ', 'Fx', 'Fy', 'Fz']}
    }
    
    _SUP_HEADER_ROW = (['Time_stamp']+["RigidBody"] * (len(state_data.rigid_bodies) + 1) * _params[save_type]['len']  +["RigidBody"]+ ["Marker"] * len(state_data.markers) * 3)
    _FPS_ROW = ["FPS", state_data.fps] + [0.0]*(len(_SUP_HEADER_ROW) - 2)
    
    _rb_col_names = []
    for rb in state_data.rigid_bodies:
        if rb == 'chisel':
            _rb_col_names.extend([f"{rb}_{axis}"  for axis in _params[save_type]['dof']] + [f"{rb}_{axis}"  for axis in _params['FORCE']['dof']])
            
        else :
            _rb_col_names.extend([f"{rb}_{axis}"  for axis in _params[save_type]['dof']])
    
    _mk_col_names = [f"{mk}_{axis}" for mk in state_data.markers for axis in ['X', 'Y', 'Z']]
    _HEADER_ROW = ['Time']+_rb_col_names + ['gripper_state'] + _mk_col_names

    _dict_data_rigid = state_data.get_rigid_TxyzRxyz()
    _on_off_data = state_data.get_rigid_state()['gripper']
    _array_force_data = force_data
    _dict_data_marker = state_data.get_marker_Txyz()
    _dict_data_time = state_data.get_time()
    lenght = len(_array_force_data)
    _dict_data_rigid['chisel'] = np.concatenate([_dict_data_rigid['chisel'][:lenght], _array_force_data], axis=1)
    
    

    _transformed_data_time = _dict_data_time.reshape((len(_dict_data_time), 1))[:lenght]
    _transformed_on_off_data = _on_off_data.reshape((len(_on_off_data), 1))[:lenght]
    
    # concatenate all the data into a single array for _dict_data_rigid    
    _transformed_data_rigid = np.concatenate([_dict_data_rigid[rb][:lenght] for rb in state_data.rigid_bodies], axis=1)
    
    _transformed_data_marker = np.concatenate([_dict_data_marker[mk][:lenght] for mk in state_data.markers], axis=1)
    
    _transformed_data = np.concatenate([_transformed_data_time, _transformed_data_rigid, _transformed_on_off_data, _transformed_data_marker], axis=1)

    # save as csv file with SUP_HEADER_ROW, FPS_ROW, HEADER_ROW, and _transformed_data
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(_SUP_HEADER_ROW)
        writer.writerow(_FPS_ROW)
        writer.writerow(_HEADER_ROW)
        writer.writerows(_transformed_data)
        
        
if __name__ == "__main__":


    state_dir = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/dataset_aug14/trimmed_traj/segmented_state_traj/csvs/'
    force_dir = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/dataset_aug14/ft_data_200/takes/'
    save_dir = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/dataset_aug14/trimmed_traj/segmented_without_gripperstate_traj/csvs/'
    
    for file in os.listdir(force_dir):
        if file.endswith(".csv"):
            pattern = re.compile(re.escape(file.split('.')[0]))
            state_files = os.listdir(state_dir)
            file_exists = [f for f in state_files if pattern.search(f)]
            print(file_exists)
            for state_file in file_exists:
                print(state_file)
                state_path = f'{state_dir}{state_file}'
                force_path = os.path.join(force_dir, file)
                save_path = f'{save_dir}{state_file.split(".")[0]}_force_state_cleaned.csv'
                
                state_data = cfp.DataParser.from_quat_file(file_path = state_path, target_fps= 120, filter=False, window_size=5, polyorder=3)
                print(len(state_data.markers))

                state_times = state_data.get_time()
                
                print('Lenght of State Data -',len(state_times))

                force_data = afp.ForceParser.from_euler_file(file_path = force_path)

                force_times = force_data.get_time()

                indices = set()
                for time in state_times:
                    # print("Time -",time)
                    closest = _df.find_closest_number(time, force_times)
                    # print("Closest -",closest)
                    index = _df.find_index(closest, force_times)
                    indices.add(index)
                    # print("Index -",index)
                # print(len(indices))

                force_data = force_data.get_force_data(indices=list(indices))
                print('Lenght of Force Data -',len(force_data))
                
                save_2_csv(state_data, force_data, save_path)





















