import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import submodules.cleaned_file_parser as cfp
import submodules.data_filter as _df
import numpy as np
from matplotlib import pyplot as plt
import submodules.robomath_addon as rma


path = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/dataset_aug14/force_traj/csvs/ft_002_force_state_cleaned.csv'

data = cfp.DataParser.from_euler_file(file_path = path, target_fps= 120, filter=False, window_size=5, polyorder=3)

print(data.rigid_bodies)
print(data.markers)

# _times = data.get_time()

# # =====
# battery = data.get_rigid_TxyzRxyz()['battery']
# W_T_bat = battery

# # ====
# W_T_bat0 = battery[0]
# bat0_T_bat = np.apply_along_axis(rma.BxyzRxyz_wrt_AxyzRxyz, 1, W_T_bat, W_T_bat0)
# yaw = bat0_T_bat[:,5]
# # =====
# # yaw = np.linalg.norm(W_T_bat, axis=1)

# from scipy.signal import savgol_filter #apply sg filter
# yaw = savgol_filter(yaw, 100, 3)
# # ====
# # yaw = np.unwrap(yaw)
# yaw = np.rad2deg(yaw)
# yaw_dot = rma.first_derivative(yaw, _times)



# plt.plot(_times, yaw_dot)
# plt.xlabel('Time')
# plt.ylabel('Yaw Rate')
# # Compute the differences (angular velocity)
# # angular_velocity = np.diff(euler_angles, axis=0)
# # angular_velocity_magnitude = np.linalg.norm(angular_velocity, axis=1)


# # Find the end of rotation
# thresehold = 1
# final_yaw = yaw[-1]
# stop_index = np.argmax(np.abs(yaw_dot[::-1]) > thresehold)  # Find from the reverse
# stop_index = len(yaw_dot) - stop_index - 1  # Convert to forward index
# stop_time = _times[stop_index]


# print(stop_time)


# print()
# '''
rigid_state = data.get_rigid_TxyzRxyz()
markers = data.get_marker_Txyz()
# time = data.get_time()


print(rigid_state['gripper'][0])
print(rigid_state['chisel'][0])
# for i in range(5):
    # print(markers['A1'][-i][0])
    # print(markers['B1'][-i][0])
    # print(markers['C1'][-i][0])

print(len(markers))
# print(rigid_state['chisel'][0])
# print(time[0])


#############################################################################
#############################################################################

# base_path = "/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cleaned_traj/"

# # Load data
# dict_of_df_rigid = {}
# dict_of_df_marker = {}


# for file in os.listdir(base_path):
#     if file.endswith(".csv") and file.startswith("cap"):
#         path_name = base_path + file
#         data = cfp.DataParser.from_quat_file(file_path = path_name, target_fps=120.0, filter=False, window_size=15, polyorder=3)
#         dict_of_df_rigid[file] = data.get_rigid_TxyzRxyz()
#         dict_of_df_marker[file] = data.get_marker_Txyz()
        
# if len(dict_of_df_rigid) == len(dict_of_df_marker):
#     rigiddataset, index = _df.episode_combiner(dict_of_df_rigid)

