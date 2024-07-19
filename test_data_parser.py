# import submodules.robomath_addon as rma
# import submodules.motive_file_cleaner as mfc
# import submodules.data_filter as _df
import submodules.cleaned_file_parser as cfp
import submodules.data_filter as _df
import os

path = '/home/cam/Downloads/test/test_test_127_cleaned.csv'

data = cfp.DataParser.from_euler_file(file_path = path, target_fps= 120, filter=False, window_size=5, polyorder=3)

print(data.rigid_bodies)

rigid = data.get_rigid_TxyzQwxyz(item = ['gripper'])
# rigid_state = data.get_rigid_state(item = ['chisel'])
# markers = data.get_marker_Txyz(marker = ['A1'])
# time = data.get_time()


print(rigid['gripper'])
# print(markers['A1'][0])
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