# import submodules.robomath_addon as rma
# import submodules.motive_file_cleaner as mfc
# import submodules.data_filter as _df
import submodules.cleaned_file_parser as cfp

path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cleaned_traj/test_128_cleaned.csv'

data = cfp.DataParser.from_quat_file(file_path = path, target_fps= 120, filter=False, window_size=5, polyorder=3)

print(data.rigid_bodies)

rigid = data.get_rigid_TxyzQwxyz(item = ['chisel'])
rigid_state = data.get_rigid_state(item = ['chisel'])
markers = data.get_marker_Txyz(marker = ['A1'])
time = data.get_time()

print(rigid['chisel'][0])
print(markers['A1'][0])
print(rigid_state['chisel'][0])
print(time[0])

