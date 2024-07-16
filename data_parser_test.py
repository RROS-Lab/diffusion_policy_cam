import submodules.robomath_addon as rma
import submodules.motive_file_cleaner as mfc
import submodules.data_filter as _df
import submodules.cleaned_file_parser as cfp


path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cleaned_traj/cap_008_cleaned.csv'


data = cfp.DataParser.for_eular_file(file_path = path, filter=True, window_size=15, polyorder=3)

print(data.rigid_bodies)

tools = data.get_rigidbody_data(data_type='EULER')

# print(tools)