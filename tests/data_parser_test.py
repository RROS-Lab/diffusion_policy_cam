# import submodules.robomath_addon as rma
# import submodules.motive_file_cleaner as mfc
# import submodules.data_filter as _df
from ..submodules import cleaned_file_parser as cfp

path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cap_008_cleaned.csv'

data = cfp.DataParser.from_quat_file(file_path = path, filter=True, window_size=15, polyorder=3)

print(data.rigid_bodies)

tools = data.get_rigid_data(object = ['chisel'])

print(tools.keys())