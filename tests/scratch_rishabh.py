import submodules.robomath_addon as rma
import submodules.motive_file_cleaner as mfc
import submodules.data_filter as _df
import submodules.cleaned_file_parser as cfp


path_cleaned = './../diffusion_pipline/data_chisel_task/cap_008_cleaned.csv'

data = cfp.DataParser.for_quat_file(file_path = path_cleaned, filter=True, window_size=15, polyorder=3)

print(data.rigid_bodies)

tools = data.get_tool_data(data_type='EULER')

print(tools.keys())
