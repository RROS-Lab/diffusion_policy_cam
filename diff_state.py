# import submodules.robomath_addon as rma
# import submodules.motive_file_cleaner as mfc
# import submodules.data_filter as _df
from submodules.plot_traj_3d import PlotTraj3D
import numpy as np
from matplotlib import pyplot as plt

# from submodules.trajectory_animator import TrajectoryAnimator




if __name__ == "__main__":
    import submodules.cleaned_file_parser as cfp
    import submodules.plot_traj_3d as pt3d
    import os
    import warnings
    warnings.filterwarnings("ignore")

    #write
    
    
    
    # read_path = write_path # test read
    

    # read_path = './no-sync/outputs/test_128_raw_cleaned.csv' #std. read ##TODO
    # data = cfp.DataParser.from_quat_file(file_path = read_path, target_fps= 120.0, filter=True, window_size=15, polyorder=3)
    base_dir = './diffusion_pipline/data_chisel_task/cleaned_traj'
    save_dir = './diffusion_pipline/data_chisel_task/cleaned_traj/velocities'
    
    cleaned_file_names = os.listdir(base_dir)

    for file_name in cleaned_file_names[2:3]:
        print(file_name)
        if file_name.split('.')[-1] != 'csv':
            continue
        
        print(file_name)
        read_path = os.path.join(base_dir, file_name)
        data = cfp.DataParser.from_quat_file(file_path = read_path, target_fps= 120.0, filter=True, window_size=15, polyorder=3)

        data_time = data.get_time()
        data_state_dict = data.get_rigid_TxyzRxyz()

        # use the time and state data to get the velocity data
        data_velocity_dict = {}
        for key in data_state_dict.keys():
            data_velocity_dict[key] = np.zeros_like(data_state_dict[key])
            for i in range(1, len(data_time)):
                data_velocity_dict[key][i] = (data_state_dict[key][i] - data_state_dict[key][i-1]) / (data_time[i] - data_time[i-1])


        #save the data_velocity_dict to csv
        file_name = read_path.split('/')[-1].split('.')[0]
        file_path_velocity = os.path.join(save_dir, 'velocity'+ file_name + '.csv')
        # convert the dictionary to a pandas dataframe
        # data_velocity = cfp.DataParser.from_dict(data_velocity_dict)
        # data_velocity.save_2_csv(file_path=file_path_velocity, save_type='EULER')

        # # plot the velocity data
        
        # file_name = read_path.split('/')[-1].split('.')[0]
        # file_path_state = os.path.join(save_dir, 'state'+ file_name + '.png')
        # file_path_velocity = os.path.join(save_dir, 'velocity'+ file_name + '.png')


        # data.save_2_csv(file_path=file_path_state, save_type='EULER')

