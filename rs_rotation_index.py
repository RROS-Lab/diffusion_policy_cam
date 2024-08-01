import submodules.cleaned_file_parser as cfp
import submodules.data_filter as _df
import os
import numpy as np
from matplotlib import pyplot as plt
import submodules.robomath_addon as rma

from scipy.signal import savgol_filter #apply sg filter

def get_battery_stop_index(data, thresehold = 1):
    _times = data.get_time()
    battery = data.get_rigid_TxyzRxyz()['battery']
    W_T_bat = battery
    
    W_T_bat0 = battery[0]
    bat0_T_bat = np.apply_along_axis(rma.BxyzRxyz_wrt_AxyzRxyz, 1, W_T_bat, W_T_bat0)
    yaw = bat0_T_bat[:,5]
    yaw = savgol_filter(yaw, 100, 3)
    yaw = np.rad2deg(yaw)
    yaw_dot = rma.first_derivative(yaw, _times)
    
    final_yaw = yaw[-1]
    stop_index = np.argmax(np.abs(yaw_dot[::-1]) > thresehold)  # Find from the reverse
    stop_index = len(yaw_dot) - stop_index - 1  # Convert to forward index
    return stop_index

def plot_battery_motion(ax, data, stop_index):
    times = data.get_time()
    battery_data = data.get_rigid_TxyzRxyz()['battery']
    ax.plot(times, battery_data)
    ax.legend(['X', 'Y', 'Z', 'R', 'P', 'Y'])
    #add vertical line at stop index
    ax.axvline(x=times[stop_index], color='r', linestyle='--')
    

    

if __name__ == "__main__":
    base_dir = 'no-sync/datasets/turn_table_chisel/tilt_25/1.cleaned_data/training_traj/'
    save_dir = 'no-sync/datasets/turn_table_chisel/tilt_25/1.cleaned_data/training_traj/battery_motion_plots'
    files = os.listdir(base_dir)
    files  = [file for file in files if file.endswith('.csv')]
    
    
    
    for file in files:
        print(file)
        
        path = os.path.join(base_dir, file)
        data = cfp.DataParser.from_quat_file(file_path = path, target_fps= 120, filter=False, window_size=5, polyorder=3)
        stop_index = get_battery_stop_index(data)
        
        fig = plt.figure()
        fig.suptitle('Battery Motion')
        ax = fig.add_subplot(111)
        plot_battery_motion(ax, data, stop_index)
        plt.savefig(os.path.join(save_dir, file.split('.')[0] + '.png'))
        plt.clf()
        
