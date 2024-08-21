import submodules.cleaned_file_parser as cfp
import submodules.data_filter as _df
import os
import numpy as np
from matplotlib import pyplot as plt
import submodules.robomath_addon as rma
import submodules.robomath as rm
from scipy.signal import savgol_filter #apply sg filter

START_FRAME = 0

def get_battery_stop_index(data, START_FRAME, thresehold):
    _times = data.get_time()
    battery = data.get_rigid_TxyzRxyz()['battery']
    W_T_bat = battery

    W_T_bat0 = battery[START_FRAME]
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


def calculate_new_marker_pos(W_V0: list[3], W_TxyzRxyz_B0: list[6], W_TxyzRxyz_Bt: list[6]) -> list[3]:
    '''
    Calculate the new marker position in the world frame at time t
    W_V0: Marker position wrt World at time 0
    W_TxyzRxyz_B0: Battery pose at time 0
    W_TxyzRxyz_Bt: Battery pose at time t
    ----------------
    Returns: Marker position wrt World at time t
    '''
    B0_V0 = rma.Vxyz_wrt_TxyzRxyz(W_V0, W_TxyzRxyz_B0) # Marker position wrt Battery at time 0
    B_V = B0_V0 # Marker position wrt Battery at time t is considered same as time 0 as the marker is rigidly attached to the battery
    W_T_B = rm.TxyzRxyz_2_Pose(W_TxyzRxyz_Bt)
    B_T_W = rm.invH(W_T_B)
    W_V = rma.Vxyz_wrt_Pose(B_V, B_T_W) # Marker position wrt World at time t (in world frame)
    return W_V



def calculate_new_mb_pos(W_TxyzRxyz_mb0: list[6], W_TxyzRxyz_B0: list[6], W_TxyzRxyz_Bt: list[6]) -> list[3]:
    '''
    Calculate the new marker position in the world frame at time t
    W_TxyzRxyz_I0: Rigid Body position wrt World at time 0
    W_TxyzRxyz_B0: Battery pose at time 0
    W_TxyzRxyz_Bt: Battery pose at time t
    ----------------
    Returns: Marker position wrt World at time t
    '''
    B0_T_mb0 = rma.BxyzRxyz_wrt_AxyzRxyz(W_TxyzRxyz_mb0, W_TxyzRxyz_B0) # Rigid Body position wrt Battery at time 0
    B_T_mb = B0_T_mb0 # Marker position wrt Battery at time t is considered same as time 0 as the marker is rigidly attached to the battery
    W_T_B = rm.TxyzRxyz_2_Pose(W_TxyzRxyz_Bt)
    W_T_mb = W_T_B*B_T_mb # Marker Body position wrt World at time t (in world frame)
    return W_T_mb


if __name__ == "__main__":
    base_dir = 'no-sync/datasets/turn_table_chisel/tilt_25/1.cleaned_data/training_traj/'
    save_dir = 'no-sync/datasets/turn_table_chisel/tilt_25/1.cleaned_data/training_traj/battery_motion_plots'
    files = os.listdir(base_dir)
    files  = [file for file in files if file.endswith('.csv')]

    for _index, file in enumerate(files):
        if _index == 1: break
        print(file)

        path = os.path.join(base_dir, file)
        data = cfp.DataParser.from_quat_file(file_path = path, target_fps= 120, filter=False, window_size=5, polyorder=3)
        stop_index = get_battery_stop_index(data, START_FRAME, thresehold=0.1)

        fig = plt.figure()
        fig.suptitle('Battery Motion')
        ax = fig.add_subplot(111)
        plot_battery_motion(ax, data, stop_index)
        plt.savefig(os.path.join(save_dir, file.split('.')[0] + '.png'))
        plt.clf()
