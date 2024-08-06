import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from submodules.plot_traj_3d import PlotTraj3D
import numpy as np
from matplotlib import pyplot as plt
import concurrent.futures
import submodules.cleaned_file_parser as cfp
import warnings

warnings.filterwarnings("ignore")
# from submodules.trajectory_animator import TrajectoryAnimator

def get_visualization(data, save_path=None, video=False):
    #change C_TxyzRxyz to apply rma.normalize_eulers to all rows of C_TxyzRxyz[:, 3:]

    rigid_bodies_dict = data.get_rigid_TxyzQwxyz()
    markers_dict = data.get_marker_Txyz()



    plot = PlotTraj3D(fig_size=(10,7))
    plot.fig.tight_layout(pad=3.0)
    
    # rigid_bodies = [rigid_bodies_dict[rb] for rb in data.rigid_bodies if rb != 'battery']
    rigid_bodies = [rigid_bodies_dict[rb] for rb in data.rigid_bodies]
    markers  = [markers_dict[mk] for mk in data.markers]


    intial_markers = [_mk[0] for _mk in markers]

    ax1 = plot.add_subplot(r=1,c=1,i=1, 
                           projection='3d',title='3D Spline Trajectory', lables=['X', 'Y', 'Z'])
    
    #add all position points from rigid bodies [0:3] and markers without hardcoding using variable names
    ALL_POINTS = np.concatenate([*[i[:, 0:3] for i in rigid_bodies], 
                                 *markers], axis=0)
    
    plot.set_3D_plot_axis_limits(ax1, ALL_POINTS)

    ax2 = plot.add_subplot(r=4,c=4,i=1)
    ax2.set_title('Chisel Trajectory')
    ax3 = plot.add_subplot(r=4,c=4,i=5)
    # ax4 = plot.add_subplot(r=4,c=4,i=9)
    ax4 = None

    #side plot Chisel trajectory
    # plot.plot_single_traj(None, ax2, ax3, ax4, rigid_bodies_dict['chisel'], density=100)
    
    

    ax5 = plot.add_subplot(r=4,c=4,i=9)
    ax5.set_title('Gripper Trajectory')
    ax6 = plot.add_subplot(r=4,c=4,i=13)
    # ax7 = plot.add_subplot(r=3,c=4,i=10)
    ax7 = None

    #side plot Gripper trajectory
    # plot.plot_single_traj(None, ax5, ax6, ax7, rigid_bodies_dict['gripper'], density=100)

    
    
    #plot intial battery markers
    if not video:
        plot.plot_single_traj(ax1, ax2, ax3, ax4, rigid_bodies_dict['chisel'], density=100)
        plot.plot_single_traj(ax1, ax5, ax6, ax7, rigid_bodies_dict['gripper'], density=100)
        plot.plot_single_traj(ax1, ax5, ax6, ax7, rigid_bodies_dict['battery'], density=100)
        plot.plot_nodes(ax1, nodes=np.array(intial_markers))
        
    
    if video:
        ani = plot.animate_multiple_trajectories(ax=ax1,
                                                 list_trajectories=[*rigid_bodies, *markers],
                                                 interval=50,
                                                 quiver_line_width=2,
                                                 quiver_size=0.07,
                                                 path_line_width=0.4)
    
    
    if save_path:
        if video:
            ani.save(save_path, writer='ffmpeg', fps=data.fps)
        if not video:
            plt.savefig(save_path)
        plt.close()
    
    if not save_path:
        plt.show()
        # plt.show()

def process_and_visualize(file_name, base_dir, save_dir):
    if file_name.split('.')[-1] != 'csv':
        return
    
    read_path = os.path.join(base_dir, file_name)
    try:
        data = cfp.DataParser.from_euler_file(file_path=read_path, target_fps=120.0, filter=True, window_size=15, polyorder=3)
        file_name = read_path.split('/')[-1].split('.')[0]

        get_visualization(data=data,
                          save_path=os.path.join(save_dir, file_name + '.mp4'),
                          video=True)
    except Exception as e:
        print(f'file: {file_name} failed with error: \n\n{e}')

def main():
    base_dir = 'no-sync/turn_table_chisel/tilt_25/5_markers_&_9_markers/csvs Aug 6/test_5_noNAN'
    save_dir = 'no-sync/turn_table_chisel/tilt_25/5_markers_&_9_markers/csvs Aug 6/test_5_noNAN/videos'
    
    cleaned_file_names = sorted([file for file in os.listdir(base_dir) if file.endswith('.csv')])

    # files_not_2_plot = 

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for file_name in cleaned_file_names:
            print(f"Submitting {file_name} for processing")
            futures.append(executor.submit(process_and_visualize, file_name, base_dir, save_dir))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f'Error in future: {e}')

if __name__ == "__main__":
    main()

