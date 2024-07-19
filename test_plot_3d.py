# import submodules.robomath_addon as rma
# import submodules.motive_file_cleaner as mfc
# import submodules.data_filter as _df
from submodules.plot_traj_3d import PlotTraj3D
import numpy as np
from matplotlib import pyplot as plt

# from submodules.trajectory_animator import TrajectoryAnimator

def get_visualization(data, save_path=None, video=False):
    

    #change C_TxyzRxyz to apply rma.normalize_eulers to all rows of C_TxyzRxyz[:, 3:]

    rigid_bodies_dict = data.get_rigid_TxyzQwxyz()
    markers_dict = data.get_marker_Txyz()



    plot = PlotTraj3D(fig_size=(10,7))
    plot.fig.tight_layout(pad=3.0)
    
    rigid_bodies = [rigid_bodies_dict[rb] for rb in data.rigid_bodies if rb != 'battery']
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


if __name__ == "__main__":
    import submodules.cleaned_file_parser as cfp
    import submodules.plot_traj_3d as pt3d
    import os

    #write
    
    
    
    # read_path = write_path # test read
    

    # read_path = './no-sync/outputs/test_128_raw_cleaned.csv' #std. read ##TODO
    # data = cfp.DataParser.from_quat_file(file_path = read_path, target_fps= 120.0, filter=True, window_size=15, polyorder=3)
    base_dir = './diffusion_pipline/try_pred'
    save_dir = './diffusion_pipline/test_save'


    cleaned_file_names = os.listdir(base_dir)

    for file_name in cleaned_file_names:
        if file_name == '.DS_Store':
            continue
        
        print(file_name)
        read_path = os.path.join(base_dir, file_name)
        data = cfp.DataParser.from_euler_file(file_path = read_path, target_fps= 120.0, filter=True, window_size=15, polyorder=3)
        file_name = read_path.split('/')[-1].split('.')[0]

        # data.save_2_csv(file_path=write_path, save_type='EULER')
        try:
            get_visualization(data=data,
                            save_path=os.path.join(save_dir, file_name + '.png'),
                            video=False)
        except Exception as e:
            print(f'file: {file_name} failed with error: \n\n{e}')
            # continue
            # raise e

    