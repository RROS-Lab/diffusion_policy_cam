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
import submodules.nan_interpolation as ni

warnings.filterwarnings("ignore")
# from submodules.trajectory_animator import TrajectoryAnimator
FILE_READ_FPS = 30.0
VIDEO_WRITE_FPS = 30.0

INTERPOLATE = False
MAKE_VIDEO = True

TIME_STAMPS = False

def get_visualization(data, save_path=None, video=False, **kwargs):
    #change C_TxyzRxyz to apply rma.normalize_eulers to all rows of C_TxyzRxyz[:, 3:]
    COMMENTS = kwargs.get('comments', "")

    rigid_bodies_dict = data.get_rigid_TxyzQwxyz()

    global INTERPOLATE
    markers_dict = data.get_marker_Txyz(interpolate=INTERPOLATE)
    
    # rigid_bodies = [rigid_bodies_dict[rb] for rb in data.rigid_bodies if rb != 'battery']
    rigid_bodies = [rigid_bodies_dict[rb] for rb in data.rigid_bodies]
    
    markers, marker_labels = zip(*[(markers_dict[mk], mk) for mk in data.markers])

    intial_markers = [_mk[0] for _mk in markers]


    plot = PlotTraj3D(fig_size=(10,7))
    plot.fig.tight_layout(pad=3.0)

    ax1 = plot.add_subplot(r=1,c=1,i=1, 
                           projection='3d',title='3D Spline Trajectory', labels=['X', 'Y', 'Z'])
    
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

    ax8 = plot.add_subplot(r=4,c=4,i=12)
    ax8.set_title('Helmet Trajectory')
    ax9 = plot.add_subplot(r=4,c=4,i=16)
    ax10 = None
    
    
    #plot intial battery markers
    if not video:
        plot.plot_single_traj(ax1, ax2, ax3, ax4, rigid_bodies_dict['chisel'], density=100, _qsize = 0.07)
        plot.plot_single_traj(ax1, ax5, ax6, ax7, rigid_bodies_dict['gripper'], density=100, _qsize = 0.07)
        # plot.plot_single_traj(ax1, ax5, ax6, ax7, rigid_bodies_dict['battery'], density=100, _qsize = 0.07)
        # plot.plot_single_traj(ax1, ax8, ax9, ax10, rigid_bodies_dict['helmet'], density=100, _qsize = 0.07)

        plot.plot_nodes(ax1, nodes=np.array(intial_markers), labels = marker_labels)
    
    if video:
        time_stamps = data.get_time() if TIME_STAMPS else np.array([])
        ani = plot.animate_trajectories_and_markers(ax=ax1,
                                                #  list_trajectories=[*rigid_bodies, *markers],
                                                 dict_trajectories=rigid_bodies_dict,
                                                 dict_markers= markers_dict,
                                                 time_data = time_stamps, # add this line if timer needed or else comment it out
                                                 interval=50,
                                                 quiver_line_width=2,
                                                 quiver_size=0.07,
                                                 path_line_width=0.7,
                                                 marker_line_width=0.3,
                                                 comments=COMMENTS)
    
    if save_path:
        if video:
            ani.save(save_path, writer='ffmpeg', fps=VIDEO_WRITE_FPS)
        if not video:
            plt.savefig(save_path)
        plt.close()
    
    if not save_path:
        plt.show()
        # plt.show()

def read_file_and_visualize(file_path, save_path, **kwargs):
    COMMENTS = kwargs.get('comments', "")

    _file_name = os.path.basename(file_path)
    data = cfp.DataParser.from_euler_file(file_path=file_path, target_fps=FILE_READ_FPS, filter=True, window_size=15, polyorder=3)
    get_visualization(data=data,
                      save_path=os.path.join(save_path), video=MAKE_VIDEO,
                      comments=COMMENTS
                      # save_path=None, video=True
                      )

# ------ Hard Coded for now ------
def main(max_workers=10, STOP_FLAG=None, **kwargs):  #THis is HARD CODED for now
    global INTERPOLATE
    # base_dir = 'no-sync/turn_table_chisel/tilt_25/1.cleaned_data/training_traj/csvs'; INTERPOLATE = False
    base_dir = 'no-sync/aug14/temp/csvs'
    save_dir = 'no-sync/aug14/temp/videos'
    
    cleaned_file_names = sorted([file for file in os.listdir(base_dir) if file.endswith('.csv')])
    cleaned_file_names = cleaned_file_names[:STOP_FLAG] if STOP_FLAG else cleaned_file_names
    
    # files_not_2_plot = 
    _SUFFIX = kwargs.get('suffix', '') # suffix to add to the file name w/o extension
    if _suffix: _SUFFIX = f"_{_suffix}"


    BOOL_PARALLELIZE = kwargs.get('parallelize', True)

    if BOOL_PARALLELIZE:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file_name in cleaned_file_names:
                _CMNTS = f"{file_name}"
                _file_path = os.path.join(base_dir, file_name)
                #### add SUFFIX to the file name before saving ####
                _new_file_name = f'{os.path.splitext(file_name)[0]}{_SUFFIX}.mp4' if MAKE_VIDEO else f'{os.path.splitext(file_name)[0]}{_SUFFIX}.png'

                _save_path = os.path.join(save_dir, _new_file_name)

                print(f"Submitting: {file_name} -> save as: {_new_file_name}")
                futures.append(executor.submit(read_file_and_visualize, _file_path, _save_path, comments = _CMNTS))

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Error in future: {e}')
                    raise e
    
    if not BOOL_PARALLELIZE:
        for file_name in cleaned_file_names:
            _CMNTS = f"{file_name}"
            _file_path = os.path.join(base_dir, file_name)
            #### add SUFFIX to the file name before saving ####
            _new_file_name = f'{os.path.splitext(file_name)[0]}{_SUFFIX}.mp4' if MAKE_VIDEO else f'{os.path.splitext(file_name)[0]}{_SUFFIX}.png'
            _save_path = os.path.join(save_dir, _new_file_name)

            print(f"Processing: {file_name} -> save as: {_new_file_name}")
            read_file_and_visualize(_file_path, _save_path, comments = _CMNTS)

if __name__ == "__main__":
    _suffix = "interpolate" if INTERPOLATE else ""
    main(max_workers = 10, STOP_FLAG=None, suffix=_suffix, parallelize = True)