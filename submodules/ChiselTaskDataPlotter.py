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

class ChiselTaskDataPlotter:
    def __init__(self, base_dir, save_dir, interpolate=False, read_file_fps=30.0, video_write_fps=30.0):
        self.base_dir = base_dir
        self.save_dir = save_dir
        self.interpolate = interpolate
        self.read_file_fps = read_file_fps
        self.video_write_fps = video_write_fps
        self.axes = []
        self.plot_type = None # [static, animated]

    @staticmethod
    def initialize_plot():
        plot = PlotTraj3D(fig_size=(10, 7))
        plot.fig.tight_layout(pad=3.0)
        return plot
    
    @staticmethod
    def setup_axes(plot):
        ax1 = plot.add_subplot(r=1, c=1, i=1, projection='3d', title='3D Spline Trajectory', labels=['X', 'Y', 'Z'])
        
        chisel_axes = ()
        gripper_axes = ()
        battery_axes = ()

        ax2 = plot.add_subplot(r=4, c=4, i=1); ax2.set_title('Chisel Trajectory')
        ax3 = plot.add_subplot(r=4, c=4, i=5)
        ax4 = None
        chisel_axes = (ax2, ax3, ax4)

        ax5 = plot.add_subplot(r=4, c=4, i=9); ax5.set_title('Gripper Trajectory')
        ax6 = plot.add_subplot(r=4, c=4, i=13)
        ax7 = None
        gripper_axes = (ax5, ax6, ax7)

        ax8 = plot.add_subplot(r=4, c=4, i=4); ax8.set_title('Battery Trajectory')
        ax9 = plot.add_subplot(r=4, c=4, i=8)
        ax10 = None
        battery_axes = (ax8, ax9, ax10)

        return ax1, chisel_axes, gripper_axes, battery_axes
    

    @staticmethod
    def concatenate_all_points(rigid_bodies, markers):
        return np.concatenate([*[i[:, 0:3] for i in rigid_bodies], *markers], axis=0)


    def plot_trajectories(self, plot, 
                          ax1, chisel_axes, gripper_axes, battery_axes,
                          rigid_bodies_dict, initial_markers):
        self.plot_type = 'static'
        plot.plot_single_traj(ax1, *chisel_axes, rigid_bodies_dict['chisel'], density=100)
        plot.plot_single_traj(ax1, *gripper_axes, rigid_bodies_dict['gripper'], density=100)
        plot.plot_single_traj(ax1, *battery_axes, rigid_bodies_dict['battery'], density=100)
        plot.plot_nodes(ax1, nodes=np.array(initial_markers))

    
    def animate_trajectories(self, plot, ax1, rigid_bodies, markers, time_stamps):
        self.plot_type = 'animated'
        return plot.animate_multiple_trajectories(
            ax=ax1,
            list_trajectories=[*rigid_bodies, *markers],
            time_data=time_stamps,
            interval=50,
            quiver_line_width=2,
            quiver_size=0.07,
            path_line_width=0.4
        )
    

    def save_plot(self, plot, ani, save_path, video):
        if save_path:
            if video:
                ani.save(save_path, writer='ffmpeg', fps=self.video_write_fps)
            else:
                plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_visualization(self, data, save_path=None, video=False):
        # Change C_TxyzRxyz to apply rma.normalize_eulers to all rows of C_TxyzRxyz[:, 3:]
        rigid_bodies_dict = data.get_rigid_TxyzQwxyz()
        markers_dict = data.get_marker_Txyz(interpolate=self.interpolate)
        
        rigid_bodies = [rigid_bodies_dict[rb] for rb in data.rigid_bodies]
        markers  = [markers_dict[mk] for mk in data.markers]
        time_stamps = data.get_time()
        initial_markers = [_mk[0] for _mk in markers]

        plot = PlotTraj3D(fig_size=(10, 7))
        plot.fig.tight_layout(pad=3.0)

        ax1 = plot.add_subplot(r=1, c=1, i=1, projection='3d', title='3D Spline Trajectory', labels=['X', 'Y', 'Z'])
        
        # Add all position points from rigid bodies [0:3] and markers without hardcoding using variable names
        ALL_POINTS = np.concatenate([*[i[:, 0:3] for i in rigid_bodies], *markers], axis=0)
        plot.set_3D_plot_axis_limits(ax1, ALL_POINTS)

        ax2 = plot.add_subplot(r=4, c=4, i=1)
        ax2.set_title('Chisel Trajectory')
        ax3 = plot.add_subplot(r=4, c=4, i=5)
        ax4 = None

        ax5 = plot.add_subplot(r=4, c=4, i=9)
        ax5.set_title('Gripper Trajectory')
        ax6 = plot.add_subplot(r=4, c=4, i=13)
        ax7 = None

        if not video:
            plot.plot_single_traj(ax1, ax2, ax3, ax4, rigid_bodies_dict['chisel'], density=100)
            plot.plot_single_traj(ax1, ax5, ax6, ax7, rigid_bodies_dict['gripper'], density=100)
            plot.plot_single_traj(ax1, ax5, ax6, ax7, rigid_bodies_dict['battery'], density=100)
            plot.plot_nodes(ax1, nodes=np.array(initial_markers))

        if video:
            ani = plot.animate_multiple_trajectories(
                ax=ax1,
                list_trajectories=[*rigid_bodies, *markers],
                time_data=time_stamps,  # Add this line if timer needed or else comment it out
                interval=50,
                quiver_line_width=2,
                quiver_size=0.07,
                path_line_width=0.4
            )

        if save_path:
            if video:
                ani.save(save_path, writer='ffmpeg', fps=self.video_write_fps)
            else:
                plt.savefig(save_path)
            plt.close()

        if not save_path:
            plt.show()



    def read_file_and_visualize(self, file_path, save_path):
        data = cfp.DataParser.from_quat_file(
            file_path=file_path, target_fps=self.read_file_fps, filter=True, window_size=15, polyorder=3
        )
        self.get_visualization(data=data, save_path=save_path, video=True)

    def process_files(self, max_workers=10, max_files_stop_flag=None, suffix='', parallelize=True):
        cleaned_file_names = sorted([file for file in os.listdir(self.base_dir) if file.endswith('.csv')])
        cleaned_file_names = cleaned_file_names[:max_files_stop_flag] if max_files_stop_flag else cleaned_file_names

        if suffix:
            suffix = f"_{suffix}"

        if parallelize:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for file_name in cleaned_file_names:
                    _file_path = os.path.join(self.base_dir, file_name)
                    _new_file_name = f'{os.path.splitext(file_name)[0]}{suffix}.mp4'
                    _save_path = os.path.join(self.save_dir, _new_file_name)
                    futures.append(executor.submit(self.read_file_and_visualize, _file_path, _save_path))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f'Error in future: {e}')
                        raise e
        else:
            for file_name in cleaned_file_names:
                _file_path = os.path.join(self.base_dir, file_name)
                _new_file_name = f'{os.path.splitext(file_name)[0]}{suffix}.mp4'
                _save_path = os.path.join(self.save_dir, _new_file_name)
                self.read_file_and_visualize(_file_path, _save_path)

    @staticmethod
    def main(base_dir, save_dir, max_workers=10, max_files_stop_flag=None, suffix='', parallelize=True, interpolate=False):
        plotter = ChiselTaskDataPlotter(base_dir, save_dir, interpolate)
        plotter.process_files(max_workers=max_workers, max_files_stop_flag=max_files_stop_flag, suffix=suffix, parallelize=parallelize)

if __name__ == "__main__":
    base_dir = 'no-sync/turn_table_chisel/tilt_25/overall data/overall_data_csvs'
    save_dir = 'no-sync/turn_table_chisel/tilt_25/overall data/interpolated_videos'
    INTERPOLATE = True
    suffix = "interpolate" if INTERPOLATE else ""
    ChiselTaskDataPlotter.main(base_dir, save_dir, max_workers=1, max_files_stop_flag=None, suffix=suffix, parallelize=False, interpolate=INTERPOLATE)
