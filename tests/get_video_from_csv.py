import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


from submodules.plot_traj_3d import PlotTraj3D
from submodules import robomath_addon as rma
import numpy as np
from matplotlib import pyplot as plt
# from submodules.trajectory_animator import TrajectoryAnimator

def get_video_from_csv(source_path: str,
                       target_path: str):
    TARGET_FPS = 30.0
    INPUT_FPS = 120.0

    CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ = mfp.extract_data(file_path=source_path, 
                                                                                                                            target_fps=TARGET_FPS,
                                                                                                                            input_fps=INPUT_FPS)

    # dir_path = './../datasets/Chisel_problem/120/cleaned_test_data/'
    # CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ = mfp.extract_data_chisel(data_path=dir_path)

    C_TxyzQwxyz = np.concatenate([np.array(CXYZ), np.array(Cwxyz)], axis=1)
    G_TxyzQwxyz = np.concatenate([np.array(GXYZ), np.array(Gwxyz)], axis=1)
    B_TxyzQwxyz = np.concatenate([np.array(BXYZ), np.array(Bwxyz)], axis=1)

    #change C_TxyzRxyz to apply rma.normalize_eulers to all rows of C_TxyzRxyz[:, 3:]

    A1_Txyz = np.array(A1XYZ).reshape(-1, 3)
    A2_Txyz = np.array(A2XYZ).reshape(-1, 3)
    A3_Txyz = np.array(A3XYZ).reshape(-1, 3)
    B1_Txyz = np.array(B1XYZ).reshape(-1, 3)
    B2_Txyz = np.array(B2XYZ).reshape(-1, 3)
    B3_Txyz = np.array(B3XYZ).reshape(-1, 3)
    C1_Txyz = np.array(C1XYZ).reshape(-1, 3)
    C2_Txyz = np.array(C2XYZ).reshape(-1, 3)
    C3_Txyz = np.array(C3XYZ).reshape(-1, 3)



    plot = PlotTraj3D(fig_size=(10,7))
    plot.fig.tight_layout(pad=3.0)
    
    rigid_bodies = [C_TxyzQwxyz, G_TxyzQwxyz, B_TxyzQwxyz]
    markers  = [A1_Txyz, A2_Txyz, A3_Txyz, B1_Txyz, B2_Txyz, B3_Txyz, C1_Txyz, C2_Txyz, C3_Txyz]
    intial_markers = [_mk[0] for _mk in markers]

    ax1 = plot.add_subplot(r=1,c=1,i=1, 
                           projection='3d',title='3D Spline Trajectory', lables=['X', 'Y', 'Z'])
    
    #add all position points from rigid bodies [0:3] and markers without hardcoding using variable names
    ALL_POINTS = np.concatenate([*[i[:, 0:3] for i in rigid_bodies], 
                                 *markers], axis=0)
    
    plot.set_3D_plot_axis_limits(ax1, ALL_POINTS)

    ax2 = plot.add_subplot(r=4,c=4,i=1)
    ax3 = plot.add_subplot(r=4,c=4,i=5)
    # ax4 = plot.add_subplot(r=4,c=4,i=9)
    ax4 = None

    #plot Chisel trajectory
    plot.plot_single_traj(None, ax2, ax3, ax4, C_TxyzQwxyz, density=100)
    # plot.plot_single_traj(ax1, ax2, ax3, ax4, C_TxyzQwxyz, density=100)
    ax2.set_title('Chisel Trajectory')

    ax5 = plot.add_subplot(r=4,c=4,i=9)
    ax6 = plot.add_subplot(r=4,c=4,i=13)
    # ax7 = plot.add_subplot(r=3,c=4,i=10)
    ax7 = None

    #plot Gripper trajectory
    plot.plot_single_traj(None, ax5, ax6, ax7, G_TxyzQwxyz, density=100)
    # plot.plot_single_traj(ax1, ax5, ax6, ax7, G_TxyzQwxyz, density=100)
    ax5.set_title('Gripper Trajectory')
    #plot intial battery markers
    # plot.plot_nodes(ax1, nodes=intial_markers, color='black', marker='o', size=10)

    # plt.show()


    ##############################
    ani = plot.animate_multiple_trajectories(ax=ax1,
                                             list_trajectories=[*rigid_bodies, *markers],
                                             interval=50,
                                             quiver_line_width=2,
                                             quiver_size=0.07,
                                             path_line_width=0.4)
    
    if target_path:
        ani.save(target_path, writer='ffmpeg', fps=TARGET_FPS)
    # plt.show()

    ##############################

    ## Set up the matplotlib figure
    # fig = plt.figure(figsize=(10, 8))
    # animator = TrajectoryAnimator(fig)

    # # Define your trajectories; just an example here with one trajectory for simplicity
    # trajectories = [C_TxyzQwxyz]  # Add more trajectories if needed

    # # Create the animation
    # animation = animator.animate_multiple_trajectories(trajectories, interval=50)

    # # If running in a script and not in a Jupyter notebook, use plt.show()
    # plt.show()