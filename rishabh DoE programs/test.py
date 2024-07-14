from submodules.plot_traj_3d import PlotTraj3D
from submodules import robomath_addon as rma
import submodules.motive_file_parser as mfp
import numpy as np
from matplotlib import pyplot as plt
from submodules.trajectory_animator import TrajectoryAnimator

file_path = './../datasets/Chisel_problem/120/cleaned_test_data/test_134_cleaned.csv'
CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ = mfp.extract_data(file_path=file_path)

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


if __name__ == '__main__':
    # plot = PlotTraj3D()
    # ax1 = plot.add_subplot(121, projection='3d',
    #                         title='3D Spline Trajectory', 
    #                         lables=['X', 'Y', 'Z'])
    
    # ALL_POINTS = np.concatenate([C_TxyzQwxyz[:, 0:3], G_TxyzQwxyz[:, 0:3], B_TxyzQwxyz[:, 0:3], A1_Txyz, A2_Txyz, A3_Txyz, B1_Txyz, B2_Txyz, B3_Txyz, C1_Txyz, C2_Txyz, C3_Txyz], axis=0) 
    
    # plot.set_3D_plot_axis_limits(ax1, ALL_POINTS)

    # ax2 = plot.add_subplot(333)
    # ax3 = plot.add_subplot(336)
    # ax4 = plot.add_subplot(339)
    # plot.plot_single_traj(ax1, ax2, ax3, ax4, C_TxyzQwxyz, density=100)
    # plot.plot_single_traj(ax1, ax2, ax3, ax4, G_TxyzQwxyz, density=100)

    # markers = np.array([A1_Txyz[0], A2_Txyz[0], A3_Txyz[0], B1_Txyz[0], B2_Txyz[0], B3_Txyz[0], C1_Txyz[0], C2_Txyz[0], C3_Txyz[0]])
    # plot.plot_nodes(ax1, nodes=markers)
    # plt.show()


    ##############################

    plotter = PlotTraj3D()

    # ani = plotter.animate_trajectory(traj)
    ani = plotter.animate_multiple_trajectories(
                                                list_trajectories=[C_TxyzQwxyz, G_TxyzQwxyz, A1_Txyz, A2_Txyz, A3_Txyz, B1_Txyz, B2_Txyz, B3_Txyz, C1_Txyz, C2_Txyz, C3_Txyz],
                                                interval=50
                                                )
    ani.save('trajectory_animation.mp4', writer='ffmpeg', fps=60)
    plt.show()

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