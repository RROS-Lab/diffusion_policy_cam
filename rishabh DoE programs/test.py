from submodules.plot_traj_3d import PlotTraj3D
from submodules import robomath_addon as rma
import submodules.motive_file_parser as mfp
import numpy as np


file_path = './../datasets/Chisel_problem/30/cleaned_test_data/test_127_cleaned.csv'
CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ = mfp.extract_data_chisel(file_path=file_path)


C_TxyzQwxyz = np.concatenate([np.array(CXYZ).reshape(-1, 3), np.array(Cwxyz).reshape(-1, 4)], axis=1)
G_TxyzQwxyz = np.concatenate([np.array(GXYZ).reshape(-1, 3), np.array(Gwxyz).reshape(-1, 4)], axis=1)
B_TxyzQwxyz = np.concatenate([np.array(BXYZ).reshape(-1, 3), np.array(Bwxyz).reshape(-1, 4)], axis=1)

C_TxyzQwxyz = np.apply_along_axis(rma.TxyzQwxyz_2_TxyzRxyz, 1, C_TxyzQwxyz)

A1_Txyz = np.array(A1XYZ).reshape(-1, 3)
A2_Txyz = np.array(A2XYZ).reshape(-1, 3)
A3_Txyz = np.array(A3XYZ).reshape(-1, 3)
B1_Txyz = np.array(B1XYZ).reshape(-1, 3)
B2_Txyz = np.array(B2XYZ).reshape(-1, 3)
B3_Txyz = np.array(B3XYZ).reshape(-1, 3)
C1_Txyz = np.array(C1XYZ).reshape(-1, 3)
C2_Txyz = np.array(C2XYZ).reshape(-1, 3)
C3_Txyz = np.array(C3XYZ).reshape(-1, 3)

PlotTraj3D().main(traj=C_TxyzQwxyz, density=10)