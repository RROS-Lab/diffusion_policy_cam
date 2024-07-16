import math
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation
from matplotlib.backends.backend_pdf import PdfPages
import submodules.robomath as rm
import submodules.robomath_addon as rma
import submodules.motive_file_cleaner as mfc

# def euler_change_rishabh(Gwxyz) : #receives a list of quaternions
#     eulers = rma.quaternions_2_eulers(Gwxyz) #reshape the quaternions to a 2D array
    
#     rolls = eulers[:,0]
#     pitches = eulers[:,1]
#     yaws = eulers[:,2]

#     smooth_rolls = rma.normalize_eulers_method2(rolls)
#     smooth_pitches = rma.normalize_eulers_method2(pitches)
#     smooth_yaws = rma.normalize_eulers_method2(yaws)

#     smooth_eulers = np.array([smooth_rolls, smooth_pitches, smooth_yaws]).T

#     return smooth_eulers


# def euler_change(Gwxyz) : #receives a list of quaternions
#     quaternions = np.array(Gwxyz) #reshape the quaternions to a 2D array
#     eulers = rma.quaternions_2_eulers(Gwxyz)
#     smooth_eulers = np.apply_along_axis(rma.normalize_eulers, axis=0, arr=eulers)    
#     return smooth_eulers



print("..........")
# path_name = "/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_handover_task/Supporting Data - Sheet1.csv"
# metadata = pd.read_csv(path_name)
# base_path = "/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_handover_task/traj/"
# _columns = metadata.columns
# exps_metadata = {}
# count = 0
# for i in range(len(metadata)):
#     if metadata[_columns[3]][i] == 'accept':
#         exps_metadata[count] = {
#             'Path': base_path + str(metadata[_columns[0]][i]) + '.csv',
#             'start_frame': int(metadata[_columns[1]][i]),
#             'end_frame': int(metadata[_columns[2]][i]),
#             'Note': metadata[_columns[4]][i]
#         }
#         count += 1



# for key in exps_metadata:
#     mfp.motive_handover_task_cleaner(
#                                     csv_path = exps_metadata[key]['Path'],
#                                     start_frame = exps_metadata[key]['start_frame'],
#                                     end_frame = exps_metadata[key]['end_frame']
#                                     )



    # euler_Gwxyz = euler_change_old(Gwxyz)  # Convert quaternions to Euler angles
    # euler_Gwxyz = Gwxyz
    # {'GR', 'GP', 'SR', 'SP', 'BxR', 'BxP', 'BuR', 'BuP'}



# with PdfPages('experiment_results.pdf') as pdf:
    # create_subplots(_data['GP'],
    #                 _data['GR'],
    #                 _data['SP'],
    #                 _data['SR'],
    #                 title=['Gripper Pos', 'Gripper Rot', 'Scooper Pos','Scooper Rot'],
    #                 layouts = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)],
    #                 sup_title = f"take_{key}")

    # print(f"creating subplots for {key}")  
    # pdf.savefig()
    # plt.close()

# for key in exps_metadata:
#     _data= extract_data(exps_metadata[key])
#     print(_data)
#     # create_xy_video(_data['GP'], _data['BxP'], _data['SP'], fps=30, video_filename=f'videos/output_{key}.avi', marker_list=['o', 'x', '*'])


dir_path = '/Users/rysavM1-Pro/Documents/GitHub/diffusion_policy_cam/no-sync/datasets/dataset july 10/test/test_128.csv'
save_path = 'test_cleaned_128.csv'

# for file in os.listdir(dir_path):
#     if file.endswith(".csv"):
#         print(file)


mfc.motive_chizel_task_cleaner(
    csv_path = dir_path, save_path=save_path
)