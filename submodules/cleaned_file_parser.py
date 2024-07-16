import pandas as pd
import submodules.robomath_addon as rma
# import robomath_addon as rma
import numpy as np
import submodules.data_filter as _df
# import data_filter as _df
from typing import Union


def process_data(data :pd.DataFrame,
                 fps: float = 30.0, filter: bool = False,
                 window_size: int = 15, polyorder: int = 3) -> pd.DataFrame:
    input_fps = float(data.iloc[0, 1])
    # print(input_fps)
    _new_data = _df.fps_sampler(data[1:], target_fps = fps, input_fps=input_fps)
    if filter:
        _new_data = _df.apply_savgol_filter(_new_data, window_size, polyorder)
    return _new_data

class DataParser:
    """
        Cleaned DATA parser.
    """
    def __extract_column_info__(self):
        # Extract unique marker, rigid body, and tool names
        marker_columns = [col for col in self.data.columns if 'Marker' in col]
        self.markers = {value.split('_')[0] for value in self.data[marker_columns].iloc[0]}
        # self.markers = {val.split('_')[0] for column in self.data.columns if column.startswith('Marker') for val in self.data[column].iloc[0]}
        rigid_body_columns = [col for col in self.data.columns if 'RigidBody' in col]
        self.rigid_bodies = {value.split('_')[0] for value in self.data[rigid_body_columns].iloc[0]}
        # self.rigid_bodies = {val.split('_')[0] for column in self.data.columns if column.startswith('RigidBody') for val in self.data[column].iloc[0]}
        tool_columns = [col for col in self.data.columns if 'Tool' in col]
        self.tools = {value.split('_')[0] for value in self.data[tool_columns].iloc[0]}
        # self.tools = {val.split('_')[0] for column in self.data.columns if column.startswith('Tool') for val in self.data[column].iloc[0]}
        self.data.columns = self.data.iloc[0]
        self.data = self.data.drop(index =0)

    # @classmethod
    def get_rigid_TxyzQwxyz(self, **kwargs) -> dict:
        """
        Process rigid body data from a DataFrame based on the specified type (QUAT or EULER).
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - rigid_bodies (list): List of rigid body names to process.
        - data_type (Union['QUAT', 'EULER']): Type of data to process ('QUAT' for quaternions, 'EULER' for Euler angles).
        
        Returns:
        - dict: Dictionary containing processed rigid body data for each rigid body.
        """
        rb_TxyzQwxyz = {}  # Dictionary to store processed data
        
        for rb in self.rigid_bodies:
            rb_columns = [col for col in self.data.columns if col.startswith(rb)]
            sorted_columns = sorted(rb_columns, key=lambda x: x.split('_')[1])
            
            # Select processing method based on data type and column length
            if self.file_type == 'QUAT':
                rb_TxyzQwxyz[rb] = self.data[sorted_columns].values.astype(float)

            elif self.file_type == 'EULER':
                rb_TxyzQwxyz[rb] = np.apply_along_axis(rma.TxyzRxyz_2_TxyzQwxyz, 1, self.data[sorted_columns].values.astype(float))

        for key, value in kwargs.items():
            if key == 'object':
                return {key: rb_TxyzQwxyz[key] for key in value if key in rb_TxyzQwxyz}
        
        return rb_TxyzQwxyz
    
    def get_rigid_TxyzRxyz(self, **kwargs) -> dict:
        """
        Process rigid body data from a DataFrame based on the specified type (QUAT or EULER).
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - rigid_bodies (list): List of rigid body names to process.
        - data_type (Union['QUAT', 'EULER']): Type of data to process ('QUAT' for quaternions, 'EULER' for Euler angles).
        
        Returns:
        - dict: Dictionary containing processed rigid body data for each rigid body.
        """
        rb_TxyzRxyz = {}  # Dictionary to store processed data
        
        for rb in self.rigid_bodies:
            rb_columns = [col for col in self.data.columns if col.startswith(rb)]
            sorted_columns = sorted(rb_columns, key=lambda x: x.split('_')[1])
            
            # Select processing method based on data type and column length
            if self.file_type == 'QUAT':
                rb_TxyzRxyz[rb] = np.apply_along_axis(rma.TxyzQwxyz_2_TxyzRxyz, 1, self.data[sorted_columns].values.astype(float))
            elif self.file_type == 'EULER':
                rb_TxyzRxyz[rb] = self.data[sorted_columns].values.astype(float)

        for key, value in kwargs.items():
            if key == 'object':
                return {key: rb_TxyzRxyz[key] for key in value if key in rb_TxyzRxyz}
    
        return rb_TxyzRxyz


    # @classmethod
    def get_marker_Txyz(self, **kwargs) -> dict:
        """
        Process marker data from a DataFrame.
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - markers (list): List of marker names to process.
        
        Returns:
        - dict: Dictionary containing processed marker data for each marker.
        """
        mk_Txyz = {}

        # Extract marker data
        for mk in self.markers:
            mk_columns = [col for col in self.data.columns if col.startswith(mk)]
            sorted_columns = sorted(mk_columns, key=lambda x: x.split('_')[1])
            mk_Txyz[mk] = np.apply_along_axis(rma.motive_2_robodk_marker, 1, self.data[sorted_columns].values.astype(float))

        for key, value in kwargs.items():
            if key == 'Marker':
                return {key: mk_Txyz[key] for key in value if key in mk_Txyz}

        return mk_Txyz

    
    @classmethod
    def from_euler_file(self, file_path, target_fps: float, filter: bool = False, window_size: int = 15, polyorder: int = 3):

        return DataParser(file_path, 'EULER', target_fps, filter, window_size, polyorder)
    

    
    @classmethod
    def from_quat_file(self, file_path, target_fps: float, filter: bool = False, window_size: int = 15, polyorder: int = 3):

        return DataParser(file_path, 'QUAT', target_fps, filter, window_size, polyorder)
    

    def __init__(self, file_path, file_type: Union['QUAT', 'EULER'], target_fps: float = 30.0, filter: bool = False, window_size: int = 15, polyorder: int = 3):
        self.data = process_data(pd.read_csv(file_path), target_fps, filter, window_size, polyorder)
        self.markers = set()
        self.rigid_bodies = set()
        self.tools = set()
        self.file_type = file_type
        self.__extract_column_info__()


if __name__ == "__main__":

    path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cap_008_cleaned.csv'

    data = DataParser.from_quat_file(file_path = path, target_fps=30.0, filter=False, window_size=15, polyorder=3)

    print(data.rigid_bodies)

    # gets a list of object id if needed
    tools = data.get_rigid_TxyzQwxyz(object = ['chisel'])

    print(tools)