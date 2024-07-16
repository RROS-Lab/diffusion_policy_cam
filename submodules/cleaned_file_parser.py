import pandas as pd
import submodules.robomath_addon as rma
import numpy as np
import submodules.data_filter as _df
from typing import Union


def process_data(data :pd.DataFrame,
                    fps: float = 30.0, filter: bool = False, 
                    window_size: int = 15, polyorder: int = 3) -> pd.DataFrame:
    

    _new_data = _df.fps_sampler(data, target_fps = fps)
    if filter:
        _new_data = _df.apply_savgol_filter(_new_data, window_size, polyorder)
    return _new_data

class DataParser:
    """
        Cleaned DATA parser.
    """
    def __init__(self, file_path, fps: float = 30.0, filter: bool = False, window_size: int = 15, polyorder: int = 3):
        self.data = process_data(pd.read_csv(file_path), fps, filter, window_size, polyorder)
        self.markers = set()
        self.rigid_bodies = set()
        self.tools = set()
        self.__extract_column_info__()

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
    def get_rigidbody_data(self, data_type: Union['QUAT', 'EULER']) -> dict:
        """
        Process rigid body data from a DataFrame based on the specified type (QUAT or EULER).
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - rigid_bodies (list): List of rigid body names to process.
        - data_type (Union['QUAT', 'EULER']): Type of data to process ('QUAT' for quaternions, 'EULER' for Euler angles).
        
        Returns:
        - dict: Dictionary containing processed rigid body data for each rigid body.
        """
        rb_TxyzQR = {}  # Dictionary to store processed data
        
        for rb in self.rigid_bodies:
            rb_columns = [col for col in self.data.columns if col.startswith(rb)]
            sorted_columns = sorted(rb_columns, key=lambda x: x.split('_')[1])
            
            # Select processing method based on data type and column length
            if len(self.data[sorted_columns].iloc[0]) == 7:
                if data_type == 'QUAT':
                    print('QUAT')
                    rb_TxyzQR[rb] = self.data[sorted_columns].values.astype(float)
                    print(rb_TxyzQR[rb][0])

                elif data_type == 'EULER':
                    print('EULER')
                    # # single = self.data[sorted_columns].values.astype(float)[0].astype(float)
                    # print(single)
                    # valaa = rma.TxyzQwxyz_2_TxyzRxyz(single)
                    rb_TxyzQR[rb] = np.apply_along_axis(rma.TxyzQwxyz_2_TxyzRxyz, 1, self.data[sorted_columns].values.astype(float))
                    # print(valaa)
                    # print(rb_TxyzQR[rb][0])

            else:
                if data_type == 'QUAT':
                    rb_TxyzQR[rb] = np.apply_along_axis(rma.TxyzRxyz_2_TxyzQwxyz, 1, self.data[sorted_columns].values.astype(float))
                elif data_type == 'EULER':
                    rb_TxyzQR[rb] = self.data[sorted_columns].values.astype(float)
        
        return rb_TxyzQR
    
    # @classmethod
    def get_marker_data(self) -> dict:
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

        return mk_Txyz
    
    # @classmethod
    def get_tool_data(self, data_type: Union['QUAT', 'EULER']) -> dict:
        """
        Process tool data from a DataFrame based on the specified type (QUAT or EULER).
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - tools (list): List of tool names to process.
        - data_type (Union['QUAT', 'EULER']): Type of data to process ('QUAT' for quaternions, 'EULER' for Euler angles).
        
        Returns:
        - dict: Dictionary containing processed tool data for each tool.
        """
        tcp_TxyzQR = {} # Dictionary to store processed data

        for tcp in self.tools:
            tcp_columns = [col for col in self.data.columns if col.startswith(tcp)]
            sorted_columns = sorted(tcp_columns, key=lambda x: x.split('_')[1])
            
            # Select processing method based on data type and column length
            if len(self.data[sorted_columns].iloc[0]) == 7:
                if data_type == 'QUAT':
                    tcp_TxyzQR[tcp] = self.data[sorted_columns].values.astype(float)
                elif data_type == 'EULER':
                    tcp_TxyzQR[tcp] = np.apply_along_axis(rma.TxyzQwxyz_2_TxyzRxyz, 1, self.data[sorted_columns].values.astype(float))
            else:
                if data_type == 'QUAT':
                    tcp_TxyzQR[tcp] = np.apply_along_axis(rma.TxyzRxyz_2_TxyzQwxyz, 1, self.data[sorted_columns].values.astype(float))
                elif data_type == 'EULER':
                    tcp_TxyzQR[tcp] = self.data[sorted_columns].values.astype(float)

        return tcp_TxyzQR
    