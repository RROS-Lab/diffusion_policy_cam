import pandas as pd
import submodules.robomath_addon as rma
import numpy as np
from data_filter import *


class DataProcessor:
    def __init__(self, file_path, fps: float = 30.0, filter: bool = False, window_size: int = 15, polyorder: int = 3):
        self.data = process_data(pd.read_csv(file_path), fps, filter, window_size, polyorder)
        self.markers = []
        self.rigid_bodies = []
        self.tools = []
        self._extract_column_info()

    @property
    def _extract_column_info(self):
        # Extract unique marker, rigid body, and tool names
        self.markers = {val.split('_')[0] for column in self.data.columns if column.startswith('Marker') for val in self.data[column].iloc[0]}
        self.rigid_bodies = {val.split('_')[0] for column in self.data.columns if column.startswith('RigidBody') for val in self.data[column].iloc[0]}
        self.tools = {val.split('_')[0] for column in self.data.columns if column.startswith('Tool') for val in self.data[column].iloc[0]}

    @classmethod
    def extract_cleaned_data(self) -> np.array:
        rb_TxyzQwxyz = {}
        tcp_TxyzQwxyz = {}
        mk_TxyzQwxyz = {}

        self.data.columns = self.data.iloc[0]
        self.data = self.data.drop(index =0)

        # Extract rigid body data
        for rb in self.rigid_bodies:
            rb_columns = [col for col in self.data.columns if col.startswith(rb)]
            sorted_columns = sorted(rb_columns, key=lambda x: x.split('_')[1])
            _rb_data = self.data[sorted_columns]
            rb_TxyzQwxyz[rb] = np.apply_along_axis(rma.motive_2_robodk_rigidbody, 1, _rb_data.values)

        # Extract marker data
        for mk in self.markers:
            mk_columns = [col for col in self.data.columns if col.startswith(mk)]
            sorted_columns = sorted(mk_columns, key=lambda x: x.split('_')[1])
            _mk_data = self.data[sorted_columns]
            mk_TxyzQwxyz[mk] = np.apply_along_axis(rma.motive_2_robodk_marker, 1, _mk_data.values)

        # Extract tool data
        for tcp in self.tools:
            tcp_columns = [col for col in self.data.columns if col.startswith(tcp)]
            sorted_columns = sorted(tcp_columns, key=lambda x: x.split('_')[1])
            _tcp_data = self.data[sorted_columns]
            tcp_TxyzQwxyz[tcp] = np.apply_along_axis(rma.motive_2_robodk_rigidbody, 1, _tcp_data.values)

        return rb_TxyzQwxyz, mk_TxyzQwxyz, tcp_TxyzQwxyz


    @staticmethod
    def process_data(data :pd.DataFrame,
                      fps: float = 30.0, filter: bool = False, 
                      window_size: int = 15, polyorder: int = 3) -> pd.DataFrame:

        _new_data = fps_sampler(data, target_fps = fps)
        if filter:
            _new_data = apply_savgol_filter(_new_data, window_size, polyorder)
        return _new_data