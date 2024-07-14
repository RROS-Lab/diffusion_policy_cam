import pandas as pd
import regex as re


def fps_sampler(data: pd.DataFrame, target_fps:float, input_fps: float = 240.0):
    sample_size = int(input_fps / target_fps)
    _output_data = []
    #get every nth row of a dataframe
    _output_data = data.iloc[::sample_size]
    return _output_data

