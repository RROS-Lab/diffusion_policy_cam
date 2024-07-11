import pandas as pd
import regex as re


def fps_sampler(data, target_fps:float, input_fps: float = 240.0):
    sample_size = int(input_fps / target_fps)
    _output_data = []
    for i in range(data.shape[0]):
        if i% sample_size == 0:
            _output_data.append(_row)
    return _output_data


def extract_motive_data(csv_path:str,
                 filter_keys: dict[str, str],
                 start_frame: int, end_frame: int, 
                 fps: float = 30.0):
    '''
    csv_path:str,
    filter_keys: dict[str, str] 
    start_frame: int, end_frame: int, 
    fps: float = 30.0
    ---

    filter_keys = {
    'GR': 'GRIPPER_2_Rotation', 
    'GP': 'GRIPPER_2_Position',
    'SR': 'diff_scooper_2_2_Rotation',
    'SP': 'diff_scooper_2_2_Position',
    'BxR': 'box3_Rotation',
    'BxP': 'box3_Position',
    'BuR': 'bucket_SC_Rotation',
    'BuP': 'bucket_SC_Position'}

    '''
    _data  = pd.read_csv(csv_path)
    _data = _data.reset_index(drop=False)
    _data = _data.drop(index =0)
    _data = _data.drop(index =2)
    _data = _data.reset_index(drop=True)

    row1 = _data.iloc[0]
    row2 = _data.iloc[1]
    row3 = _data.iloc[2]

    combined_values = []
    for a, b, c in zip(row1, row2, row3):
        combined_values.append(str(a) + '_' + str(b) + '_' + str(c))

    _data.columns = combined_values
    _data = _data.drop(index =0)
    _data = _data.drop(index =1)
    _data = _data.drop(index =2)
    _data = _data.drop(_data.columns[:2], axis=1)
    # print(result_dict)
    _data = _data.iloc[start_frame:end_frame]
    _data = _data.dropna()
    _data = _data.reset_index(drop=True)
    _data = fps_sampler(_data, target_fps=fps, input_fps=240.0)

    useful_data = {}
    for _key, _value in filter_keys.items():
        pattern = re.compile(_value)
        useful_data[_key] = _data.filter(regex=pattern).values.astype('float64').tolist()
        
    # Regular expression pattern to match columns start_frameing with 'gripper_1_Rotation'
    return useful_data


def motive_new_parser():
    pass