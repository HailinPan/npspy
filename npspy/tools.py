import random
import numpy as np
import pandas as pd
import copy
from scipy import interpolate
from scipy.ndimage import median_filter
from typing import Dict, Optional, Tuple, Union, Literal, List

def extract_reads_as_an_obj(
    obj: dict,
    read_ids: list,
) -> dict:
    """extract reads in an obj as a new obj

    Args:
        obj (dict): obj
        read_ids (list): read ids for reads that would be selected

    Returns:
        dict: new obj
    """
    sub_obj = {}
    for read_id in read_ids:
        sub_obj[read_id] = copy.deepcopy(obj[read_id])
    return sub_obj

def delete_reads_in_an_obj(
    obj: dict,
    reads_need_to_remove: list,
) -> dict:
    
    delete_read_dict = {i:1 for i in reads_need_to_remove}

    sub_obj = {}
    for read_id, read_obj in obj.items():
        if read_id not in delete_read_dict:
            sub_obj[read_id] = copy.deepcopy(read_obj)

    return sub_obj

    
    

def extract_reads_with_window(
    obj: dict,
) -> dict:
    read_ids = []
    for read_id, read_obj in obj.items():
        if 'window' in read_obj and read_obj['window'] != None:
            read_ids.append(read_id)
    obj = extract_reads_as_an_obj(obj, read_ids=read_ids)
    return obj



def extract_reads_with_labels(
    obj: dict,
    labels: Union[list, str],
) -> dict:
    """extract reads with labels

    Args:
        obj (dict): obj
        labels (list): labels

    Returns:
        dict: obj
    """
    if isinstance(labels, str):
        labels = [labels]
    obj = {read_id:read_obj for read_id, read_obj in obj.items() if read_obj['label'] in labels}
    return obj

def extract_reads_by_stair(
    obj: dict,
    stair_nums: Union[int, list],
    read_num: int = -1,
    seed: int = 0,
) -> dict:
    read_ids = []
    if isinstance(stair_nums, int):
        stair_nums = [stair_nums]
    for read_id, read_obj in obj.items():
        if len(read_obj['transitions']) - 1 in stair_nums:
            read_ids.append(read_id)
    if read_num > 0:
        np.random.seed(seed)
        np.random.shuffle(read_ids)
        read_ids = read_ids[:read_num]
    obj = extract_reads_as_an_obj(obj, read_ids=read_ids)
    return obj
    


def substrac_signal_of_an_obj(
    obj: dict,
    tail: int = 0,
    in_place: bool = False,
):
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    for read_id, read_obj in new_obj.items():
        read_obj['signal'] = read_obj['signal'][-tail:]
    
    if in_place:
        return None
    else:
        return new_obj
    

def set_att_for_an_obj(
    obj: dict,
    atts: Literal['mean_of_I/I0', 'std_of_I/I0', 'median_of_I/I0', 'window_length'] = ['mean_of_I/I0', 'std_of_I/I0', 'window_length'],
    in_place: bool = False,
    scale_by_openpore: bool = True,
) -> Union[dict, None]:
    assert np.all(np.isin(atts, ['mean_of_I/I0', 'std_of_I/I0', 'median_of_I/I0', 'window_length'])) == True

    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    for read_id, read_obj in new_obj.items():
        if 'window' in read_obj and read_obj['window'] != None:
            x = read_obj['signal'][read_obj['window'][0]:read_obj['window'][1]]
        else:
            x = read_obj['signal']
        
        x = x.astype(np.float32)

        if scale_by_openpore:
            x = x/read_obj['OpenPore']

        if 'mean_of_I/I0' in atts:
            read_obj['mean_of_I/I0'] = np.mean(x)
        if 'std_of_I/I0' in atts:
            read_obj['std_of_I/I0'] = np.std(x)
        if 'median_of_I/I0' in atts:
            read_obj['median_of_I/I0'] = np.median(x)
        if 'window_length' in atts:
            if 'window' in read_obj and read_obj['window'] != None:
                read_obj['window_length'] = read_obj['window'][1] - read_obj['window'][0]
            else:
                read_obj['window_length'] = None

    if in_place:
        return None
    else:
        return new_obj
    

# ref to https://stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python
def _remove_spikes(
    signal: np.array,
    smooth_method: Literal['median_filter', 'ewma_fb'] = 'median_filter',
    high_clip: float = 0.8, 
    low_clip: float = 0.0,
    span: int = 100,
    delta: float = 0.05,
):
    clipped_signal = clip_data(signal, high_clip=high_clip, low_clip=low_clip)
    if smooth_method == 'median_filter':
        smooth_signal = median_filter(clipped_signal, size=span)
    elif smooth_method == 'ewma_fb':
        smooth_signal = ewma_fb(clipped_signal, span=span)
    # remove_outlier_signal = remove_outliers(clipped_signal, smooth_signal, delta=delta)
    # clean_signal = pd.Series(remove_outlier_signal).interpolate().values
    clean_signal = np.where(np.abs(clipped_signal - smooth_signal) > delta, smooth_signal, clipped_signal)
    return clean_signal, clipped_signal, smooth_signal

def remove_spikes(
    signal: np.array,
    smooth_method: Literal['median_filter', 'ewma_fb'] = 'median_filter',
    high_clip: float = 300.0, 
    low_clip: float = 0.0,
    span: int = 30,
    delta: float = 20,
    head_keeping_len: int = 200,
    tail_keeping_len: int = 200,
) -> np.array:
    """remove spikes within the signal. the head and tail of signal would not be changed.

    Args:
        signal (np.array): 1d array
        high_clip (float, optional): change values higher than this value as this value. Defaults to 300.0.
        low_clip (float, optional): change values lower than this value as this value. Defaults to 0.0.
        span (int, optional): the span of ewm or size of median_filter. Defaults to 100. suggest 100 for ewm and 30 for median_filter.
        delta (float, optional): if the distance between smoothed and raw value larger than this value,
                                set raw value as smoothed value. Defaults to 20.
        head_keeping_len (int, optional): the head of this number would not be changes. Defaults to 200.
        tail_keeping_len (int, optional): the tail of this number would not be changes. Defaults to 200.

    Returns:
        np.array: signal after removing spikes
    """
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    clean_signal, clipped_signal, smooth_signal = _remove_spikes(
        signal=signal,
        smooth_method=smooth_method,
        high_clip=high_clip, 
        low_clip=low_clip,
        span=span,
        delta=delta,
    )
    clean_signal[0:head_keeping_len] = signal[0:head_keeping_len]
    clean_signal[-tail_keeping_len:] = signal[-tail_keeping_len:]
    return clean_signal, clipped_signal, smooth_signal
    



def ewma_fb(x, span):
    ''' Apply forwards, backwards exponential weighted moving average (EWMA) to df_column. '''
    # Forwards EWMA.
    fwd = pd.Series(x).ewm(span=span).mean()
    # Backwards EWMA.
    bwd = pd.Series(x[::-1]).ewm(span=span).mean()
    # Add and take the mean of the forwards and backwards EWMA.
    stacked_ewma = np.vstack(( fwd, bwd[::-1] ))
    fb_ewma = np.mean(stacked_ewma, axis=0)
    return fb_ewma

# def remove_outliers(spikey, fbewma, delta):
#     ''' Remove data from df_spikey that is > delta from fbewma. '''
#     cond_delta = (np.abs(spikey-fbewma) > delta)
#     np_remove_outliers = np.where(cond_delta, np.nan, spikey)
#     return np_remove_outliers

def clip_data(unclipped, high_clip, low_clip):
    ''' Clip unclipped between high_clip and low_clip. 
    unclipped contains a single column of unclipped data.'''
    
    # clip data above HIGH_CLIP or below LOW_CLIP
    # cond_clip = (unclipped > high_clip) | (unclipped < low_clip)
    # np_clipped = np.where(cond_clip, np.nan, unclipped)
    np_clipped = np.clip(unclipped, low_clip, high_clip)
    return np_clipped


def extract_read_with_att_range(
    obj: dict,
    att: str,
    att_min: float = -np.inf,
    att_max: float = np.inf,
) -> dict:
    assert att in obj[list(obj.keys())[0]]

    new_obj = {}
    for read_id, read_obj in obj.items():
        if read_obj[att] >= att_min and read_obj[att] <= att_max:
            new_obj[read_id] = read_obj
    
    return new_obj



def smooth_signal_for_an_obj(
    obj: dict,
    in_place: bool = False,
    span: int = 200,
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)
    
    for read_id, read_obj in new_obj.items():
        smoothed_signal = ewma_fb(read_obj['signal'], span=span)
        read_obj['smoothed_signal'] = np.array(pd.Series(smoothed_signal).interpolate())
    
    if in_place:
        return None
    else:
        return new_obj


def down_sampling(
    array: np.array, 
    down_sample_to: int = 1000
) -> np.array:
    total_length = len(array)
    
    if total_length < down_sample_to:
        return np.concatenate((array,[0 for i in range(down_sample_to-total_length)]))
    
    sample_idx = np.round(np.linspace(start=0, stop=total_length-1, num=down_sample_to)).astype(np.int16)
    return array[sample_idx]


def down_sample_signal_for_an_obj(
    obj: dict,
    att: str = 'signal',
    new_att: str = 'dowm_sample',
    down_sample_to: int = 1000,
    target_window: bool = True,
    in_place: bool = False,
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)
    
    for read_id, read_obj in new_obj.items():
        x = read_obj[att]
        if target_window:
            s, e = read_obj['window']
            x = x[s:e]
        x = down_sampling(x, down_sample_to=down_sample_to)
        read_obj[new_att] = x
    
    if in_place:
        return None
    else:
        return new_obj


def extract_x_reads_randomly(
    obj: dict,
    seed: int = 0,
    read_num: int = 100,
) -> dict:
    np.random.seed(seed)
    all_read_ids = list(obj.keys())
    np.random.shuffle(all_read_ids)
    select_read_ids = all_read_ids[0:read_num]
    sub_obj = extract_reads_as_an_obj(
        obj=obj,
        read_ids=select_read_ids,
    )
    return sub_obj


def get_heights_of_first_stairs_for_an_obj(
    obj: dict, 
    sample_name: str = 'sample'
) -> pd.DataFrame:
    read_ids, heights = [], []
    for read_id, read_obj in obj.items():
        stair_signal = read_obj['stair_signal'] / read_obj['OpenPore']
        heights.append(stair_signal[0])
        read_ids.append(read_id)
    h_df = pd.DataFrame({'read_id': read_ids, 'first_stair': heights})
    h_df['sample'] = sample_name
    return h_df


def nor_stair_signal_for_an_obj_by_first_stair(
    obj: dict,
    first_stair_height: float = 0.3,
    in_place: bool = False,
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    for read_id, read_obj in new_obj.items():
        read_obj['stair_signal_norm'] = read_obj['stair_signal'] / read_obj['stair_signal'][0] * first_stair_height

    if in_place:
        return None
    else:
        return new_obj
    

def get_nor_stair_height_df(
    obj: dict,
    target_signal_att: str = 'stair_signal_norm',
) -> pd.DataFrame:
    stair_height_dfs = []
    for read_id, read_obj in obj.items():
        stair_height_dfs.append(read_obj[target_signal_att][read_obj['transitions'][0:-1]])
    stair_height_dfs = pd.DataFrame(stair_height_dfs)
    stair_height_dfs.columns = [f'stair{i}' for i in range(stair_height_dfs.shape[1])]
    stair_height_dfs = stair_height_dfs.stack().reset_index()
    stair_height_dfs.columns = ['read_id', 'stair_num', 'stair_height']
    return stair_height_dfs



def filter_obj_by_stair_height_cutoffs(
    obj: dict,
    cutoff_config,
    target_pep: str,
) -> dict:
    new_obj = {}

    for read_id, read_obj in obj.items():
        stair_heights = read_obj['stair_signal_norm'][read_obj['transitions'][0:-1]]
        flag = True
        for i, stair_height in enumerate(stair_heights):
            if i == 0:
                continue
            stair_min = cutoff_config.getfloat(target_pep, f'stair{i}_min')
            stair_max = cutoff_config.getfloat(target_pep, f'stair{i}_max')
            if stair_height < stair_min or stair_height > stair_max:
                flag = False
                break
        if flag:
            new_obj[read_id] = read_obj

    return new_obj

        
    
def sample_1d_array(
    X: np.ndarray,
    sample_num: int = 1000,
) -> np.ndarray:
    x = np.linspace(0, 1, num=len(X))
    f = interpolate.interp1d(x, X)
    new_x = np.linspace(0, 1, num=sample_num)
    new_X = f(new_x)
    return new_X


def get_bound_by_iqr(
    a: np.array,
) -> Tuple[float, float]:
    q1 = np.quantile(a, 0.25)
    q3 = np.quantile(a, 0.75)
    iqr = q3-q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    return lower_bound, upper_bound


def get_iqr(
    a: np.array
) -> float:
    q1 = np.quantile(a, 0.25)
    q3 = np.quantile(a, 0.75)
    iqr = q3-q1

    return iqr


def set_min_max_stair_signal_for_an_obj(
    obj: dict,
    new_stt: str = 'min_max_stair_signal',
) -> None:
    for read_id, read_obj in obj.items():
        stair0_index, stair_second_to_last_index = read_obj['transitions'][0], read_obj['transitions'][-3]
        stair0_signal = read_obj['stair_signal'][stair0_index]
        stair_second_to_last_signal = read_obj['stair_signal'][stair_second_to_last_index]
        read_obj[new_stt] = (read_obj['stair_signal'] - stair0_signal) / (stair_second_to_last_signal - stair0_signal)
    return None
