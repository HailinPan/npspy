import random
import numpy as np
import pandas as pd
import copy


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
        sub_obj[read_id] = obj[read_id]
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
    atts: Literal['mean_of_I/I0', 'std_of_I/I0', 'median_of_I/I0', 'window_length'] = ['mean_of_I/I0', 'std_of_I/I0'],
    in_place: bool = False,
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

        if 'mean_of_I/I0' in atts:
            read_obj['mean_of_I/I0'] = np.mean(x/read_obj['OpenPore'])
        if 'std_of_I/I0' in atts:
            read_obj['std_of_I/I0'] = np.std(x/read_obj['OpenPore'])
        if 'median_of_I/I0' in atts:
            read_obj['median_of_I/I0'] = np.median(x/read_obj['OpenPore'])
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
def remove_spikes(
    signal: np.array,
    high_clip: float = 0.8, 
    low_clip: float = 0.0,
    span: int = 100,
    delta: float = 0.05,
):
    clipped_signal = clip_data(signal, high_clip=high_clip, low_clip=low_clip)
    ewma_signal = ewma_fb(clipped_signal, span=span)
    remove_outlier_signal = remove_outliers(clipped_signal, ewma_signal, delta=delta)
    clean_signal = pd.Series(remove_outlier_signal).interpolate()
    return clean_signal



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

def remove_outliers(spikey, fbewma, delta):
    ''' Remove data from df_spikey that is > delta from fbewma. '''
    cond_delta = (np.abs(spikey-fbewma) > delta)
    np_remove_outliers = np.where(cond_delta, np.nan, spikey)
    return np_remove_outliers

def clip_data(unclipped, high_clip, low_clip):
    ''' Clip unclipped between high_clip and low_clip. 
    unclipped contains a single column of unclipped data.'''
    
    # clip data above HIGH_CLIP or below LOW_CLIP
    cond_clip = (unclipped > high_clip) | (unclipped < low_clip)
    np_clipped = np.where(cond_clip, np.nan, unclipped)
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
        read_obj['smoothed_signal'] = pd.Series(smoothed_signal).interpolate()
    
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
    in_place: bool = False,
) -> Union[dict, None]:
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)
    
    for read_id, read_obj in new_obj.items():
        x = read_obj[att]
        x = down_sampling(x, down_sample_to=down_sample_to)
        read_obj[new_att] = x
    
    if in_place:
        return None
    else:
        return new_obj

