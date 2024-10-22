import pickle
import numpy as np
import pandas as pd
import copy
import os
import re
import glob
import configparser
from configparser import ConfigParser, ExtendedInterpolation
from typing import Dict, Optional, Tuple, Union, Literal, List

def get_read_number(
    obj: dict,
    targets: list = ['total', 'read_with_window'],
):
    read_num_dict = {}
    for target in targets:
        read_num_dict[target] = 0
    for read_id, read_obj in obj.items():
        if 'total' in targets:
            read_num_dict['total'] += 1
        if 'read_with_window' in targets:
            if 'window' in read_obj and read_obj['window'] != None:
                read_num_dict['read_with_window'] += 1
    return read_num_dict
