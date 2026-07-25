import os
import sys
import glob
import tqdm
import time
import argparse

from joblib import Parallel, delayed

def set_gpus(gpus: str='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

def get_gpus(key='CUDA_VISIBLE_DEVICES'):
    try: os.environ[key]
    except KeyError: os.environ[key] = '0'
    return os.environ[key].split(',')

def is_linux():
    return sys.platform == 'linux'

def create_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def listdir(dir_path):
    if '*' in dir_path: return glob.glob(dir_path)
    else: return os.listdir(dir_path)

def progress(data, desc=None):
    return tqdm.tqdm(data, desc)

def cpus(): 
    return os.cpu_count()

def parallel(fn, data_list, n_jobs=cpus(), show=False):
    print_dict = {}
    if show: print_dict = {'verbose': 10, 'pre_dispatch': 'all'}
    return Parallel(n_jobs, **print_dict)([delayed(fn)(*data) for data in data_list])

def log(string, path=None):
    print(string)
    if path is not None: 
        open(path, 'a+').write(string+'\n')

def get_num_digits(number):
    count = 0
    while number > 0:
        count += 1
        number //= 10
    return count

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def get(self):
        end = time.time()
        interval = end - self.start

        # print(interval, self.start, end)
        
        self.start = end
        return interval
    
    def __exit__(self, _type, _value, _trackback):
        pass

class Progress:
    def __init__(self, total, desc):
        self.__enter__(total, desc)

    def __enter__(self, total, desc):
        self.pbar = tqdm.tqdm(total=total, desc=desc)
        return self
    
    def update(self):
        self.pbar.update(1)
        
    def __exit__(self, _type, _value, _trackback):
        pass

class Parser:
    def __init__(self, input_dict: dict=None):
        self.parser = argparse.ArgumentParser()
        if input_dict is not None:
            self.add_from_inputs(input_dict)
    
    def add(self, tag, default):
        if isinstance(default, bool): option = {'action': 'store_true'}
        elif isinstance(default, list): option = {'nargs': '+', 'type': type(default[0])}
        else: option = {'default': default, 'type': type(default)}
        self.parser.add_argument(f'--{tag}', **option)

    def add_from_inputs(self, inputs):
        if isinstance(inputs, list):
            for data in inputs:
                self.add(*data)
        else:
            for tag in inputs.keys():
                self.add(tag, inputs[tag])
        return self.get()

    def get(self): 
        return self.parser.parse_args()
