import pickle
import numpy as np
from random import uniform
import re
from pathlib import Path
import os
import math

def unzip(a):
    return tuple(map(list,zip(*a)))

def unzip_arrays(a):
    return [np.array(x) for x in unzip(a)]

# return true with probability rate
def rand_bool(rate):
    return uniform(0,1) < rate

def one_hot(n,i):
    r = np.zeros((n,))
    r[i] = 1
    return r

def save_obj(obj, name ):
    ensure_folders(name)
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    # add extension if not provided
    if '.' not in name: name += '.pkl'
    with open(name, 'rb') as f:
        return pickle.load(f)

# pads along the zero axis to the given length
# truncating beginning elements if the length
# is less than the given data
def padTo(length, dat):
    ds = list(dat.shape)
    ds[0] = length
    out = np.zeros(tuple(ds))
    if (len(dat) > length):
        # take last elements
        out = dat[-length:]
    elif (len(dat) > 0):
        out[-len(dat):] = dat
    return out

def sample_axis(mat,size,axis=0):
    indices = np.random.choice(mat.shape[axis],size=size,replace=False)
    slices = [slice(None)] * len(mat.shape)
    slices[axis] = indices
    return mat[tuple(slices)]

def ensure_folders(path,is_dir=False):
    if is_dir:
        Path(path).mkdir(parents=True, exist_ok=True)
    else:
        match = re.match("(.*)/.*?$",path)
        if match:
            Path(match[1]).mkdir(parents=True, exist_ok=True)


# shape should be an array containing the sizes of each
# column group to be broken down
def split_columns(obj,shape):
    st = 0
    res = []
    if obj.shape[-1] != sum(shape):
        raise Exception(f"shape sum {sum(shape)} incompatible with {obj.shape}")
    for i in shape:
        if len(obj.shape) == 2:
            res.append(obj[:,st:st+i])
        elif len(obj.shape) == 1:
            res.append(obj[st:st+i])
        else:
            raise Exception("bad shape")
        st = st+i
    return tuple(res)

def files(folder,regex = '.*',reject=None):
    fil = os.popen(f'ls {folder}').read().split()
    fils = [ e for e in fil if re.search(regex,e)]
    if reject is not None:
        fils = [ e for e in fil if not re.search(reject,e)]
    return fils

def dirs(folder,regex = '.*',reject=None):
    fil = os.popen(f"cd {folder}; ls -d */ | sed 's/.$//'").read().split() 
    fils = [ e for e in fil if re.search(regex,e)]
    if reject is not None:
        fils = [ e for e in fil if not re.search(reject,e)]
    return fils

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def argmax(li,func = lambda x: x):
    index, max_val,max_el = None,None,None
    for i,el in enumerate(li):
        val = func(el)
        if max_val is None or val > max_val:
            index, max_val,max_el = i, val,el
    return index,max_el,max_val

def argmin(li,func=lambda x: x):
    index, min_val,max_el = None,None,None
    for i,el in enumerate(li):
        val = func(el)
        if min_val is None or val < min_val:
            index, min_val,max_el = i, val,el
    return index,max_el,min_val


# only works on last axis right now
from functools import wraps
import inspect

def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)
    if defaults is None: defaults = []

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def chunks_num(lst, n):
    """Yield n evenly sized chunks from lst."""
    low = len(lst)//n
    rem = len(lst)-(low*n)
    counts = [low]*n
    for i in range(rem): counts[i] += 1
    ptr = 0
    res = []
    for count in counts:
        res.append(lst[ptr:ptr+count])
        ptr += count
    return res


def angle_delta(x,y):
    return math.atan2(math.sin(x-y), math.cos(x-y))
