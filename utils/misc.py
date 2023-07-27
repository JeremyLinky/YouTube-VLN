import random
import logging
from pathlib import Path
import sys
import numpy as np
import torch
import torch.distributed as dist
from glob import glob
import re
import time
import os
import socket
from shlex import quote
import zipfile

from colorama import init
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
from termcolor import cprint 
from pyfiglet import figlet_format


def get_logger(name: str, file: str = "") -> logging.Logger:
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger(name)
    if file !="":
        fh = logging.FileHandler(file)
        logger.addHandler(fh)
    return logger



def set_seed(args) -> None:
    if args.seed:
        print("Setting the seed")
        seed = args.seed
        if args.local_rank != -1:
            seed += args.local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)  # type: ignore
        random.seed(seed)

def is_default_gpu(args) -> bool:
    """
    check if this is the default gpu
    """
    return args.local_rank == -1 or dist.get_rank() != 0


def get_output_dir(args, sep='/train') -> Path:

    odir = increment_path(Path(args.output_dir) / args.save_name, increment=True, sep=sep, note=args.note)

    return odir.resolve()

def increment_path(path, increment=True, sep='', note = ''):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)  # make directory
    if increment:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 1  # increment number
        time_now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    path = Path(f"{path}{sep}{n}_{time_now}_{note}_{suffix}")  # increment path
    return path

def save_code(name, file_list=None,ignore_dir=['']):
    with zipfile.ZipFile(name, mode='w',
                        compression=zipfile.ZIP_DEFLATED) as zf:
        if file_list is None:
            file_list = []

        first_list = []
        for first_contents in ['*']:
            first_list.extend(glob(first_contents, recursive=True))

        for dir in ignore_dir:
            if dir in first_list:
                first_list.remove(dir)
        patterns = [x + '/**' for x in first_list]
        for pattern in patterns:
            file_list.extend(glob(pattern, recursive=True))

        file_list = [x[:-1] if x[-1] == "/" else x for x in file_list]
        for filename in file_list:
            zf.write(filename)

def exp_saver(save_folder, run_type, stage):
    sh_n_codes_path = Path(os.path.join(save_folder,'sh_n_codes'))
    if not sh_n_codes_path.exists():
        sh_n_codes_path.mkdir(parents=True, exist_ok=True)  # make directory
    sh_n_codes_path = str(sh_n_codes_path)
    with open(sh_n_codes_path +'/run_{}_{}_{}.sh'.format(run_type,socket.gethostname(), stage), 'w') as f:
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('unzip -d {}/code {}\n'.format(sh_n_codes_path, os.path.join(sh_n_codes_path, 'code.zip')))
        f.write('cp -r -f {} {}\n'.format(os.path.join(sh_n_codes_path, 'code', '*'), quote(os.getcwd())))
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')

    save_code(os.path.join(sh_n_codes_path, 'code.zip'),
            ignore_dir=['youtube_data', 'data', 'result', 'results','.vscode','__pycache__'])


def logo_print():
    cprint(figlet_format('YouTube VLN', font='slant'),
        'blue')

class NoneLogger():
  def info(self, msg, *args, **kwargs):
        pass