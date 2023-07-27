import torch
from torch.utils import data
from PIL import Image
from scripts.inverse_action.util.torch import imageNetTransformPIL
import csv
import numpy as np
import os
import re
import scripts.inverse_action.util as util
import cv2

transform = imageNetTransformPIL()

class ImageStream(data.Dataset):
    def __init__(self,paths):
        self.paths = paths

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    # returns before image, after image, action, reward, terminal
    def __getitem__(self, index):
        'Generates one sample of data'
        row = self.paths[index]
        ims = [transform(Image.open(p)) for p in row]
        return tuple(ims)
