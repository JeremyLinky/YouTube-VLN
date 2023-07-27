from os import listdir, mkdir, rename
import numpy as np
import re

import pandas as pd

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d as gaussian

import torchvision
import argparse


parser = argparse.ArgumentParser(description='filter frames')
parser.add_argument('-g',
                    '--gpu',
                    dest='gpu',
                    default='4',
                    help='which gpu to run on')
parser.add_argument('-d','--dump',action='store_true',help='dump frames from video files')
parser.add_argument('--frames_location',
                    default='data/YouTube-VLN/raw_frames',
                    help='location of extracted frames')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

torch.set_num_threads(4) 

io_labels = pd.read_csv('data/YouTube-VLN/model4youtube/io_places.txt', delimiter=' ', header=None)

io_dict = {}
for i in range(len(io_labels)):
    label = io_labels[0][i]
    label = label[3:]
    io_dict[label] = 2 - io_labels[1][i]

# PlacesCNN for scene classification
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

# the architecture to use
arch = 'alexnet'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access('data/YouTube-VLN/model4youtube/' + model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url + ' -P ' + 'data/YouTube-VLN/model4youtube')

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load('data/YouTube-VLN/model4youtube/' + model_file, map_location=lambda storage, loc: storage)
state_dict = {
    str.replace(k, 'module.', ''): v
    for k, v in checkpoint['state_dict'].items()
}
model.load_state_dict(state_dict)
model.eval()
model.cuda()

# load the image transformer
centre_crop = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'data/YouTube-VLN/model4youtube/categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url + ' -P ' + 'data/YouTube-VLN/model4youtube')
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
maskrcnn.eval()
maskrcnn.cuda()

def files(folder,regex = '.*',reject=None):
    fil = os.popen(f'ls {folder}').read().split()
    fils = [ e for e in fil if re.search(regex,e)]
    if reject is not None:
        fils = [ e for e in fil if not re.search(reject,e)]
    return fils

def classify_person(img):
    PERSON = 1
    im = trn.ToTensor()(img).cuda()
    predictions = maskrcnn([im])[0]['labels']
    return PERSON in predictions[:5]

def smooth(values, window):
    window = window // 2
    for i in range(window, len(values) - window):
        values[i] = round(np.mean(values[i - window:i + window]))

    return values


def classify_indoors(image_file):
    input_img = V(centre_crop(image_file).unsqueeze(0))
    logit = model.forward(input_img.cuda())
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    indoor_prob = 0
    for i in range(0, 10):
        if io_dict[classes[idx[i]]]:
            indoor_prob += probs[i].item()
    return indoor_prob


def get_labels(folder):
    person_labels = []
    io_labels = []
    image_files = files(folder,'\d+.jpg$')
    img_range = range(0, len(image_files))
    for img_no in tqdm(img_range):
        img = Image.open(folder + f'/{image_files[img_no]}')
        io_labels.append(classify_indoors(img))
        person_labels.append(classify_person(img))
        img.close()

    io_smoothed = gaussian(io_labels, sigma=6)
    io_smoothed = [round(io_smooth) for io_smooth in io_smoothed]
    person_smoothed = smooth(person_labels, 6)

    return np.array(io_smoothed), np.array(person_smoothed), image_files


def filter_frames(folder):
    indoor_range, person_range, image_files = get_labels(folder)
    indoor_locs = np.argwhere(indoor_range)
    person_locs = np.argwhere(person_range)
    data = {
        'indoor_locs': np.array(image_files)[indoor_locs].flatten(),
        'person_locs': np.array(image_files)[person_locs].flatten()
    }
    return data


# videos = [ f.name for f in os.scandir(args.frames_location) if f.is_dir()]

for video_id in os.listdir(args.frames_location):

    if not os.path.exists(f'{os.path.dirname(args.frames_location)}/indoor_frames_maskrcnn_coco/{video_id}/{video_id}.npy'):
        print(f'Processing video: {video_id}')
        data = filter_frames(f'{args.frames_location}/{video_id}')
        os.makedirs(f'{os.path.dirname(args.frames_location)}/indoor_frames_maskrcnn_coco/{video_id}/')
        np.save(f'{os.path.dirname(args.frames_location)}/indoor_frames_maskrcnn_coco/{video_id}/{video_id}',data)
    else:
        continue
    
