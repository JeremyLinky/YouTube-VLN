#!/usr/bin/env python

''' Script to precompute image features using a Caffe ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import argparse
import numpy as np
import base64
import csv
import sys
import os

csv.field_size_limit(sys.maxsize)


# CLIP Support
import torch
import clip
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, "scripts")



TSV_FIELDNAMES = ['video_id','frame_id','features']
IMAGE_SIZE = 1
parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='vit', help='architecture')
parser.add_argument('--images', default='data/YouTube-VLN/raw_frames/', type=str, help='nothing to help')
parser.add_argument('--outfile', default=None, type=str, help='nothing to help')
args = parser.parse_args()

# Labels Prob
if args.arch == "resnet":
    FEATURE_SIZE = 1024
    MODEL = "RN50"
    OUTFILE = 'data/YouTube-VLN/CLIP-ResNet-50-views.tsv'
elif args.arch == "vit":
    FEATURE_SIZE = 512
    MODEL = "ViT-B/32"
    OUTFILE = 'data/YouTube-VLN/CLIP_features/CLIP-ViT-B-32-views.tsv'
else:
    assert False

if args.outfile:
    OUTFILE = args.outfile

def read_tsv(infile):
    # Verify we can read a tsv
    exist_feature = []
    all_features = {}
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
        for item in reader:
            long_id = item['video_id'] + "_" + item['frame_id']
            all_features[long_id] = np.frombuffer(base64.decodebytes(item['features'].encode('ascii')),
                                          dtype=np.float32).reshape((IMAGE_SIZE, FEATURE_SIZE))
            exist_feature.append(item['video_id'])
    return exist_feature, all_features

def build_tsv(images):
    videos = os.listdir(images)

    try:
        exist_feature, all_features = read_tsv(OUTFILE)
    except:
        exist_feature, all_features = [], {}

    new_feature = list(set(videos))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL, device=device)
    if not os.path.exists(os.path.dirname(OUTFILE)):
        os.makedirs(os.path.dirname(OUTFILE))
    with open(OUTFILE, 'a') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)   
        for vid in tqdm(new_feature):
            img_path = os.path.join(images,vid)
            imgs = os.listdir(img_path)
   

            for img in imgs:
                if img.split(".")[-1] != "jpg":
                    continue
                if f'{vid}_{img}' in all_features.keys():
                    continue
                blobs = []
                blobs.append(Image.open(os.path.join(img_path, img)).convert("RGB"))

                blobs = [
                    preprocess(blob).unsqueeze(0)
                    for blob in blobs
                ]
                blobs = torch.cat(blobs, 0)
                blobs = blobs.to(device)

                features = model.encode_image(blobs).float()

                features = features.detach().cpu().numpy()
                writer.writerow({
                    'video_id': vid,
                    'frame_id': img,
                    'features': base64.b64encode(features).decode(),
                })


if __name__ == "__main__":

    build_tsv(args.images)
    data, _ = read_tsv(OUTFILE)
    print(f'Completed! There are {len(set(data))} videos\' features')

