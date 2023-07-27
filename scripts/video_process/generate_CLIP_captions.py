import os
import sys
sys.path.append(os.getcwd())

import csv
import json
import clip
import numpy as np
import base64
import torch
from tqdm import tqdm
import cv2
from math import log

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CLIP_features', default='data/YouTube-VLN/CLIP_features/CLIP-ViT-B-32-views.tsv', type=str, help='nothing to help')
parser.add_argument('--video_path', default='data/YouTube-VLN/raw_frames', type=str, help='nothing to help')
parser.add_argument('--outfile', default='data/YouTube-VLN/CLIP_captioned_images', type=str, help='nothing to help')
args = parser.parse_args()

# From Matterport3D
areas = ['office', 'lounge', 'family room', 'entry way', 'dining room', 'living room', 'stairs', 'kitchen',
         'porch', 'bathroom', 'bedroom', 'hallway']
objects = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa', 'bed',
           'curtain', 'chest of drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool', 'towel',
           'mirror', 'tv monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting', 'beam',
           'railing', 'shelving', 'blinds', 'gym equipment', 'seating', 'board panel', 'furniture',
           'appliances', 'clothes','person']
IMAGE_SIZE = 1
FEATURE_SIZE = 512
TOP_K = 1
font=cv2.FONT_HERSHEY_SIMPLEX
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
area_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in areas]).to(device)
object_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in objects]).to(device)
with torch.no_grad():
    area_text_features = model.encode_text(area_inputs)
    object_text_features = model.encode_text(object_inputs)

features = {}

tsv_fieldnames = ['video_id','frame_id','features']
with open(args.CLIP_features, "r") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
    for item in reader:
        long_id = item['video_id'] + "_" + item['frame_id']
        features[long_id] = np.frombuffer(base64.decodebytes(item['features'].encode('ascii')),
                                          dtype=np.float32).reshape((IMAGE_SIZE, FEATURE_SIZE))

video_path = args.video_path 
videos = os.listdir(video_path)
if not os.path.exists(args.outfile):
    os.makedirs(args.outfile)
cap_videos = os.listdir(args.outfile)
new_cap_videos = list(set(videos) - set(cap_videos))
# videos = videos[0:1] # debug
# videos = ['_1t6z04Ir0g'] # debug

# noun_phases = []
# candidate_infos = []

def ShannonEntropy(similarity):
    shannonEnt = 0.0
    for i in range(len(similarity[0])):
        p = similarity[0,i] + 1e-7
        shannonEnt -= p * log(p, 2)

    return float(shannonEnt.cpu())


captioned_images_root = args.outfile
raw_images_root = args.video_path
for vid in tqdm(videos):
    img_path = os.path.join(video_path, vid)
    imgs = os.listdir(img_path)

    if not os.path.exists(captioned_images_root + f"/{vid}/"):
        os.makedirs(captioned_images_root + f"/{vid}/")
    caption_infos = {}
    for img in imgs:
        if img.split(".")[-1] != "jpg":
            continue
        long_id = vid + '_' + img
        cand_feat = torch.from_numpy(features[long_id]).to(device).half()
        cand_feat /= cand_feat.norm(dim=-1, keepdim=True)
        area_text_features /= area_text_features.norm(dim=-1, keepdim=True)
        object_text_features /= object_text_features.norm(dim=-1, keepdim=True)
        area_similarity = (100.0 * cand_feat @ area_text_features.T).softmax(dim=-1)
        object_similarity = (100.0 * cand_feat @ object_text_features.T).softmax(dim=-1)
        area_pred = areas[area_similarity.topk(TOP_K)[1]]
        object_pred = objects[object_similarity.topk(TOP_K)[1]]
        SE = ShannonEntropy(area_similarity)
        caption_infos[img] = [area_pred + ' with ' + object_pred, \
                              {"ShannonEntropy": SE},\
                              {"similarity": float(area_similarity.topk(TOP_K)[0].cpu())}]
        
        # write caption to image
        # path = os.path.join(raw_images_root+'/'+vid, img)
        # x = cv2.imread(path)
        # blank_img=np.zeros((100,x.shape[1],3),np.uint8)
        # x_cap = np.vstack((x,blank_img))
        
        # image=cv2.putText(x_cap,area_pred + ' with ' + object_pred + '.',(40,x.shape[0]+40),font,1.2,(255,255,255),2)
        # cv2.imwrite(captioned_images_root + f"/{vid}/" + img, image)
        
    sorted_keys = sorted(caption_infos.keys())
    caption_infos_after = { e: caption_infos[e] for e in sorted_keys}    
    with open(captioned_images_root + f"/{vid}/captions_SE.json", 'w') as f:
        json.dump(caption_infos_after, f, indent=4)


