import os
from tqdm import tqdm
import argparse
import pandas as pd
import json
from os import listdir

parser = argparse.ArgumentParser(description='process episodes')
parser.add_argument('--location', default='data/YouTube-VLN', help='folder root of what to process')
parser.add_argument('--caption_path', default='CLIP_captioned_images', help='use which caption (ie,clip,blip)')
parser.add_argument('--feather_note', default='', help='folder root of what to process')
args = parser.parse_args()

actions = pd.read_feather(f'{args.location}/inverse_data{args.feather_note}.feather')

actions_dict = actions.to_dict()

new_actions_dict = {actions_dict['before_image'][key]: key for key in actions_dict['before_image'].keys()}

action_map = {
    0: "forward",
    1: "left",
    2: "right"
}

videos = listdir(f'{args.location}/trajectory{args.feather_note}/')

os.makedirs(f'{args.location}/inverses_actions{args.feather_note}/', exist_ok=True)
exist_videos = os.listdir(f'{args.location}/inverses_actions{args.feather_note}/')
new_videos = list(set(videos) - set(exist_videos))
# videos = ['_1t6z04Ir0g']


for vid in tqdm(videos):
    action_generation = []
    caption_generation = []
    sample_tradj = json.load(open(f'{args.location}/trajectory{args.feather_note}/{vid}/trajectory.json'))
    tradj_captions = json.load(open(f'{args.location}/{args.caption_path}/{vid}/captions_SE.json'))
    for path in sample_tradj:
        tradj = sample_tradj[path]['images_trajectory']
        actions_per_tradj = {}
        captions_per_tradj = {}
        for i in range(len(tradj)):
            if i < len(tradj)-1:
                action_i = actions_dict['inverse_actions'][new_actions_dict[f'{args.location}/raw_frames/{vid}/%04d.jpg'%(tradj[i])]]
                actions_per_tradj[tradj[i]] = [action_map[action_i]]
                captions_per_tradj[tradj[i]] = tradj_captions[f'%04d.jpg'%(tradj[i])][0]
            elif i == len(tradj)-1:
                captions_per_tradj[tradj[i]]= tradj_captions[f'%04d.jpg'%(tradj[i])][0]
                
        action_generation.append(actions_per_tradj)
        caption_generation.append(captions_per_tradj)
    
    if not os.path.isdir(f'{args.location}/inverses_actions{args.feather_note}/{vid}/'):
        os.makedirs(f'{args.location}/inverses_actions{args.feather_note}/{vid}/')                    
    with open(f'{args.location}/inverses_actions{args.feather_note}/{vid}/inverses_actions.json', 'w') as f:
        json.dump(action_generation, f)

    if not os.path.isdir(f'{args.location}/{args.caption_path}{args.feather_note}/{vid}/'):
        os.makedirs(f'{args.location}/{args.caption_path}{args.feather_note}/{vid}/')                    
    with open(f'{args.location}/{args.caption_path}{args.feather_note}/{vid}/captions_pro.json', 'w') as f:
        json.dump(caption_generation, f)
