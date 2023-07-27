import os
import numpy as np
from tomlkit import key
import torch
from tqdm import tqdm
import argparse
import scripts.inverse_action.inverse_model as inverse_action
import pandas as pd
from torch.utils import data
from scripts.inverse_action.image_streams import ImageStream
from os import listdir
import csv
import json
import feather
import random

detection_thresholds = [
    0.9700177907943726, 0.9738382697105408, 0.9512060284614563,
    0.7334915995597839, 0.7058018445968628
]


score = lambda x: 0 if x is None else x.max()
score_detections = np.vectorize(score)
score_vals = lambda x: score_detections(x[:, 1])


def get_parser() -> argparse.ArgumentParser:
    """Return an argument parser."""
    parser = argparse.ArgumentParser(description='process episodes')
    
    parser.add_argument('-g',
                        '--gpu',
                        dest='gpu',
                        default='6',
                        help='which gpu to run on')

    parser.add_argument(
        "--location",
        default="data/YouTube-VLN",
        type=str,
        help="folder root of what to process"
    )

    parser.add_argument(
        "--feature",
        default="data/YouTube-VLN/youtube_img_features/check/total-error.tsv",
        type=str,
        help=""
    )

    parser.add_argument(
        "--reset_feather",
        default=False,
        action="store_true",
        help="whether you want to regenerate the actions of the whole video"
    )

    # improvements from v5
    parser.add_argument(
        "--use_clip_filter_person",
        action="store_false",
        help=""
    )     
    parser.add_argument(
        "--use_se_key_frames",
        action="store_false",
        help=""
    )
    parser.add_argument(
        "--use_resnet_filters",
        action="store_false",
        help=""
    )        
    parser.add_argument(
        "--use_random_key_frames",
        default=False,
        action="store_true",
        help=""
    )        
    parser.add_argument(
        "--feather_note",
        default="",
        type=str,
        help=""
    )
    ########################
    parser.add_argument(
        "--use_maskrcnn_filters",
        action="store_false",
        help=""
    )          
    parser.add_argument(
        "--caption_path",
        default="CLIP_captioned_images",
        type=str,
        help=""
    )

    return parser

# forward only
def calculate_steps(rewards):
    target_locs = []

    # identify steps with at target location
    for img_idx in range(len(rewards)):
        if rewards[img_idx]:
            target_locs.append(img_idx)

    steps = []
    for img_idx in range(len(rewards)):
        possible = list(filter(lambda x: x >= img_idx, target_locs))
        if len(possible) > 0:
            steps.append(min(possible) - img_idx)
        else:
            steps.append(float('inf'))
    steps = np.array(steps)
    return steps



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))   # str, arg_type
        
    inverse_actions = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    videos = listdir(f'{args.location}/raw_frames/')
    print(f'initial videos:{len(videos)}')

    model_path = 'data/YouTube-VLN/model4youtube/inverse_model.torch'
    if not os.path.exists(model_path):
        import urllib.request
        url = 'http://matthewchang.web.illinois.edu/data/inverse_model.torch'
        print("\n\n\nDownloading Inverse Model File...")
        urllib.request.urlretrieve(url,model_path)

    # set up model
    model = inverse_action.model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.cuda()
    
    filter_up_data = {}
    filter_video_data = {}
    filter_frame = {}

    # read the existed videos
    if os.path.exists(f'{args.location}/inverse_data{args.feather_note}.feather'):
        exist_df = feather.read_dataframe(f'{args.location}/inverse_data{args.feather_note}.feather')
    else:
        exist_df = {'vid_id': []}

    if args.reset_feather:
        to_inverse_videos = videos
    else:
        to_inverse_videos = list(set(videos)-set(exist_df['vid_id']))
    

    skip = []
    all_samples = pd.DataFrame()
    for vid in tqdm(to_inverse_videos):
        if vid in skip:
            continue
        # baisc filters
        if args.use_maskrcnn_filters:
            filters = np.load(f'{args.location}/indoor_frames__maskrcnn_coco/{vid}/{vid}.npy',
                          allow_pickle=True)[()]
        else:
            filters = {
                'indoor_locs': [f'{i:04d}.jpg' for i in range(151)],
                'person_locs':[]
            }

        resnet_place365_filters = []
        if args.use_resnet_filters:
            with open(f'{args.location}/indoor_frames_resnet_place365/{vid}/indoor.tsv', "r") as fid:
                reader = csv.DictReader(fid, delimiter='\t',fieldnames=["listing_id", "photo_id"])
                for item in reader:
                    resnet_place365_filters.append(int(item['photo_id']))

        clip_filters = []
        if args.use_clip_filter_person:
            with open(f'{args.location}/{args.caption_path}/{vid}/captions_SE.json', "r") as fid:
                json_reader = json.load(fid)
                for item in json_reader:
                    if json_reader[item][0].split(' with ')[1] == 'person':
                        clip_filters.append(item)

        feature_filters = []
        with open(args.feature, "r") as fid:
            reader = csv.DictReader(fid, delimiter='\t',fieldnames=["listing_id", "photo_id"])
            for item in reader:
                if item['listing_id'] == vid:
                    feature_filters.append(int(item['photo_id']))

        valid_frame = lambda x: f'%04d.jpg' % (x) in filters['indoor_locs'] and \
                                (x in resnet_place365_filters if args.use_resnet_filters else True) and \
                                f'%04d.jpg' % (x) not in clip_filters and \
                                f'%04d.jpg' % (x) not in filters['person_locs'] and\
                                x not in feature_filters

        filename = lambda x: f'{args.location}/raw_frames/{vid}/%04d.jpg' % (x)
        valid_frame_new = lambda x: os.path.exists(filename(x))
        all_frame = listdir(f'{args.location}/raw_frames/{vid}')
        key_frames = []

        if vid in filter_frame:
            front = filter_frame[vid]["front"]
            back = filter_frame[vid]["back"]
        else:
            front = 0
            back = 0
        
        if front < 0:
            front += len(all_frame)+1
        if back < 0:
            back += len(all_frame)+1

        if args.use_se_key_frames:
            # leverage shannon entropy to decide the key frames
            last_room_type = None
            last_room_SE = []
            last_room_SE_index = []

            with open(f'{args.location}/{args.caption_path}/{vid}/captions_SE.json', "r") as fid:
                json_reader = json.load(fid)
                valid_frames = []
        


                for x in range(1,len(all_frame)+1):
                    if front!= 0 and x <= front:
                        continue
                    if back!= 0 and x >= back:
                        continue
                    
                    if valid_frame(x) and valid_frame_new(x):
                        valid_frames.append(f'%04d.jpg' % (x))
                        if json_reader[f'%04d.jpg' % (x)][0].split(' with ')[0] != last_room_type:

                            # After seeing a new room type, look for the frame with the least information entropy in the previous room type
                            if last_room_type != None:        
                                key_frames.append(last_room_SE_index[last_room_SE.index(min(last_room_SE))])
                            # clear
                            last_room_SE = []
                            last_room_SE_index = []
                            # Add information about seeing this room type for the first time
                            last_room_SE.append(json_reader[f'%04d.jpg' % (x)][1]['ShannonEntropy'])
                            last_room_SE_index.append(x)
                            last_room_type = json_reader[f'%04d.jpg' % (x)][0].split(' with ')[0]

                        else:
                            last_room_SE.append(json_reader[f'%04d.jpg' % (x)][1]['ShannonEntropy'])
                            last_room_SE_index.append(x)
                    # Prevent the situation where there is only one type of group
                    if x == len(listdir(f'{args.location}/raw_frames/{vid}')) and \
                        last_room_type == json_reader[f'%04d.jpg' % (x)][0].split(' with ')[0]:
                        key_frames.append(last_room_SE_index[last_room_SE.index(min(last_room_SE))])
        
        elif args.use_random_key_frames:
            keyframe_sources = []
            for x in range(1,len(all_frame)+1):
                if front!= 0 and x <= front:
                    continue
                if back!= 0 and x >= back:
                    continue
                if valid_frame(x) and valid_frame_new(x):
                    keyframe_sources.append(x)
            if len(keyframe_sources) == 0:
                continue
            keyframe_len = random.randint(1, len(keyframe_sources))
            key_frames = random.sample(keyframe_sources,keyframe_len)
            

        current_range_id = 0
        trajectory_infos ={}
        episode_ranges = [[]]

        key_frames.sort()

        episode_ranges[current_range_id] = key_frames

        for t_id, trajectory in enumerate(episode_ranges):
            trajectory_id = f'path {t_id+1}'
            trajectory_infos[trajectory_id] = {}
            trajectory_infos[trajectory_id]['images_trajectory'] = trajectory

            for j in range(0,len(trajectory)-1):
                start = trajectory[j]
                stop = trajectory[j+1]

                samples = []

                samples.append((filename(start), filename(stop),vid,start,stop))

                sample_frame = pd.DataFrame(
                    samples,
                    columns=['before_image', 'after_image','vid_id','im_start','im_stop'])

                all_samples = pd.concat((all_samples, sample_frame))

        if not os.path.isdir(f'{args.location}/trajectory{args.feather_note}/{vid}/'):
            os.makedirs(f'{args.location}/trajectory{args.feather_note}/{vid}/')
        if not os.path.isdir(f'{args.location}/trajectory{args.feather_note}/{vid}/trajectory.json'):                    
            with open(f'{args.location}/trajectory{args.feather_note}/{vid}/trajectory.json', 'w') as f:
                json.dump(trajectory_infos, f)
     

    print("Start inverse labeling")
    all_acts = []

    ims = np.stack(
        (all_samples['before_image'], all_samples['after_image']), axis=1)

    image_loader = data.DataLoader(ImageStream(ims),
                                    num_workers=4,
                                    batch_size=2)
    for be, ae in tqdm(image_loader):
        acts = model(be.cuda(), ae.cuda())[1]
        acts = acts.argmax(dim=1, keepdim=True).cpu().detach()
        all_acts.append(acts)

    all_samples['inverse_actions'] = torch.cat(all_acts).numpy()
    if args.reset_feather or type(exist_df) == dict:
        new_df = all_samples
    else:
        new_df = exist_df.append(all_samples)
    new_df.reset_index(drop=True,inplace=True)
    new_df.to_feather(f'{args.location}/inverse_data{args.feather_note}.feather')
