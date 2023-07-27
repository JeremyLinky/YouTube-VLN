"""
Create image merging datasets
"""
import csv
import sys
import json
import logging
import random
import json
from pathlib import Path
from itertools import groupby
from operator import itemgetter
from tqdm.auto import tqdm
import argtyped
import math
from typing import Tuple, List

random.seed(1)
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


def load_json(f):
    with open(f) as fid:
        return json.load(fid)


def save_json(data, f):
    with open(f, "w") as fid:
        json.dump(data, fid, indent=2)

def split_group_adjacent(group):
    for idx in range(len(group)-1):
        if group[idx+1]['frame'] - group[idx]['frame'] != 1:
            # not adjacent
            return group[:idx+1], group[idx+1:]
    else:
        return group, []

def split_group_continue(group):
    for idx in range(len(group)-1):
        if group[idx+1]['room'] != group[idx]['room']:
            # not adjacent
            return group[:idx+1], group[idx+1:]
    else:
        return group, []

class Arguments(argtyped.Arguments):
    source: Path = Path("data/YouTube-VLN/ytb/ytb_test.json")
    output: Path = Path("data/YouTube-VLN/ytb/merge+ytb_test.json")
    caption_dir: Path = Path("data/YouTube-VLN/CLIP_captioned_images")
    min_captioned: int = 2  # minimum of captioned images once merged
    min_length: int = 4  # minimum of images once merged
    max_photo_per_merging: int = 10
    merge_method: str = 'max'  # the method to merge photos (max, least, all, adjacent, continue)


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))
    if args.merge_method not in ['max', 'least', 'all', 'adjacent', 'continue']:
        raise ValueError(
            "please select collect merge_method from 'max', 'least', 'all' , 'adjacent' or 'continue'!"
            )
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info(f"Loading Ytb JSON from {args.source}")
    data = load_json(args.source)

    # gather frame_ids by video_id
    data = sorted(data, key=itemgetter("video_id"))
    frame_ids_by_video = {
        video_id: (sorted(list(frames), key=itemgetter("frame_id")))
        for video_id, frames in groupby(data, key=itemgetter("video_id"))
    }

    captions = {}
    
    for video_id, frames in tqdm(frame_ids_by_video.items()):
        caption_dir = args.caption_dir / video_id / 'captions_SE.json'
        captions[video_id] = {}
        captions[video_id]['data'] = load_json(caption_dir)
        
        # get room types from captions
        captions[video_id]['frame_list'] = [
            {
                'frame': frame['frame_id'],
                'caption': frame['caption'],
                'action': frame['action'],
                'captions': captions[video_id]['data'][f"%04d.jpg"%frame['frame_id']][0],
                'room': captions[video_id]['data'][f"%04d.jpg"%frame['frame_id']][0].split(' with ')[0]
            }
            for frame in frames
        ]

        # group frames by room
        captions[video_id]['room'] = {
            room: sorted(frames, key=itemgetter("frame"))
            for room, frames in groupby(sorted(captions[video_id]['frame_list'], key=itemgetter("room")), key=itemgetter("room"))
        }

        
        # merge adjacent frames of the same room type
        captions[video_id]['frame'] = {}


        # ungroup groups that contain too many photos
        # and each new groups must have at least one captioned image
        captions[video_id]['new_groups'] = {}
        if args.merge_method == 'least' or args.merge_method == 'adjacent' or args.merge_method == 'continue':
            if args.merge_method == 'least':    # Ensure that each group has a caption, group according to the caption, and then fill the other frames into the group, there may be more than max_photo_per_merging groups
                for room, frames in captions[video_id]['room'].items():
                    if len(frames) <= args.max_photo_per_merging:
                        captions[video_id]['new_groups'][room] = [frames]
                        continue
                    caption_stats = [
                        frame['caption'] == ''
                        for frame in frames
                    ]

                    num_groups = math.ceil(len(frames) / args.max_photo_per_merging)
                    num_captioned = sum(caption_stats)
                    num_groups = min(num_groups, num_captioned)
                    num_groups = max(num_groups, 1)
                    new_groups: List[List[Tuple[int, str, float]]] = [
                        [] for _ in range(num_groups)
                    ]

                    # each new groups must have at least one captioned image
                    # so we start by distributing them
                    done: List[int] = []
                    for idx, frame in enumerate(frames):
                        if caption_stats[idx]:
                            new_groups[len(done)] = [frame]
                            done.append(idx)
                            if len(done) == num_groups:
                                break

                    # then we distribute the remaining photos
                    for idx, frame in enumerate(frames):
                        if idx in done:
                            continue
                        new_groups[idx % num_groups].append(frame)

                    captions[video_id]['new_groups'][room] = new_groups

            elif args.merge_method == 'adjacent':   # Strictly ensure that the same room type and the frames are contiguous (invalid frames will also be split)
                for room, frames in captions[video_id]['room'].items():
                    new_groups: List[List[Tuple[int, str, float]]] = []
                    group = frames.copy()
                    while True:
                        new_group, group = split_group_adjacent(group)
                        new_groups.append(new_group)
                        if len(group) == 0:
                            break
                    captions[video_id]['new_groups'][room] = new_groups
            
            elif args.merge_method == 'continue':   # Ensure that there are no other room types between frames (regardless of invalid frames)
                new_groups_all: List[List[Tuple[int, str, float]]] = []
                group = captions[video_id]['frame_list'].copy()
                while True:
                    new_group, group = split_group_continue(group)
                    new_groups_all.append(new_group)
                    if len(group) == 0:
                        break
                for group in new_groups_all:
                    room = group[0]['room']
                    if room not in captions[video_id]['new_groups']:
                        captions[video_id]['new_groups'][room] = []
                    captions[video_id]['new_groups'][room].append(group)

            # update new_groups
            for new_groups in captions[video_id]['new_groups'].values():
                for frame_group in new_groups:
                    group = [
                        frame['frame']
                        for frame in frame_group
                    ]
                    for idx, frame in enumerate(frame_group):
                        captions[video_id]['frame'][frame['frame']] = {
                        'caption': frame['caption'],
                        'captions': frame['captions'],
                        'room': frame['room'],
                        'merging':  [group[idx]]+group[:idx]+group[idx+1:]    # the current frame is first
                    }
        else:
            for room, frames in captions[video_id]['room'].items():
                group = [frame['frame'] for frame in frames]
                group_len = len(group)
                for idx in range(group_len):
                    if args.merge_method == 'max':  # merge max_photo_per_merging/ Merging 2 frames
                        # merge up to args.max_photo_per_merging photos 
                        if group_len < args.max_photo_per_merging:
                            # too short
                            start = 0
                        elif idx < args.max_photo_per_merging/2:
                            # too early
                            start = 0
                        elif idx > group_len - args.max_photo_per_merging/2:
                            # too late
                            start = group_len - args.max_photo_per_merging
                        else:
                            start = idx - int(args.max_photo_per_merging/2)
                        end = start + args.max_photo_per_merging
                    elif args.merge_method == 'all':
                        # merge all photos in the same group
                        start = 0
                        end = None
                    else: 
                        raise ValueError(
                            "please select collect merge_method from 'max', 'least' and 'all'!"
                            )

                    frame = frames[idx]
                    captions[video_id]['frame'][frame['frame']] = {
                        'caption': frame['caption'],
                        'captions': frame['captions'],
                        'room': frame['room'],
                        'merging':  [group[idx]]+group[start:idx]+group[idx+1:end]    # the current frame is first
                    }

    # Export merging
    for item in data:
        video_id = item['video_id']
        frame_id = item['frame_id']
        item["merging"] = captions[video_id]['frame'][frame_id]['merging']
        item["room"] = captions[video_id]['frame'][frame_id]['room']

    logger.info(f"Outputting to {args.output}")
    save_json(data, args.output)
