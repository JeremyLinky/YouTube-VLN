"""
Build a valid ytb_dataset
"""
import argparse
from os import listdir
from tqdm import tqdm
import numpy as np
import csv
import json

TSV_FIELDNAMES = [
    "video_id",
    "frame_id",
    "caption",
    "action"
]

ERROR_TSV_FIELDNAMES = [
    "video_id",
    "frame_id",
]

def get_parser() -> argparse.ArgumentParser:
    """Return an argument parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--location",
        default="data/YouTube-VLN",
        type=str,
        help="folder root of what to process"
    )
    parser.add_argument(
        "--output",
        default="data/YouTube-VLN/Extra",
        type=str,
        help=""
    )
    parser.add_argument(
        "--feature",
        default="data/YouTube-VLN/youtube_img_features/check/total-error.tsv",
        type=str,
        help=""
    )
    parser.add_argument(
        "--test_rate",
        default=0.05,
        type=float,
        help=""
    )
    parser.add_argument(
        "--use_resnet_filters",
        action="store_false",
        help=""
    )    
    parser.add_argument(
        "--use_clip_filter_person",
        action="store_false",
        help=""
    )
    parser.add_argument(
        "--feather_note",
        default="",
        type=str,
        help=""
    )
    parser.add_argument(
        "--use_maskrcnn_filters",
        action="store_false",
        help=""
    )      
    parser.add_argument(
        "--note",
        default="ytb",
        type=str,
        help=""
    )
    parser.add_argument(
        "--caption_dir",
        default="data/YouTube-VLN/CLIP_captioned_images",
        type=str,
        help=""
    )

    return parser


def build_dataset(args, videos, tag: str = "dataset", start_idx: int = 0, stop_idx: int = -1, filter_frame: dict = {}):
    
    with open(f"{args.output}/{tag}{args.feather_note}.tsv", "w", newline="") as out:
        writer = csv.DictWriter(
            out, delimiter="\t", fieldnames=TSV_FIELDNAMES
        )

        for vid in tqdm(range(start_idx, stop_idx)):
            vid = videos[vid]

            # detection result with dqn
            if args.use_maskrcnn_filters:
                filters = np.load(f'{args.location}/indoor_frames__maskrcnn_coco/{vid}/{vid}.npy',
                            allow_pickle=True)[()]
            else:
                filters = {
                    'indoor_locs': [f'{i:04d}.jpg' for i in range(151)],
                    'person_locs':[]
                }
            # detection result with resnet_place365
            resnet_place365_filters = []
            if args.use_resnet_filters:
                with open(f'{args.location}/indoor_frames_resnet_place365/{vid}/indoor.tsv', "r") as fid:
                    reader = csv.DictReader(fid, delimiter='\t',fieldnames=ERROR_TSV_FIELDNAMES)
                    for item in reader:
                        resnet_place365_filters.append(int(item['frame_id']))

            clip_filters = []
            if args.use_clip_filter_person:
                with open(f'{args.caption_dir}/{vid}/captions_SE.json', "r") as fid:
                    json_reader = json.load(fid)
                    for item in json_reader:
                        if json_reader[item][0].split(' with ')[1] == 'person':
                            clip_filters.append(item)

            # frame with error feature
            feature_filters = []
            if args.feature != "":
                with open(args.feature, "r") as fid:
                    reader = csv.DictReader(fid, delimiter='\t',fieldnames=ERROR_TSV_FIELDNAMES)
                    for item in reader:
                        if item['video_id'] == vid:
                            feature_filters.append(int(item['frame_id']))

            # filter and get valid frame
            all_frame = listdir(f'{args.location}/raw_frames/{vid}')
            valid_frame = []
            
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

            for x in range(1,len(all_frame)+1):
                if front!= 0 and x <= front:
                    continue
                if back!= 0 and x >= back:
                    continue

                filename = f'%04d.jpg' % (x)
                if filename in filters['indoor_locs'] and \
                    filename not in filters['person_locs'] and\
                    (x in resnet_place365_filters if args.use_resnet_filters else True) and \
                    filename not in clip_filters and \
                    x not in feature_filters:
                    valid_frame.append(x)

            ## save valid frame, caption and action
            try:
                trajectory = json.load(open(f"{args.location}/trajectory{args.feather_note}/{vid}/trajectory.json"))
            except:
                print(f"not such trajectory file {vid}")
                continue
            
            if len(trajectory) != 1:
                raise ValueError("Trajectory length is not 1")
            
            trajectory = trajectory['path 1']['images_trajectory']

            captions = json.load(open(f"{args.caption_dir}/{vid}/captions_SE.json"))

            actions = json.load(open(f"{args.location}/inverses_actions{args.feather_note}/{vid}/inverses_actions.json"))[0]

            for frame in valid_frame:
                if frame in trajectory:
                    # save caption and action of key frames
                    caption = captions['%04d.jpg' % (frame)][0]
                    if frame == trajectory[-1]:
                        # there is no action in the last frame
                        action = ""
                    else:
                        action = actions[str(frame)][0]
                else:
                    caption = ""
                    action = ""
                writer.writerow({
                    "video_id": vid,
                    "frame_id": frame,
                    "caption": caption,
                    "action": action
                })

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))   # str, arg_type

    videos = listdir(f'{args.location}/raw_frames/')
    print(f'initial videos:{len(videos)}')
    test_size = int(len(videos)*args.test_rate)
    
    # # all dataset
    # build_dataset(args, videos=videos,tag="ytb_dataset", start_idx=0, stop_idx=len(videos))

    test_size = int(len(videos)*args.test_rate)
    # test dataset
    build_dataset(args, videos=videos, tag=f"{args.note}_test", start_idx=0, stop_idx=test_size)

    # train dataset
    build_dataset(args, videos=videos, tag=f"{args.note}_train", start_idx=test_size, stop_idx=len(videos))
