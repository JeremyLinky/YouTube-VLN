"""
Build a testset for Youtube Video
"""
import sys  
sys.path.append('/mnt/cephfs/home/lance/Projects/lily') 
from pathlib import Path
from typing import List, Union, Dict, Callable, TypeVar, Iterator
import itertools
from itertools import groupby
from operator import itemgetter
import argtyped
from tqdm.auto import tqdm
import json
import random
import copy
T = TypeVar("T")

class Arguments(argtyped.Arguments):
    output: Path = Path("data/YouTube-VLN/ytb/merge+testset.json")
    captions: Path = Path("data/YouTube-VLN/ytb/merge+ytb_test.json")
    num_negatives: int = 2
    min_length: int = 4
    max_length: int = 7
    min_captioned: int = 2
    max_captioned: int = 7
    out_listing: bool = False
    traj_judge: bool = False
    seed: int = -1
    negative_style: str = 'normal'
    epochs: int = 1

args = Arguments()
if args.seed != -1:
    random.seed(args.seed)

from utils.dataset.features_reader import PhotoId
from utils.dataset.common import (
    ytb_generate_trajectory_from_listing,
    generate_negative_trajectories,
    ytb_get_key,
)

def load_json_data(path):
    with open(path, "r") as fid:
        data = json.load(fid)
    return data

def save_json(data, filename: Union[str, Path]):
    with open(filename, "w") as fid:
        json.dump(data, fid, indent=2)

def shuffle_two(seq: List[T]) -> Iterator[List[T]]:
    n = len(seq)
    ij = list(itertools.permutations(range(n), 2))
    random.shuffle(ij)
    for i, j in ij:
        seq2 = copy.deepcopy(seq)
        seq2[i], seq2[j] = seq2[j], seq2[i]
        yield seq2

def pick_frame_ids(
    video_id: int,
    video_ids: List[int],
    frame_ids_by_video: Dict[str, List[PhotoId]],
    key_id_to_caption: Dict[int, Dict],
    trajectories: List[List[int]],
    min_length: int,
    max_length: int,
    min_captioned: int,
    max_captioned: int,
    num_negatives: int,
    shuffler: Callable,
    out_listing: bool,
    traj_judge: bool,
    negative_style: str,
):

    positive_trajectory, captioned  = ytb_generate_trajectory_from_listing(
        video_id,
        video_ids,
        frame_ids_by_video,
        key_id_to_caption,
        trajectories[video_id],
        min_length,
        max_length,
        min_captioned,
        max_captioned,
    )

    neg_captions, neg_images, neg_randoms, order_labels= generate_negative_trajectories(
        positive_trajectory,
        captioned,
        video_ids,
        frame_ids_by_video,
        key_id_to_caption,
        num_negatives,
        shuffler,
        'ytb',
        traj_judge,
        negative_style,
    )
    
    return positive_trajectory, neg_captions, neg_images, neg_randoms, order_labels



def filter(items, cond_func=lambda x: True):
    """Yield items from any nested iterable"""
    for x in items:
        if cond_func(x):
            yield x




def build_testset(args: Arguments):
    print("Loading frame by video")
    captions = load_json_data(args.captions)
    key_id_to_caption = {
        ytb_get_key(caption["video_id"], caption["frame_id"]): caption for caption in captions
    }
    captions = sorted(captions, key=itemgetter("video_id"))
    frame_ids_by_video = {
        video_id: list(frames)
        for video_id, frames in groupby(captions, key=itemgetter("video_id"))
    }
    video_ids = list(frame_ids_by_video.keys())
    print(f"Loaded {len(video_ids)} listings")

    trajectories = {}
    for vid in video_ids:
        trajectory = []
        for frame in frame_ids_by_video[vid]:
            if frame['caption'] != "":
                trajectory.append(frame['frame_id'])
        trajectories[vid] = trajectory
    
    testset = {}
    for video_id, frames in tqdm(frame_ids_by_video.items()):
        if len(frames) >= args.min_length or args.out_listing:
            if args.epochs == 1:
                temp = pick_frame_ids(
                    video_id,
                    video_ids,
                    frame_ids_by_video,
                    key_id_to_caption,
                    trajectories,
                    args.min_length,
                    args.max_length,
                    args.min_captioned,
                    args.max_captioned,
                    args.num_negatives,
                    shuffle_two,
                    args.out_listing,
                    args.traj_judge,
                    args.negative_style,
                )
            else:
                temp = [
                    pick_frame_ids(
                        video_id,
                        video_ids,
                        frame_ids_by_video,
                        key_id_to_caption,
                        trajectories,
                        args.min_length,
                        args.max_length,
                        args.min_captioned,
                        args.max_captioned,
                        args.num_negatives,
                        shuffle_two,
                        args.out_listing,
                        args.traj_judge,
                        args.negative_style,
                    )
                    for _ in range(args.epochs) 
                ]

            testset[str(video_id)] = temp
    
    print(f"Testset size is {len(testset)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(testset, args.output)


if __name__ == "__main__":
    print(args.to_string(width=80))

    build_testset(args)
