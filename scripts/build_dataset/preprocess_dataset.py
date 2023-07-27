"""
Create ytb_train.json and ytb_test.json
"""
from os import mkdir
from pathlib import Path
import csv
from operator import itemgetter
from itertools import groupby
import argtyped
from typing import List, Union, Dict, Iterable, Sequence
import json

def save_json(data, filename: Union[str, Path]):
    with open(filename, "w") as fid:
        json.dump(data, fid, indent=2)

def save_txt(data: Iterable, filename: Union[Path, str]):
    with open(filename, "w") as fid:
        for item in data:
            print(item, file=fid)

def flatten(seq):
    ret = []
    for sub in seq:
        ret += sub
    return ret


class Arguments(argtyped.Arguments):
    csv: Path = Path("data/YouTube-VLN/Extra/ytb_test.tsv")# frames_path
    location: Path = Path("data/YouTube-VLN/ytb")
    name : str = "ytb_test"
    min_caption: int = 2
    min_length: int = 4
    captionless: bool = True    # wheter to use captionless
    test_rate : float = 0.05


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))
    
    args.location.mkdir(parents=True,exist_ok=True)

    with open(args.csv) as fid:
        reader = csv.DictReader(
            fid, delimiter="\t", fieldnames=["video_id", "frame_id", "caption", "action"]
        )
        frames = [
            {
                "video_id": r["video_id"],
                "frame_id": int(r["frame_id"]),
                "caption": r["caption"],
                "action": r["action"]
            }
            for r in reader
        ]
    print("Loaded frames", len(frames))

    if not args.captionless:
        # filter out captionless images
        frames = [f for f in frames if f["caption"] != ""]
    print("After filtering out captionless frames", len(frames))

    # group by videos
    captions = sorted(frames, key=itemgetter("video_id"))
    captions_by_video = {
        listing: list(items)
        for listing, items in groupby(captions, key=itemgetter("video_id"))
    }
    print("Listings of video", len(captions_by_video))

    # filter out listings not having enough captions
    captions_by_video = {
        listing: items
        for listing, items in captions_by_video.items()
        if sum(f["caption"] != "" for f in items) >= args.min_caption
        and len(items) >= args.min_length
    }
    print("Listings with enough captions", len(captions_by_video))

    print("Number of frames", sum(len(items) for items in captions_by_video.values()))

    # export
    save_json(flatten(captions_by_video.values()), f"{args.location}/{args.name}.json")
    save_txt(captions_by_video.keys(), f"{args.location}/{args.name}-videos.txt")
