import os
from pathlib import Path
import csv
import base64
from multiprocessing import Pool
import sys
from glob import glob
from typing import List, Optional, Iterator, Dict, Union, Tuple, Any, Iterable
import pickle
import lmdb
import pickle
from tqdm.auto import tqdm
import argtyped
import numpy as np
import json

csv.field_size_limit(sys.maxsize)


class Arguments(argtyped.Arguments):
    output: Path = Path("data/YouTube-VLN/youtube_img_features/img_features")
    keys: Path = Path("data/YouTube-VLN/Extra/all_photo_path.txt") # contains the frame_id, video_id pairs we want to put on the LMDB files
    tsv_folder: Path = Path("data/YouTube-VLN/youtube_img_features")
    map_size: int = int(5e11)
    num_splits: int = 11
    start: int = 0
    num_workers: int = 11
    buffer_size: int = 1000

    tsv_fieldnames: Tuple[str, ...] = (
        "video_id",
        "frame_id",
        "image_w",
        "image_h",
        "vfov",
        "features",
        "boxes",
        "cls_prob",
        "attr_prob",
        "featureViewIndex",
        "featureHeading",
        "featureElevation",
    )

def load_redundant_features(
    tsv_files: List, image_folder: Path, tsv_fieldnames, cache: Path = Path("redundant_features.json")
) -> List[str]:
    cache = image_folder.parent / cache
    if not cache.is_file():
        print("create redundant_features.json")
        finished = {}
        finished['key'] = {}
        for f in tqdm(tsv_files):
            finished[f] = {}
            with open(f, "r") as cid:
                reader = csv.DictReader(cid, fieldnames=tsv_fieldnames, delimiter="\t")
                for i in reader:
                    key = f'{i["video_id"]}/{i["frame_id"]}'
                    if key in finished['key']:
                        finished['key'][key].append(f)
                        continue
                    finished['key'][key] = [f]
                    finished[f][key] = f
            
        with open(cache, "w") as fid:
            json.dump(finished,fid)
    print("redundant_features.json exits")
    return 

class LMDBWriter:
    def __init__(self, path: Union[Path, str], map_size: int, buffer_size: int):
        self._env = lmdb.open(str(path), map_size=map_size)
        self._buffer: List[Tuple[bytes, bytes]] = []
        self._buffer_size = buffer_size
        with self._env.begin(write=False) as txn:
            value = txn.get("keys".encode())
            self._keys: List[bytes] = [] if value is None else pickle.loads(value)

    def write(self, key: str, value: bytes):
        if key in self._keys:
            return
        bkey = key.encode()
        self._keys.append(bkey)
        self._buffer.append((bkey, value))
        if len(self._buffer) == self._buffer_size:
            self.flush()

    def flush(self):
        with self._env.begin(write=True) as txn:
            txn.put("keys".encode(), pickle.dumps(self._keys))
            for bkey, value in self._buffer:
                txn.put(bkey, value)

        self._buffer = []


def load_key(filename: Path) -> List[int]:
    with open(filename, "r", errors="replace") as fid:
        keys = [p.strip() for p in fid.readlines()]
    return keys


def non_redundant_features_to_lmdb(args: Arguments, proc_id: int):
    lmdb_file = f"{args.output}_{proc_id}"
    writer = LMDBWriter(lmdb_file, args.map_size, args.buffer_size)

    tsv_file = args.tsv_folder / f"ResNet-101-faster-rcnn-genome-{proc_id}.tsv"

    list_key_ids = load_key(args.keys)
    if proc_id == 0:
        print("Found", len(list_key_ids), "photos")
    list_key_ids = list_key_ids[proc_id :: args.num_splits]
    if proc_id == 0:
        print("Worker:", len(list_key_ids), "photos")
    totalrows = len(list_key_ids)
    
    check = args.output.parent / "check"
    check.mkdir(exist_ok=True)

    # return

    with open(f"{check}/{proc_id}_error.tsv", "w") as error_file, open(f"{check}/{proc_id}_save.tsv", "w") as fid:
        error_writer = csv.DictWriter(
            error_file, 
            delimiter="\t", 
            fieldnames=["video_id", "frame_id"]
        )
        save_writer = csv.DictWriter(
            fid, 
            delimiter="\t", 
            fieldnames=["video_id", "frame_id"]
        )

        with open(tsv_file, "rt") as fid:

            reader = csv.DictReader(fid, delimiter="\t", fieldnames=args.tsv_fieldnames)
            
            for item in tqdm(
                reader,
                desc="#{0}: ".format(proc_id),
                position=proc_id,
                total=totalrows,
                leave=True):

                key = f'{item["video_id"]}/{item["frame_id"]}'
                boxes = np.frombuffer(base64.b64decode(item["boxes"]))
                if boxes.size == 0:
                    error_writer.writerow({"video_id": item["video_id"], "frame_id": item["frame_id"]})
                    continue
                save_writer.writerow({"video_id": item["video_id"], "frame_id": item["frame_id"]})
                writer.write(key, pickle.dumps(item))

        writer.flush()


def redundant_features_to_lmdb(args: Arguments, proc_id: int):
    lmdb_file = f"{args.output}_{proc_id}"
    writer = LMDBWriter(lmdb_file, args.map_size, args.buffer_size)

    tsv_file = args.tsv_folder / f"ResNet-101-faster-rcnn-genome-{proc_id}.tsv"
    finished = json.load(open("data/YouTube-VLN/Extra/redundant_features.json", 'r'))
    if proc_id == 0:
        print("Found", len(finished['key']), "photos")
    list_key_ids = finished[str(tsv_file)]
    if proc_id == 0:
        print("Worker:", len(list_key_ids), "photos")
    totalrows = len(list_key_ids)
    
    check = args.output.parent / "check"
    check.mkdir(exist_ok=True)

    # return

    with open(f"{check}/{proc_id}_error.tsv", "w") as error_file, open(f"{check}/{proc_id}_save.tsv", "w") as fid:
        error_writer = csv.DictWriter(
            error_file,
            delimiter="\t",
            fieldnames=["video_id", "frame_id"]
        )
        save_writer = csv.DictWriter(
            fid, 
            delimiter="\t", 
            fieldnames=["video_id", "frame_id"]
        )

        with open(tsv_file, "rt") as fid:

            reader = csv.DictReader(fid, delimiter="\t", fieldnames=args.tsv_fieldnames)
            
            for item in tqdm(
                reader,
                desc="#{0}: ".format(proc_id),
                position=proc_id,
                leave=True):

                key = f'{item["video_id"]}/{item["frame_id"]}'
                if key in list_key_ids:
                    boxes = np.frombuffer(base64.b64decode(item["boxes"]))
                    if boxes.size == 0:
                        error_writer.writerow({"video_id": item["video_id"], "frame_id": item["frame_id"]})
                        continue
                    save_writer.writerow({"video_id": item["video_id"], "frame_id": item["frame_id"]})
                    writer.write(key, pickle.dumps(item))

        writer.flush()


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    tsv_files = list(glob(str(args.tsv_folder / "ResNet-101-faster-rcnn-genome-*.tsv"), recursive=True))

    if len(tsv_files) != args.num_splits:
        raise ValueError(
                f"the number of tsv files {len(tsv_files)} is not equal to the number of splits {args.num_splits}!"
            )

    print("Found", len(tsv_files), "files")

    # Whether image features are repeatedly extracted from different TSV files
    if False:
        features_to_lmdb = redundant_features_to_lmdb
        load_redundant_features(tsv_files, args.keys, args.tsv_fieldnames)

    else:
        features_to_lmdb = non_redundant_features_to_lmdb

    if args.num_workers == 0:
        features_to_lmdb(args, args.start)

    else:
        p = Pool(args.num_workers)
        p.starmap(
            features_to_lmdb,
            [
                (args, proc_id)
                for proc_id in range(args.start, args.start + args.num_workers)
            ],
        )
    print("create total tsv")
    os.system(f'cat {args.output.parent / "check" / "*_error.tsv"} > {args.output.parent / "check" / "total-error.tsv"}')
    os.system(f'cat {args.output.parent / "check" / "*_save.tsv"} > {args.output.parent / "check" / "total-save.tsv"}')
