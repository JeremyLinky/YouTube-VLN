#!/usr/bin/env python3
"""
Script to precompute image features using bottom-up attention
(i.e., Faster R-CNN pretrained on Visual Genome) 
"""
import sys  
import shutil
sys.path.append('your_path/bottom-up-attention') 
sys.path.append('your_path/bottom-up-attention/caffe/python') 
sys.path.insert(0,'your_path/bottom-up-attention/lib') # Raise priorities and avoid conflicts

from pathlib import Path
import base64
import sys
import csv
from multiprocessing import Pool
import math
import random
from typing import List, Dict, Tuple, Any, Optional
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import pairwise_distances
from PIL import Image, UnidentifiedImageError
import argtyped
import cv2
import caffe  # type: ignore
from fast_rcnn.config import cfg_from_file  # type: ignore
from fast_rcnn.test import im_detect, _get_blobs  # type: ignore
from fast_rcnn.nms_wrapper import nms  # type: ignore
import matplotlib as mpl
import time
mpl.use("Agg")
import matplotlib.pyplot as plt

import os
from glob import glob

Image.MAX_IMAGE_PIXELS = None
random.seed(1)
csv.field_size_limit(sys.maxsize)


TSV_FIELDNAMES = [
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
]

# Camera sweep parameters
NUM_SWEEPS = 3
VIEWS_PER_SWEEP = 12
VIEWPOINT_SIZE = NUM_SWEEPS * VIEWS_PER_SWEEP  # Number of total views from one pano
HEADING_INC = 360 / VIEWS_PER_SWEEP  # in degrees
ANGLE_MARGIN = (
    5  # margin of error for deciding if an object is closer to the centre of another view
)
ELEVATION_START = -30  # Elevation on first sweep
ELEVATION_INC = 30  # How much elevation increases each sweep

# Filesystem etc
FEATURE_SIZE = 2048

# Simulator image parameters
WIDTH = 600  # Max size handled by Faster R-CNN model
HEIGHT = 600
VFOV = 80
ASPECT = WIDTH / HEIGHT
HFOV = math.degrees(2 * math.atan(math.tan(math.radians(VFOV / 2)) * ASPECT))
FOC = (HEIGHT / 2) / math.tan(math.radians(VFOV / 2))  # focal length

# Settings for the number of features per image
MIN_LOCAL_BOXES = 5
MAX_LOCAL_BOXES = 20
MAX_TOTAL_BOXES = 100
NMS_THRESH = 0.2  # bottom-up 0.3
CONF_THRESH = 0.1  # bottom-up 0.2

MAX_SIZE: int = 1333
MIN_SIZE: int = 800


class Arguments(argtyped.Arguments):
    caffe_root: Path = Path("your_path/bottom-up-attention")
    proto: Path = Path("models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt")
    model: Path = Path("data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel")
    cfg_file: Path = Path("experiments/cfgs/faster_rcnn_end2end_resnet.yml")
    updown_data: Path = Path("data/genome/1600-400-20")
    dry_run: bool = False
    outfile: Path = Path("data/YouTube-VLN/youtube_img_features")
    num_splits: int = 11
    start: int = 0
    num_workers: int = 3
    images: Path = Path("data/YouTube-VLN/raw_frames")
    gpu: str = "-1"

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
            
def _build_caffe_model(args: Arguments, proc_id: int):

    # Set up Caffe Faster R-CNN
    cfg_from_file(str(args.caffe_root / args.cfg_file))
    caffe.set_mode_gpu()
    gpu_id = (proc_id - args.start) % args.num_workers
    # print(gpu_ids, gpu_id)
    caffe.set_device(gpu_id)
    net = caffe.Net(
        str(args.caffe_root / args.proto),
        caffe.TEST,
        weights=str(args.caffe_root / args.model),
    )
    return net


def visual_overlay(im, dets, ix, classes, attributes):
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    valid = np.where(dets["featureViewIndex"] == ix)[0]
    objects = np.argmax(dets["cls_prob"][valid, 1:], axis=1)
    obj_conf = np.max(dets["cls_prob"][valid, 1:], axis=1)
    attr_thresh = 0.1
    attr = np.argmax(dets["attr_prob"][valid, 1:], axis=1)
    attr_conf = np.max(dets["attr_prob"][valid, 1:], axis=1)
    boxes = dets["boxes"][valid]

    for i in range(len(valid)):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        cls = classes[objects[i] + 1]
        if attr_conf[i] > attr_thresh:
            cls = attributes[attr[i] + 1] + " " + cls
        cls += " %.2f" % obj_conf[i]
        plt.gca().add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="red",
                linewidth=2,
                alpha=0.5,
            )
        )
        plt.gca().text(
            bbox[0],
            bbox[1] - 2,
            "%s" % (cls),
            bbox=dict(facecolor="blue", alpha=0.5),
            fontsize=10,
            color="white",
        )
    return fig


def load_classes(args):
    # Load updown object classes
    classes = ["__background__"]
    with open(args.caffe_root / args.updown_data / "objects_vocab.txt") as f:
        for object in f.readlines():
            classes.append(object.split(",")[0].lower().strip())

    # Load updown attributes
    attributes = ["__no_attribute__"]
    with open(args.caffe_root / args.updown_data / "attributes_vocab.txt") as f:
        for att in f.readlines():
            attributes.append(att.split(",")[0].lower().strip())
    return classes, attributes


def load_all_photo_paths(
    image_folder: Path, cache: Path = Path("all_photo_path.txt")
) -> List[str]:
    cache = image_folder.parent / "Extra" / cache
    if not cache.is_file():
        print("create all_photo_path.txt")
        # locations = list(image_folder.iterdir())
        # with open(cache, "w") as fid:
        #     for location in tqdm(locations):
        #         for image in location.glob("*.jpg"):
        #             fid.write(f"{image}\n")
        with open(cache, "w") as fid:
            for image in tqdm(image_folder.rglob("*.jpg")):
                fid.write(f"{image}\n")
    print("read all_photo_path.txt")
    with open(cache, "r", errors="replace") as fid:
        photos = [p.strip() for p in fid.readlines()]
    return photos

def load_finished_photo_paths(
    tsv_files: List, image_folder: Path, cache: Path = Path("finished_photo_path.txt")
) -> List[str]:
    cache = image_folder.parent / "Extra" / cache
    if not cache.is_file():
        print("create finished_photo_path.txt")
        with open(cache, "w") as fid:
            for f in tqdm(tsv_files):
                with open(f, "r") as cid:
                    reader = csv.DictReader(cid, fieldnames=TSV_FIELDNAMES, delimiter="\t")
                    for i in reader:
                        fid.write(f'{image_folder}/{i["video_id"]}/{i["frame_id"]}.jpg\n')
    print("read finished_photo_path.txt")
    with open(cache, "r", errors="replace") as fid:
        finished = [p.strip() for p in fid.readlines()]
    return finished

def load_photo_paths(
    tsv_files: List, image_folder: Path, cache: Path = Path("doing_photo_path.txt")
) -> List[str]:
    cache = image_folder.parent / "Extra" / cache
    if not cache.is_file():
        print("create doing_photo_path.txt")
        
        all_photos = set(load_all_photo_paths(image_folder))
        finished = set(load_finished_photo_paths(tsv_files, image_folder))
        doing = list(all_photos - finished)

        with open(cache, "w") as fid:
            for image in tqdm(doing):
                fid.write(f"{image}\n")

    print("read doing_photo_path.txt")
    with open(cache, "r", errors="replace") as fid:
        doing = [p.strip() for p in fid.readlines()]
    return doing

def transform_img(image: Image) -> np.ndarray:
    """Prep opencv BGR 3 channel image for the network"""
    im = np.array(image)[:, :, ::-1]
    # Scale based on minimum size
    im_size_min = np.min(im.shape[0:2])
    im_size_max = np.max(im.shape[0:2])
    im_scale = MIN_SIZE / im_size_min

    # Prevent the biggest axis from being more than max_size
    # If bigger, scale it down
    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = MAX_SIZE / im_size_max

    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
    )
    blob = np.array(im, copy=True)
    return blob


class Dataloader:
    def __init__(self, photos: List[str]):
        self.photos = photos

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, index: int) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
        path = Path(self.photos[index])
        video_id = path.parent.stem
        frame_id = path.stem
        try:
            image = Image.open(path)
            image = image.convert("RGB")
        except (OSError, UnidentifiedImageError, UnicodeEncodeError) as e:
            # FIXME deal with utf8 path
            try:
                problem = Path('data/problem')
                problem.mkdir(exist_ok=True, parents=True)
                shutil.copy(str(path),str(problem))
            except: 
                return None
            return None

        im = transform_img(image)
        record = {
            "video_id": video_id,
            "frame_id": frame_id,
            "image_h": im.shape[0],
            "image_w": im.shape[1],
        }
        return record, im


def get_detections_from_im(
    record: Dict, net, im: np.ndarray, conf_thresh: float = CONF_THRESH
):

    if "features" not in record:
        ix = 0  # First view in the pano
    elif record["featureViewIndex"].shape[0] == 0:
        ix = 0  # No detections in pano so far
    else:
        ix = int(record["featureViewIndex"][-1]) + 1

    # Code from bottom-up and top-down
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs["rois"].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs["cls_prob"].data
    attr_prob = net.blobs["attr_prob"].data
    pool5 = net.blobs["pool5_flat"].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)  # type: ignore
        keep = np.array(nms(dets, NMS_THRESH))
        max_conf[keep] = np.where(
            cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
        )

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_LOCAL_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_LOCAL_BOXES]
    elif len(keep_boxes) > MAX_LOCAL_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_LOCAL_BOXES]

    # Discard any box that would be better centered in another image
    # threshold for pixel distance from center of image
    hor_thresh = FOC * math.tan(math.radians(HEADING_INC / 2 + ANGLE_MARGIN))
    vert_thresh = FOC * math.tan(math.radians(ELEVATION_INC / 2 + ANGLE_MARGIN))
    center_x = 0.5 * (cls_boxes[:, 0] + cls_boxes[:, 2])
    center_y = 0.5 * (cls_boxes[:, 1] + cls_boxes[:, 3])
    reject = (center_x < WIDTH / 2 - hor_thresh) | (center_x > WIDTH / 2 + hor_thresh)
    if ix >= VIEWS_PER_SWEEP:  # Not lowest sweep
        reject |= center_y > HEIGHT / 2 + vert_thresh
    if ix < VIEWPOINT_SIZE - VIEWS_PER_SWEEP:  # Not highest sweep
        reject |= center_y < HEIGHT / 2 - vert_thresh
    keep_boxes = np.setdiff1d(keep_boxes, np.argwhere(reject))

    # Calculate the heading and elevation of the center of each observation
    featureHeading = np.arctan2(center_x[keep_boxes] - WIDTH / 2, FOC)
    # normalize featureHeading
    featureHeading = np.mod(featureHeading, math.pi * 2)
    # force it to be the positive remainder, so that 0 <= angle < 360
    featureHeading = np.expand_dims(
        np.mod(featureHeading + math.pi * 2, math.pi * 2), axis=1  # type: ignore
    )
    # force into the minimum absolute value residue class, so that -180 < angle <= 180
    featureHeading = np.where(
        featureHeading > math.pi, featureHeading - math.pi * 2, featureHeading
    )
    featureElevation = np.expand_dims(
        np.arctan2(-center_y[keep_boxes] + HEIGHT / 2, FOC), axis=1
    )

    # Save features, etc
    if "features" not in record:
        record["boxes"] = cls_boxes[keep_boxes]
        record["cls_prob"] = cls_prob[keep_boxes]
        record["attr_prob"] = attr_prob[keep_boxes]
        record["features"] = pool5[keep_boxes]
        record["featureViewIndex"] = np.ones((len(keep_boxes), 1), dtype=np.float32) * ix
        record["featureHeading"] = featureHeading
        record["featureElevation"] = featureElevation
    else:
        record["boxes"] = np.vstack([record["boxes"], cls_boxes[keep_boxes]])
        record["cls_prob"] = np.vstack([record["cls_prob"], cls_prob[keep_boxes]])
        record["attr_prob"] = np.vstack([record["attr_prob"], attr_prob[keep_boxes]])
        record["features"] = np.vstack([record["features"], pool5[keep_boxes]])
        record["featureViewIndex"] = np.vstack(
            [
                record["featureViewIndex"],
                np.ones((len(keep_boxes), 1), dtype=np.float32) * ix,
            ]
        )
        record["featureHeading"] = np.vstack([record["featureHeading"], featureHeading])
        record["featureElevation"] = np.vstack(
            [record["featureElevation"], featureElevation]
        )
    return


def filter(record: Dict, max_boxes: int):
    if record["features"].shape[0] == 0:
        return

    # Remove the most redundant features (that have similar heading, elevation and
    # are close together to an existing feature in cosine distance)
    feat_dist = pairwise_distances(record["features"], metric="cosine")
    # Heading and elevation diff
    heading_diff = pairwise_distances(record["featureHeading"], metric="euclidean")
    heading_diff = np.minimum(heading_diff, 2 * math.pi - heading_diff)  # type: ignore
    elevation_diff = pairwise_distances(record["featureElevation"], metric="euclidean")
    feat_dist = feat_dist + heading_diff + elevation_diff  # type: ignore
    # Discard diagonal and upper triangle by setting large distance
    feat_dist += 10 * np.identity(feat_dist.shape[0], dtype=np.float32)
    feat_dist[np.triu_indices(feat_dist.shape[0])] = 10.0
    ind = np.unravel_index(np.argsort(feat_dist, axis=None), feat_dist.shape)
    # Remove indices of the most similar features (in appearance and orientation)
    keep = set(range(feat_dist.shape[0]))
    ix = 0
    while len(keep) > max_boxes:
        i = ind[0][ix]
        j = ind[1][ix]
        if i not in keep or j not in keep:
            ix += 1
            continue
        if record["cls_prob"][i, 1:].max() > record["cls_prob"][j, 1:].max():
            keep.remove(j)
        else:
            keep.remove(i)
        ix += 1
    # Discard redundant features
    for k, v in record.items():
        if k in [
            "boxes",
            "cls_prob",
            "attr_prob",
            "features",
            "featureViewIndex",
            "featureHeading",
            "featureElevation",
        ]:
            record[k] = v[sorted(keep)]


def collate_fn(x):
    return list(zip(*x))


def build_tsv(args: Arguments, proc_id: int, info: dict):
    model = _build_caffe_model(args, proc_id)
    
    photos = info['photos'][proc_id :: args.num_splits]
    print(proc_id, args.num_splits, len(photos))

    dataloader = Dataloader(photos)

    count = 0
    t_net = Timer()

    # append
    with open(args.outfile / f"ResNet-101-faster-rcnn-genome-{proc_id}.tsv", "a") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for batch in tqdm(
            dataloader,
            desc="#{0}: ".format(proc_id),
            position=proc_id,
            total=len(dataloader),
            leave=True):
            if batch is None:
                continue
            record, image = batch

            # import pdb
            # pdb.set_trace()
            get_detections_from_im(record, model, image)
            if args.dry_run:
                print(
                    "%d: Detected %d objects in pano"
                    % (proc_id, record["features"].shape[0])
                )

            filter(record, MAX_TOTAL_BOXES)

            if args.dry_run:
                print(
                    "%d: Reduced to %d objects in pano"
                    % (proc_id, record["features"].shape[0])
                )
                fig = visual_overlay(image, record, 0, info['classes'], info['attributes'])
                fig.savefig(
                    "img_features/examples/%s/%s.png"
                    % (record["video_id"], record["frame_id"])
                )
                plt.close()

            for k, v in record.items():
                if isinstance(v, np.ndarray):
                    record[k] = str(base64.b64encode(v), "utf-8")

            writer.writerow(record)
            count += 1
            t_net.toc()
            if count % 100 == 0 and args.dry_run:
                print(
                    "%d: Processed %d / %d viewpoints,  %.1fs avg net time, projected %.1f hours"
                    % (
                        proc_id,
                        count,
                        len(dataloader),
                        t_net.average_time,
                        t_net.average_time * len(dataloader) / 3600,
                    )
                )
                return


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))


    if os.path.exists(args.outfile):
        print("output folder exists!")
        tsv_files = list(glob(str(args.outfile / "ResNet-101-faster-rcnn-genome-*.tsv"), recursive=True))
        if len(tsv_files) !=0 and len(tsv_files) != args.num_splits and len(tsv_files) != args.start:
            raise ValueError(
                f"the number of tsv files {len(tsv_files)} is not equal to the number of splits {args.num_splits}!"
            )
    else:
        print("create output folder!")
        os.makedirs(args.outfile)
        tsv_files = []
    
    Path(args.images.parent / "Extra").mkdir(exist_ok=True, parents=True)

    print("Start to work !")
    
    # some public information
    info = {}
    info['classes'], info['attributes'] = load_classes(args)
    info['photos'] = load_photo_paths(tsv_files, args.images)
    
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if args.num_workers == 0 or args.num_workers == 1:
        build_tsv(args, args.start, info)

    elif args.num_workers > 0:
        p = Pool(args.num_workers)
        p.starmap(
            build_tsv,
            [
                (args, proc_id, info)
                for proc_id in range(args.start, args.start + args.num_workers)
            ],
        )
