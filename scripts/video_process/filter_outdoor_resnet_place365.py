"""
Filter out the outdoor frames using wideresnet pretrained on places365
Inspired by https://github.com/airbert-vln/bnb-dataset/blob/b72941ede4166a3c7637370cbcb1602c70e2b56b/detect_room.py
"""
from pathlib import Path
import os
from typing import List, Tuple, Optional, Union, Dict
import hashlib
import urllib.request
import sys
import csv
from tqdm import tqdm
import torch
from torchvision import transforms as trn
from torch import nn
from torch.utils.data._utils.collate import default_collate
from torch.nn import functional as F
import numpy as np
import cv2
from PIL import Image
import argtyped
from torch.utils.data import Dataset, DataLoader
import scripts.video_process.wideresnet as wideresnet
from os import listdir
import multiprocessing.dummy as mp 

# os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"

csv.field_size_limit(sys.maxsize)
Image.MAX_IMAGE_PIXELS = None


class Arguments(argtyped.Arguments, underscore=True):
    output: Path = Path("data/YouTube-VLN/indoor_frames_resnet_place365")
    images: Path = Path("data/YouTube-VLN/raw_frames")
    batch_size: int = 100
    num_cat: int = 5
    num_attr: int = 10
    num_splits: int = 1
    start: int = 0
    num_workers: int = 5



def download_url(url, cache_dir):
    stem = hashlib.sha1(str(url).encode())
    filename = cache_dir / stem.hexdigest()
    if not filename.is_file():
        urllib.request.urlretrieve(url, filename)
    return filename


def load_labels(
    cache_dir: Union[Path, str]
) -> Tuple[Tuple[str, ...], np.ndarray, List[str], np.ndarray]:
    """
    prepare all the labels
    """

    # indoor and outdoor relevant
    filename_io = download_url(
        "https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt",
        cache_dir,
    )
    with open(filename_io) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene category relevant
    filename_category = download_url(
        "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt",
        cache_dir,
    )
    _classes = list()
    with open(filename_category) as class_file:
        for line in class_file:
            _classes.append(line.strip().split(" ")[0][3:])
    classes = tuple(_classes)

    # scene attribute relevant
    filename_attribute = download_url(
        "https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt",
        cache_dir,
    )
    with open(filename_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]

    filename_W = download_url(
        "http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy",
        cache_dir,
    )

    W_attribute = np.load(filename_W)
    return classes, labels_IO, labels_attribute, W_attribute


def get_tf():
    # load the image transformer
    tf = trn.Compose(
        [
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf


class Hooker:
    def __init__(self, model: nn.Module, features_names=("layer4", "avgpool")):
        self.features: List[np.ndarray] = []

        # this is the last conv layer of the resnet
        for name in features_names:
            model._modules.get(name).register_forward_hook(self)  # type: ignore

    def __call__(self, module: nn.Module, input, output):
        self.features.append(output.data.cpu().numpy())

    def reset(self):
        self.features = []


# load the model
def load_model(cache_dir: Union[Path, str]) -> nn.Module:
    # this model has a last conv feature map as 14x14

    model_file = download_url(
        "http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar",
        cache_dir,
    )

    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded avgpool layers...
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)  # type: ignore

    model.eval()

    return model


def load_photos(images: Path, cache_dir: Union[Path, str]) -> List[Union[str, Path]]:
    photo_cache = Path(cache_dir) / "photos.txt"

    # if photo_cache.is_file():
    #     with open(photo_cache, "r") as fid:
    #         photos: List[Union[str, Path]] = [l.strip() for l in fid.readlines()]
    # else:
    print("Preloading every images")
    photos = list(images.rglob("*.jpg"))
    with open(photo_cache, "w") as fid:
        fid.writelines(f"{l}\n" for l in photos)

    return photos


class ImageDataset(Dataset):
    def __init__(self, photos: List[Union[Path, str]]):
        self.photos = photos
        self.tf = get_tf()  # image transformer

    def __len__(self):
        return len(self.photos)

    def __getitem__(
        self, index: int
    ) -> Optional[Tuple[str, torch.Tensor, torch.Tensor]]:
        path = Path(self.photos[index])

        try:
            image = Image.open(path)
            image = image.convert("RGB")
        except:
            return None
        tensor = self.tf(image)

        listing_id = path.parent.name
        photo_id = int(str(path.name).split(".")[0])
        return str(listing_id), torch.tensor(photo_id), tensor


def collate_fn(batch: Tuple):
    batch = tuple([b for b in batch if b is not None])

    if not batch:
        return None

    return default_collate(batch)


def is_indoor(idx, labels_io):
    # vote for the indoor or outdoor
    io_image = np.mean(labels_io[idx[:10]])
    ans = bool(io_image < 0.5)
    return io_image, ans


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # type: ignore


@torch.no_grad()
def run_model(
    batch: List[torch.Tensor],
    model,
    hook,
    classes: Tuple[str, ...],
    labels_IO: np.ndarray,
    labels_attribute: List[str],
    W_attribute: np.ndarray,
    num_cat: int,
    num_attr: int,
    weight_softmax: Optional[np.ndarray] = None,
) -> List[Dict]:
    listing_ids, photo_ids, input_img = batch

    # forward pass
    logit = model.forward(input_img.cuda())
    h_x = F.softmax(logit, 1)

    detections = []

    for i, p in enumerate(h_x):  # type: ignore
        listing_id = str(listing_ids[i])
        photo_id = int(photo_ids[i])

        probs, idx = p.sort(0, True)  # type: ignore
        probs = probs.detach().cpu().numpy()
        idx = idx.detach().cpu().numpy()

        # scene category
        category = [(probs[j], classes[idx[j]]) for j in range(0, num_cat)]

        # output the scene attributes
        ft = [np.squeeze(f[i]) for f in hook.features]
        responses_attribute = softmax(W_attribute.dot(ft[1]))
        idx_a = np.argsort(responses_attribute)
        attributes = [
            (responses_attribute[idx_a[j]], labels_attribute[idx_a[j]])
            for j in range(-1, -num_attr, -1)
        ]

        detections.append(
            {
                "listing_id": listing_id,
                "photo_id": photo_id,
                "category": category,
                "attributes": attributes,
                "is_indoor": is_indoor(idx, labels_IO),
            }
        )

    hook.reset()

    return detections


def detection(args: Arguments, proc_id: int, cache_dir: Union[Path, str]):
    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels(cache_dir)

    model = load_model(cache_dir)
    hook = Hooker(model)
    model = model.cuda()
    def detection_sub(vid,
                       model = model,
                       hook = hook,
                       classes = classes, 
                       labels_IO = labels_IO, 
                       labels_attribute = labels_attribute, 
                       W_attribute = W_attribute):
        photos = load_photos(Path(f'{args.images}/{vid}'), cache_dir)
        print("The dataset contains a total of", len(photos))
        photos = photos[proc_id :: args.num_splits]
        print(
            "The split", proc_id, "over", args.num_splits, "contains", len(photos), "photos"
        )

        dataset = ImageDataset(photos)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,  # type: ignore
        )


        
        print(f"Start split {proc_id} on {len(dataset)} photos")
        print(f"Processing {args.output}/{vid}")
        if not os.path.exists(args.output/ f"{vid}"):
            os.makedirs(args.output/ f"{vid}")
        else:
            print("Skipping...")
            return
        with open(args.output/ f"{vid}" / "indoor.tsv", "wt") as fid:
            for batch in tqdm(dataloader):
                if batch is None:
                    print("none")
                    continue
                detections = run_model(
                    batch,
                    model,
                    hook,
                    classes,
                    labels_IO,
                    labels_attribute,
                    W_attribute,
                    num_cat=args.num_cat,
                    num_attr=args.num_attr,
                )
            
                # save the id of indoor images
                writer = csv.DictWriter(
                    fid, 
                    delimiter="\t", 
                    fieldnames=["listing_id", "photo_id"]
                )
                for d in detections:
                    if d["is_indoor"][1]:
                        writer.writerow({"listing_id":d["listing_id"],"photo_id":d["photo_id"]})
        order_frame(args,vid)

    videos = listdir(args.images)
    for vid in videos:
        detection_sub(vid)


            
def order_frame(args: Arguments, vid):
    with open(args.output/ f"{vid}" / "indoor.tsv", "r") as fid:
        reader = csv.DictReader(
            fid, 
            delimiter="\t", 
            fieldnames=["listing_id", "photo_id"]
        )
        dic_total = {}
        for row in reader:
            if row["listing_id"] not in dic_total:
                dic_total[row["listing_id"]] = [int(row["photo_id"])]
            dic_total[row["listing_id"]].append(int(row["photo_id"]))
        for name in dic_total:
            dic_total[name].sort()
        
        with open(args.output/ f"{vid}" / "indoor_order.tsv", "w") as fid:
            for item in dic_total:
                print(item,'\t',dic_total[item], file=fid)
            

if __name__ == "__main__":
    args = Arguments()
    print(args.to_string())

    cache_dir = Path.home() / ".cache" / args.output.name
    cache_dir.mkdir(exist_ok=True, parents=True)
    args.output.mkdir(exist_ok=True, parents=True)

    local_rank = os.environ.get('LOCAL_RANK', 0)
    start = max(local_rank, 0) + args.start
    
    detection(args, start, cache_dir)

    
