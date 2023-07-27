import base64
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Set, Sequence
from pathlib import Path
import pickle
import lmdb
import numpy as np

PhotoId = Union[int, Tuple[int, ...]]
Sample = Tuple[str, PhotoId]  # listing id, photo id
Trajectory = List[Sample]

PhotoIdType = Dict[int, Tuple[Trajectory, List[Trajectory], List[Trajectory], List[Trajectory]]]


class FeaturesReader:
    def __init__(self, path: Union[Path, str, Sequence[Union[Path, str]]]):
        if isinstance(path, (Path, str)):
            path = [path]

        # open database
        self.envs = [
            lmdb.open(
                str(p),
                readonly=True,
                readahead=False,
                max_readers=20,
                lock=False,
                map_size=int(1e9),
            )
            for p in path
        ]

        # get keys
        self.keys = {}
        for i, env in enumerate(self.envs):
            with env.begin(write=False, buffers=True) as txn:
                bkeys = txn.get("keys".encode())
                if bkeys is None:
                    raise RuntimeError("Please preload keys in the LMDB")
                for k in pickle.loads(bkeys):
                    self.keys[k.decode()] = i

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, keys: Tuple) -> List:
        for key in keys:
            if not isinstance(key, str) or key not in self.keys:
                raise TypeError(f"invalid key: {key}")

        env_idx = [self.keys[key] for key in keys]
        items = [None] * len(keys)

        # we minimize the number of connections to an LMDB
        for idx in set(env_idx):
            with self.envs[idx].begin(write=False) as txn:
                for i, (idx_i, key) in enumerate(zip(env_idx, keys)):
                    if idx_i != idx:
                        continue
                    item = txn.get(key.encode())
                    if item is None:
                        continue
                    items[i] = pickle.loads(item)

        return items

@dataclass
class Record:
    photo_id: int
    listing_id: str
    num_boxes: int
    image_width: int
    image_height: int
    cls_prob: np.ndarray
    features: np.ndarray
    boxes: np.ndarray




class BaseFeaturesReader(FeaturesReader):
    def _split_key(self, key: str) -> Tuple[str, str]:
        raise NotImplementedError("_split_key: not implemented!")

    def _get_boxes(self, record: Record) -> np.ndarray:
        image_width = record.image_width
        image_height = record.image_height

        boxes = record.boxes
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area /= image_width * image_height

        N = len(boxes)
        output = np.zeros(shape=(N, 5), dtype=np.float32)

        # region encoding
        output[:, 0] = boxes[:, 0] / image_width
        output[:, 1] = boxes[:, 1] / image_height
        output[:, 2] = boxes[:, 2] / image_width
        output[:, 3] = boxes[:, 3] / image_height
        output[:, 4] = area

        return output


    def _get_locations(self, boxes: np.ndarray):
        """ Convert boxes and orientation information into locations. """
        N = len(boxes)
        locations = np.ones(shape=(N, 11), dtype=np.float32)

        # region encoding
        locations[:, 0] = boxes[:, 0]
        locations[:, 1] = boxes[:, 1]
        locations[:, 2] = boxes[:, 2]
        locations[:, 3] = boxes[:, 3]
        locations[:, 4] = boxes[:, 4]

        # other indices are used for Room-to-Room

        return locations


    def _convert_item(self, key: str, item: Dict) -> Record:
        # FIXME use one convention and not two!!
        photo_id, listing_id = self._split_key(key)
        old = "image_width" in item

        image_w = int(item["image_width" if old else "image_w"])  # pixels
        image_h = int(item["image_height" if old else "image_h"])  # pixels
        features = np.frombuffer(
            item["feature"] if old else base64.b64decode(item["features"]),
            dtype=np.float32,
        )
        features = features.reshape((-1, 2048))  # K x 2048 region features
        boxes = np.frombuffer(
            item["bbox"] if old else base64.b64decode(item["boxes"]), dtype=np.float32,
        )
        boxes = boxes.reshape((-1, 4))  # K x 4 region coordinates (x1, y1, x2, y2)
        num_boxes = int(boxes.shape[0])
        cls_prob = np.frombuffer(
            item["cls_prob"] if old else base64.b64decode(item["cls_prob"]),
            dtype=np.float32,
        )
        cls_prob = cls_prob.reshape(
            (-1, 1601)
        )  # K x 1601 region object class probabilities
        return Record(
            photo_id, listing_id, num_boxes, image_w, image_h, cls_prob, features, boxes,
        )


    def __getitem__(self, query: Tuple):
        l_boxes, l_probs, l_features = [], [], []
        items = super().__getitem__(query)
        for key, item in zip(list(query), items):
            record = self._convert_item(key, item)
            l_boxes.append(self._get_boxes(record))
            l_probs.append(record.cls_prob)
            l_features.append(record.features)

        features: np.ndarray = np.concatenate(l_features, axis=0)
        boxes = np.concatenate(l_boxes, axis=0)
        probs = np.concatenate(l_probs, axis=0)
        locations = self._get_locations(boxes)

        if features.size == 0:
            raise RuntimeError("Features could not be correctly read")

        # add a global feature vector
        g_feature = features.mean(axis=0, keepdims=True)
        g_location = np.array([[0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,]])
        g_prob = np.ones(shape=(1, 1601)) / 1601  # uniform probability

        features = np.concatenate([g_feature, features], axis=0)
        locations = np.concatenate([g_location, locations], axis=0)
        probs = np.concatenate([g_prob, probs], axis=0)

        return features, locations, probs


class BnBFeaturesReader(BaseFeaturesReader):
    def _split_key(self, key: str) -> Tuple[str, str]:
        return map(int, key.split("-"))


class YTbFeaturesReader(BaseFeaturesReader):
    def _split_key(self, key: str) -> Tuple[str, str]:
        return key.split("/")



def _convert_item(item):
    # item['scanId'] is unchanged
    # item['viewpointId'] is unchanged
    item["image_w"] = int(item["image_w"])  # pixels
    item["image_h"] = int(item["image_h"])  # pixels
    item["vfov"] = int(item["vfov"])  # degrees
    item["features"] = np.frombuffer(
        base64.b64decode(item["features"]), dtype=np.float32
    ).reshape(
        (-1, 2048)
    )  # K x 2048 region features
    item["boxes"] = np.frombuffer(
        base64.b64decode(item["boxes"]), dtype=np.float32
    ).reshape(
        (-1, 4)
    )  # K x 4 region coordinates (x1, y1, x2, y2)
    item["cls_prob"] = np.frombuffer(
        base64.b64decode(item["cls_prob"]), dtype=np.float32
    ).reshape(
        (-1, 1601)
    )  # K x 1601 region object class probabilities
    # item["attr_prob"] = np.frombuffer(
    #     base64.b64decode(item["attr_prob"]), dtype=np.float32
    # ).reshape(
    #     (-1, 401)
    # )  # K x 401 region attribute class probabilities
    item["viewHeading"] = np.frombuffer(
        base64.b64decode(item["viewHeading"]), dtype=np.float32
    )  # 36 values (heading of each image)
    item["viewElevation"] = np.frombuffer(
        base64.b64decode(item["viewElevation"]), dtype=np.float32
    )  # 36 values (elevation of each image)
    item["featureHeading"] = np.frombuffer(
        base64.b64decode(item["featureHeading"]), dtype=np.float32
    )  # K headings for the features
    item["featureElevation"] = np.frombuffer(
        base64.b64decode(item["featureElevation"]), dtype=np.float32
    )  # K elevations for the features
    item["featureViewIndex"] = np.frombuffer(
        base64.b64decode(item["featureViewIndex"]), dtype=np.float32
    )  # K indices mapping each feature to one of the 36 images


def _get_boxes(item):
    image_width = item["image_w"]
    image_height = item["image_h"]

    boxes = item["boxes"]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area /= image_width * image_height

    N = len(boxes)
    output = np.zeros(shape=(N, 5), dtype=np.float32)

    # region encoding
    output[:, 0] = boxes[:, 0] / image_width
    output[:, 1] = boxes[:, 1] / image_height
    output[:, 2] = boxes[:, 2] / image_width
    output[:, 3] = boxes[:, 3] / image_height
    output[:, 4] = area

    return output


def _get_locations(boxes, feat_headings, feat_elevations, heading, next_heading):
    """ Convert boxes and orientation information into locations. """
    N = len(boxes)
    locations = np.ones(shape=(N, 11), dtype=np.float32)

    # region encoding
    locations[:, 0] = boxes[:, 0]
    locations[:, 1] = boxes[:, 1]
    locations[:, 2] = boxes[:, 2]
    locations[:, 3] = boxes[:, 3]
    locations[:, 4] = boxes[:, 4]

    # orientation encoding
    locations[:, 5] = np.sin(feat_headings - heading)
    locations[:, 6] = np.cos(feat_headings - heading)
    locations[:, 7] = np.sin(feat_elevations)
    locations[:, 8] = np.cos(feat_elevations)

    # next orientation encoding
    locations[:, 9] = np.sin(feat_headings - next_heading)
    locations[:, 10] = np.cos(feat_headings - next_heading)

    return locations


class PanoFeaturesReader(FeaturesReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get viewpoints
        self.viewpoints: Dict[str, Set[str]] = {}
        for key in self.keys:
            scan_id, viewpoint_id = key.split("-")
            if scan_id not in self.viewpoints:
                self.viewpoints[scan_id] = set()
            self.viewpoints[scan_id].add(viewpoint_id)

    def __getitem__(self, query: Tuple):
        key, heading, next_heading = query  # unpack key
        if key not in self.keys:
            raise TypeError(f"invalid key: {key}")

        env = self.envs[self.keys[key]]
        # load from disk
        with env.begin(write=False) as txn:
            item = pickle.loads(txn.get(key.encode()))  # type: ignore
            _convert_item(item)

            boxes = _get_boxes(item)
            probs = item["cls_prob"]
            features = item["features"]
            headings = item["featureHeading"]
            elevations = item["featureElevation"]

        if not isinstance(features, np.ndarray):
            raise RuntimeError(f"Unexpected type for features ({type(features)})")

        locations = _get_locations(boxes, headings, elevations, heading, next_heading)

        # add a global feature vector
        g_feature = features.mean(axis=0, keepdims=True)
        g_location = np.array(
            [
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    np.sin(0 - heading),
                    np.cos(0 - heading),
                    np.sin(0),
                    np.cos(0),
                    np.sin(0 - next_heading),
                    np.cos(0 - next_heading),
                ]
            ]
        )
        g_prob = np.ones(shape=(1, 1601)) / 1601  # uniform probability

        features = np.concatenate([g_feature, features], axis=0)
        locations = np.concatenate([g_location, locations], axis=0)
        probs = np.concatenate([g_prob, probs], axis=0)

        return features, locations, probs
