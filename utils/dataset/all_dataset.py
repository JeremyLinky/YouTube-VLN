from inspect import ArgSpec
import logging
from itertools import groupby
import random
from typing import (
    List,
    Union,
    Tuple,
    TypeVar,
    Dict,
)
import math
from operator import itemgetter
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils.dataset.common import (
    load_json_data,
    perm2num,
    generate_negative_trajectories,
    load_shuffler,
    ytb_get_key,
    _check_enough_images,
    load_trajectories,
    ytb_generate_trajectory_from_listing,
    randomize_regions,
    randomize_tokens,
    load_tokens,
    generate_trajectory_out_listing,
    generate_trajectory_from_listing,
    merge_images,
    merge_frames,
    get_headings,
    shuffle_different,
    shuffle_non_adjacent,
    load_nav_graphs,
    load_distances,
    get_viewpoints,
    save_json_data,
    tokenize,
    InstructionGenerator,
    RephraseInstructionGenerator,
    ConcatenateInstructionGenerator,  
    YTBRephraseInstructionGenerator,
)

PhotoId = Union[int, Tuple[int, ...]]
Sample = Tuple[str, PhotoId]  # listing id, photo id
Trajectory = List[Sample]

from utils.dataset.features_reader import FeaturesReader

logger = logging.getLogger(__name__)

T = TypeVar("T")

class BaseDataset(Dataset):
    def load_captions(self, *args):
        # load captions from the json file
        raise NotImplementedError()
        
    def get_key(self, caption):
        raise NotImplementedError()
        
    def gather(self, captions):
        # gather photo_ids by listing and generate the photo_ids_by_listing dictionary
        raise NotImplementedError()
        
    def build_instructions(self, skeleton_path, separators):
        raise NotImplementedError()
        
    def _pick_photo_ids(
        self, listing_id: str
    ) -> Tuple[Trajectory, List[Trajectory], List[Trajectory], List[Trajectory]]:
        # select negative and positive photo ids
        raise NotImplementedError()
        
    def generate_instruction(self, build_instruction, traj):
        raise NotImplementedError()
        
    def get_listing_ids(self, listing_id):
        raise NotImplementedError()
        
    def get_feature_key(self, listing_id, pid):
        raise NotImplementedError()


    def __init__(
        self,
        args,
        caption_path: Union[Path, str],
        tokenizer: BertTokenizer,
        features_reader: FeaturesReader,
        masked_vision: bool,
        masked_language: bool,
        training: bool = False,
        separators: Tuple[str, ...] = tuple(),
        num_positives: int = 1,
        shuffler: str = "different",
    ):
        # set common parameters
        self.args = args
        self._tokenizer = tokenizer
        self._features_reader = features_reader
        self._masked_vision = masked_vision
        self._masked_language = masked_language
        self._training = training
        self._num_positives = num_positives
        self._shuffler = load_shuffler(shuffler)

        self._traj_judge = args.traj_judge and not args.ranking

        # load captions from the json file
        captions = self.load_captions(caption_path, tokenizer, self.args.max_instruction_length)

        # generate the key_id_to_caption dictionary
        self._key_id_to_caption = {
            self.get_key(caption): caption for caption in captions
        }

        # gather photo_ids by listing and generate the photo_ids_by_listing dictionary
        self.gather(captions)
        
        # WARNING: Make sure the listing contains enough images
        if not self.args.out_listing:
            self._photo_ids_by_listing = _check_enough_images(self._photo_ids_by_listing, self.args.min_path_length)
        
        # WARNING: Make sure the _video_ids after clean
        self._listing_ids = list(self._photo_ids_by_listing.keys())

        # TODO: why to use List?
        self._build_instructions: List[InstructionGenerator] = []
        self.build_instructions(self.args.skeleton_path, separators)


    def __len__(self):
        if self.args.out_listing:
            threshold = 15000 if self._training else 500
            return min(len(self._listing_ids), threshold)
        else:
            return len(self._listing_ids)
    def __getitem__(self, index: int):
        # get a random listing_id
        listing_id = self._listing_ids[index]

        # select negative and positive photo ids
        (
            positive_ids,
            negative_captions,
            negative_images,
            negative_random,
            order_labels
        ) = self._pick_photo_ids(listing_id)

        # get the order label of trajectory
        ordering_target = []
        order_atteneded_visual_feature = 1
        
        prob_order = 1
            
        for key in order_labels:
            if key == "normal_idx" or key == "negative_captions_idx":
                # Skip normal_idx and negative_captions_idx and consider only negative_images_idx
                continue
            else:
                for random_order_path in range(len(order_labels[key])):
                    if prob_order < 0.7:
                        order_atteneded_visual_feature = 1 # 1 indicates random and 0 indicates normal
                        temp = [v for v in order_labels[key][random_order_path] ]
                        # If the path length is too short, it is automatically filled to the longest path
                        temp +=  [-1] * (self.args.max_path_length - len(positive_ids))
                        ordering_target.append(temp)
                    else:
                        order_atteneded_visual_feature = 0 # 1 indicates random and 0 indicates normal
                        ordering_target.append([i for i in range(len(positive_ids))] + \
                                                [-1] * (self.args.max_path_length - len(positive_ids)))

        # get the positive pair
        build_instruction = random.choice(self._build_instructions)
        self.templete = None
        
        instructions = [self.generate_instruction(build_instruction,positive_ids)]
        f, b, p, m = self._get_visual_features(positive_ids)
        features, boxes, probs, masks = [f], [b], [p], [m] # This feature will patch to the longest length (8)
        
        if self._traj_judge: # Trajectory judgment task
            negative_traj = negative_captions + negative_images + negative_random
            for traj in negative_traj:
                instructions += [instructions[0]]
                f, b, p, m = self._get_visual_features(traj)
                features += [f]
                boxes += [b]
                probs += [p]
                masks += [m]

        else:
            # get the negative captions
            for traj in negative_captions:
                instructions += [self.generate_instruction(build_instruction,traj)]
                features += [features[0]]
                boxes += [boxes[0]]
                probs += [probs[0]]
                masks += [masks[0]]

            if self.args.negative_style == 'shuffle_instruction':
                # get the negative captions
                for traj in negative_images:
                    instructions += [self.generate_instruction(build_instruction,traj)]
                    features += [features[0]]
                    boxes += [boxes[0]]
                    probs += [probs[0]]
                    masks += [masks[0]]
            else:
                # get the negative images
                for traj in negative_images:
                    instructions += [instructions[0]]
                    f, b, p, m = self._get_visual_features(traj)
                    features += [f]
                    boxes += [b]
                    probs += [p]
                    masks += [m]

            # get the random images
            for traj in negative_random:
                instructions += [instructions[0]]
                f, b, p, m = self._get_visual_features(traj)
                features += [f]
                boxes += [b]
                probs += [p]
                masks += [m]

        # convert data into tensors
        image_features = torch.from_numpy(np.array(features)).float()
        image_boxes = torch.from_numpy(np.array(boxes)).float()
        image_probs = torch.from_numpy(np.array(probs)).float()
        image_masks = torch.from_numpy(np.array(masks)).long()
        instr_tokens = torch.from_numpy(np.array(instructions)).long()
        instr_mask = instr_tokens > 0
        segment_ids = torch.zeros_like(instr_tokens)
        instr_highlights = torch.zeros((image_features.shape[0], 0)).long()


        # randomly mask image features
        if self._masked_vision:
            image_features, image_targets, image_targets_mask = randomize_regions(
                image_features, image_probs, image_masks
            )
        else:
            image_targets = torch.ones_like(image_probs) / image_probs.shape[-1]
            image_targets_mask = torch.zeros_like(image_masks)

        # randomly mask instruction tokens
        if self._masked_language:
            instr_tokens, instr_targets = randomize_tokens(
                instr_tokens, instr_mask, self._tokenizer, self.args
            )
        else:
            instr_targets = torch.ones_like(instr_tokens) * -1

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self.args.max_path_length * self.args.max_num_boxes, self.args.max_instruction_length
        ).long()
        
        ordering_target = torch.tensor(ordering_target)
        if self._training:
            ranking_target = torch.tensor(0)
        else:
            ranking_target = torch.zeros(image_features.shape[0]).bool()
            ranking_target[0] = 1
        
        return (
            ranking_target,
            image_features,
            image_boxes,
            image_masks,
            image_targets,
            image_targets_mask,
            instr_tokens,
            instr_mask,
            instr_targets,
            instr_highlights,
            segment_ids,
            co_attention_mask,
            torch.tensor(self.get_listing_ids(listing_id)).long(),
            torch.ones(image_features.shape[0]).bool(),
            ordering_target,
            order_atteneded_visual_feature,
        )

    def _get_visual_features(self, trajectory: Trajectory):

        """ Get features for a given path. """
        path_length = min(len(trajectory), self.args.max_path_length)
        path_features, path_boxes, path_probs, path_masks = [], [], [], []
        for i, (listing_id, photo_id) in enumerate(trajectory):
            # get image features
            if isinstance(photo_id, int):
                photo_id = tuple([photo_id])
            keys = tuple(self.get_feature_key(listing_id, pid) for pid in photo_id)
            features, boxes, probs = self._features_reader[keys]

            num_boxes = min(len(boxes), self.args.max_num_boxes)

            # pad features and boxes (if needed)
            pad_features = np.zeros((self.args.max_num_boxes, 2048))
            pad_features[:num_boxes] = features[:num_boxes]

            pad_boxes = np.zeros((self.args.max_num_boxes, 12))
            pad_boxes[:num_boxes, :11] = boxes[:num_boxes, :11]  # type: ignore
            pad_boxes[:, 11] = np.ones(self.args.max_num_boxes) * i

            pad_probs = np.zeros((self.args.max_num_boxes, 1601))
            pad_probs[:num_boxes] = probs[:num_boxes]

            box_pad_length = self.args.max_num_boxes - num_boxes
            pad_masks = [1] * num_boxes + [0] * box_pad_length

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        # pad path lists (if needed)
        for path_idx in range(path_length, self.args.max_path_length):
            pad_features = np.zeros((self.args.max_num_boxes, 2048))
            pad_boxes = np.zeros((self.args.max_num_boxes, 12))
            pad_boxes[:, 11] = np.ones(self.args.max_num_boxes) * path_idx
            pad_probs = np.zeros((self.args.max_num_boxes, 1601))
            pad_masks = [0] * self.args.max_num_boxes

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        return (
            np.vstack(path_features),
            np.vstack(path_boxes),
            np.vstack(path_probs),
            np.hstack(path_masks),
        )


class YTbDataset(BaseDataset):
    def load_captions(self, *args):
        # load the generated data directly
        return load_json_data(args[0])

    def get_key(self, caption):
        return ytb_get_key(caption["video_id"], caption["frame_id"])

    def gather(self, captions):
        captions = sorted(captions, key=itemgetter("video_id"))
        
        # gather frame_ids by video_id
        self._photo_ids_by_listing = {
            video_id: list(frames)
            for video_id, frames in groupby(captions, key=itemgetter("video_id"))
        }

    def build_instructions(self, skeleton_path, separators):
        if skeleton_path == "":
            raise NotImplementedError()
        self._build_instructions.append(
            YTBRephraseInstructionGenerator(
                skeleton_path=skeleton_path,
                tokenizer=self._tokenizer,
                separators=separators,
                photo_id_to_caption=self._key_id_to_caption,
                max_instruction_length=self.args.max_instruction_length,
                random_action=self.args.random_action
            )
        )

    def __init__(
        self,
        testset_path: Union[Path, str],
        **kwargs
    ):
        super().__init__(**kwargs)

        # load trajectories of video
        self._trajectories = {}
        for vid in self._listing_ids:
            trajectory = []
            for frame in self._photo_ids_by_listing[vid]:
                if frame['caption'] != "":
                    trajectory.append(frame['frame_id'])
            self._trajectories[vid] = trajectory

        self._testset = {
            key: load_trajectories(value,'ytb') for key, value in testset_path.items()
        } if not self._training else {}

        if self.args.out_listing:
            raise NotImplementedError()

    def _pick_photo_ids(
        self, listing_id: str, 
    ) -> Tuple[Trajectory, List[Trajectory], List[Trajectory], List[Trajectory]]:
        if not self._training:
            # val
            if self._traj_judge:
                return self._testset['traj'][listing_id]
            else:
                return self._testset['ranking'][listing_id]

        # Generate a positive path based on listing_id
        positive_trajectory, captioned = ytb_generate_trajectory_from_listing(
            listing_id,
            self._listing_ids,
            self._photo_ids_by_listing,
            self._key_id_to_caption,
            self._trajectories[listing_id],
            self.args.min_path_length,
            self.args.max_path_length,
            self.args.min_captioned,
            self.args.max_captioned,
        )

        if self._training and self.args.no_serial:
            # shuffle the correct path as correct path
            relative = list(range(len(positive_trajectory)))
            random.shuffle(relative)
            positive_trajectory = [positive_trajectory[i] for i in relative]
            captioned = [captioned[i] for i in relative]
        
        # Generate negative paths from the positive path
        neg_captions, neg_images, neg_randoms, order_labels = generate_negative_trajectories(
            positive_trajectory,
            captioned,
            self._listing_ids,
            self._photo_ids_by_listing,
            self._key_id_to_caption,
            self.args.num_negatives,
            shuffler=self._shuffler,
            dataset_type='ytb',
            traj_judge=self._traj_judge,
            negative_style=self.args.negative_style,
        )

        return positive_trajectory, neg_captions, neg_images, neg_randoms, order_labels

    def generate_instruction(self, build_instruction, traj):
        sentence, templete = build_instruction(traj,self.templete)
        self.templete = templete
        return sentence
    
    def get_listing_ids(self, listing_id):
        return 0    # useless

    def get_feature_key(self, listing_id, pid):
        return ytb_get_key(listing_id, pid)


class BnBDataset(BaseDataset):
    def load_captions(self, *args):
        # load tokens of captions
        return load_tokens(*args)
    
    def get_key(self, caption):
        return int(caption["photo_id"])

    def gather(self, captions):
        captions = sorted(captions, key=itemgetter("listing_id"))
        
        # gather photo_ids by listing
        self._photo_ids_by_listing = {
            str(listing): merge_images(photos)
            for listing, photos in groupby(captions, key=itemgetter("listing_id"))
        }

    def build_instructions(self, skeleton_path, separators):
        if skeleton_path == "":
            self._build_instructions.append(
                ConcatenateInstructionGenerator(
                    tokenizer=self._tokenizer,
                    separators=separators,
                    photo_id_to_caption=self._key_id_to_caption,
                    max_instruction_length=self.args.max_instruction_length,
                )
            )
        else:
            self._build_instructions.append(
                RephraseInstructionGenerator(
                    skeleton_path=skeleton_path,
                    tokenizer=self._tokenizer,
                    separators=separators,
                    photo_id_to_caption=self._key_id_to_caption,
                    max_instruction_length=self.args.max_instruction_length,
                )
            )

    def __init__(
        self,
        testset_path: Union[Path, str],
        **kwargs
    ):
        super().__init__(**kwargs)

        self._testset = {
            key: load_trajectories(value,'bnb') for key, value in testset_path.items()
        } if not self._training else {}

    def _pick_photo_ids(
        self, listing_id: str, 
    ) -> Tuple[Trajectory, List[Trajectory], List[Trajectory], List[Trajectory]]:
        if not self._training:
            # val
            if self._traj_judge:
                return self._testset['traj'][int(listing_id)]
            else:
                return self._testset['ranking'][int(listing_id)]

        # Select the generation method
        fn = generate_trajectory_from_listing if not self.args.out_listing else generate_trajectory_out_listing

        # Generate a positive path based on listing_id
        positive_trajectory, captioned = fn(
            listing_id,
            self._listing_ids,
            self._photo_ids_by_listing,
            self._key_id_to_caption,
            self.args.min_path_length,
            self.args.max_path_length,
            self.args.min_captioned,
            self.args.max_captioned,
        )

        # Generate negative paths from the positive path
        neg_captions, neg_images, neg_randoms, order_labels = generate_negative_trajectories(
            positive_trajectory,
            captioned,
            self._listing_ids,
            self._photo_ids_by_listing,
            self._key_id_to_caption,
            self.args.num_negatives,
            shuffler=self._shuffler,
            dataset_type='bnb',
            traj_judge=self._traj_judge,
            negative_style=self.args.negative_style,
        )

        if self.args.out_listing:
            neg_randoms = []

        return positive_trajectory, neg_captions, neg_images, neg_randoms, order_labels
    
    def generate_instruction(self, build_instruction, traj):
        return build_instruction(traj)

    def get_listing_ids(self, listing_id):
        return int(listing_id)

    def get_feature_key(self, listing_id, pid):
        return f"{listing_id}-{pid}"


class BeamDataset(Dataset):
    def __init__(
        self,
        args,
        vln_path: str,
        beam_path: str,
        tokenizer: BertTokenizer,
        features_reader: FeaturesReader,
        num_beams: int,
        num_beams_strict: bool,
        training: bool,
        masked_vision: bool,
        masked_language: bool,
        default_gpu: bool,
        ground_truth_trajectory: bool,
        shuffle_visual_features: bool,
        shuffler: str = "different",
        **kwargs,
    ):
        # set common parameters
        self.args = args
        self._features_reader = features_reader
        self._training = training
        self._masked_vision = masked_vision
        self._masked_language = masked_language
        self._default_gpu = default_gpu
        self._ground_truth_trajectory = ground_truth_trajectory
        self._shuffle_visual_features = shuffle_visual_features
        self._batch_size = args.batch_size // args.gradient_accumulation_steps

        self._traj_judge = args.traj_judge and not (args.ranking or args.not_traj_judge_data)

        # load and tokenize data (with caching)
        tokenized_path = f"_tokenized_{self.args.max_instruction_length}".join(
            os.path.splitext(vln_path)
        )
        if os.path.exists(tokenized_path):
            self._vln_data = load_json_data(tokenized_path)
        else:
            self._vln_data = load_json_data(vln_path)
            tokenize(self._vln_data, tokenizer, self.args.max_instruction_length)
            save_json_data(self._vln_data, tokenized_path)
        self._tokenizer = tokenizer

        # load navigation graphs
        scan_list = list(set([item["scan"] for item in self._vln_data]))

        self._graphs = load_nav_graphs(scan_list)
        self._distances = load_distances(scan_list)

        # get all of the viewpoints for this dataset
        self._viewpoints = get_viewpoints(scan_list, self._graphs, features_reader)

        # in training we only need 4 beams
        self._num_beams = num_beams
        if training:
            num_beams_strict = False

        # load beamsearch data
        temp_beam_data = load_json_data(beam_path)

        # filter beams based on length
        self._beam_data = []
        for idx, item in enumerate(temp_beam_data):
            if len(item["ranked_paths"]) >= num_beams:
                if num_beams_strict:
                    item["ranked_paths"] = item["ranked_paths"][:num_beams]
                self._beam_data.append(item)
            elif default_gpu:
                logger.warning(
                    f"skipping index: {idx} in beam data in from path: {beam_path}"
                )

        # get mapping from path id to vln index
        path_to_vln = {}
        for idx, vln_item in enumerate(self._vln_data):
            path_to_vln[vln_item["path_id"]] = idx

        # get mapping from beam to vln
        self._beam_to_vln = {}
        for idx, beam_item in enumerate(self._beam_data):
            path_id = int(beam_item["instr_id"].split("_")[0])
            if path_id not in path_to_vln:
                if default_gpu:
                    logger.warning(f"Skipping beam {beam_item['instr_id']}")
                continue
            self._beam_to_vln[idx] = path_to_vln[path_id]

        if shuffler == "different":
            self._shuffler = shuffle_different
        elif shuffler == "nonadj":
            self._shuffler = shuffle_non_adjacent
        else:
            raise ValueError(f"Unexpected shuffling mode ({shuffler})")

    def __len__(self):
        return len(self._beam_data)

    def __getitem__(self, beam_index: int):
        vln_index = self._beam_to_vln[beam_index]
        vln_item = self._vln_data[vln_index]

        self._visual_data = {}
        self._visual_data['beam_index'] = beam_index
        self._visual_data['path'] = []
        self._visual_data['headings'] = []
        self._visual_data['next_headings'] = []
        self._visual_data['instructions'] = vln_item["instructions"]

        # get beam info
        path_id, instruction_index = map(
            int, self._beam_data[beam_index]["instr_id"].split("_")
        )

        # get vln info
        scan_id = vln_item["scan"]
        heading = vln_item["heading"]
        gt_path = vln_item["path"]

        # get the instruction data
        instr_tokens = torch.tensor(vln_item["instruction_tokens"][instruction_index])
        instr_mask = instr_tokens > 0

        segment_ids = torch.zeros_like(instr_tokens)

        instr_highlights = torch.tensor([])

        # get all of the paths
        beam_paths = []

        for ranked_path in self._beam_data[beam_index]["ranked_paths"]:
            beam_paths.append([p for p, _, _ in ranked_path])

        success = self._get_path_success(scan_id, gt_path, beam_paths)
        target: Union[List[int], int]
        order_labels = []

        # select one positive and three negative paths
        if self._training:
            # special case for data_aug with negative samples
            if "positive" in vln_item and not vln_item["positive"][instruction_index]:
                target = -1
                selected_paths = beam_paths[: self._num_beams]
                assert not self._ground_truth_trajectory, "Not compatible"

            # not enough positive or negative paths (this should be rare) we only need one successful path
            if np.sum(success == 1) == 0 or np.sum(success == 0) < self._num_beams - 1: # TODO why is "self._num_beams - 1"?
                target = -1  # default ignore index
                if self._ground_truth_trajectory:
                    selected_paths = [self._vln_data[vln_index]["path"]] + beam_paths[
                        : self._num_beams - 1
                    ]
                else:
                    selected_paths = beam_paths[: self._num_beams]
            else:
                target = 0
                selected_paths = []
                # first select a positive
                if self._ground_truth_trajectory:
                    selected_paths.append(self._vln_data[vln_index]["path"])
                else:
                    idx = np.random.choice(np.where(success == 1)[0])  # type: ignore
                    selected_paths.append(beam_paths[idx])
                    
                if not self._traj_judge:
                    # next select three negatives
                    idxs = np.random.choice(  # type: ignore
                        np.where(success == 0)[0], size=self._num_beams - 1, replace=False
                    )
                    for idx in idxs:
                        selected_paths.append(beam_paths[idx])

            # shuffle the visual features from the ground truth as a free negative path
            path = self._vln_data[vln_index]["path"]
            path_range = range(len(path))

            if self._shuffle_visual_features:

                # we only shuffle the order of positive samples
                for corr, _ in zip(self._shuffler(path_range), range(self.args.num_negatives)):
                    order_labels.append(corr)
                    corr_path = [path[i] for i in corr]
                    selected_paths += [corr_path]
            else:
                if not self._traj_judge:
                    order_labels = [list(range(self.args.max_path_length))]*self.args.num_negatives
        else:
            if self._traj_judge:
                target = np.zeros(len(beam_paths))
            else:
                target = success
            selected_paths = beam_paths

            # # This should be used only for testing the influence of shuffled sequences!
            # if self._shuffle_visual_features:
            #     if isinstance(target, int):
            #         raise ValueError("fix mypy")

            #     # we shuffled all positive trajectories
            #     # the target is now zero everywhere
            #     for i in np.arange(len(success))[success.astype("bool")]:
            #         if i > self.args.num_negatives:
            #             break
            #         selected_paths.append(next(self._shuffler(selected_paths[i])))
            #         target = np.append(target, 0)
            
            # shuffle the visual features from the ground truth as a free negative path
            if self._shuffle_visual_features:
                if isinstance(target, int):
                    raise ValueError("fix mypy")

                # we shuffled all positive trajectories
                # the target is now zero everywhere
                for i in np.arange(len(success))[success.astype("bool")]:
                    if i > self.args.num_negatives:
                        break
                    selected_paths.append(next(self._shuffler(selected_paths[i])))
                    target = np.append(target, 0)
            
            if self._batch_size !=1 and len(target) != self._num_beams:
                target = np.tile(target,int(self._num_beams/len(target))+1)[:self._num_beams]
                selected_paths = (selected_paths*(int(self._num_beams/len(selected_paths))+1))[:self._num_beams]

            order_labels = [list(range(self.args.max_path_length))]*self.args.num_negatives
        
        # get path features
        features, boxes, probs, masks = [], [], [], []
        if self._training and self._traj_judge:
            # positive
            path_length = min(len(selected_paths[0]), self.args.max_path_length)
            pos = selected_paths[0][:path_length]
            path_features = [list(self._get_path_features(scan_id, pos, heading))]

            path_range = range(len(pos))
            normal_path = list(path_range)

            # get negative
            max_out_num = 4 # max number of out_listing

            shuffle_type = random.randint(1,3)
            
            if shuffle_type == 1:
                # shuffle the order of the same path
                for corr, _ in zip(self._shuffler(path_range), range(self.args.num_negatives)):
                    order_labels.append(corr)
                    path = [pos[i] for i in corr]
                    path_features.append(list(self._get_path_features(scan_id, path, heading)))
            elif shuffle_type == 2:
                # shuffle the order of positive samples
                for corr, _ in zip(self._shuffler(path_range), range(self.args.num_negatives)):
                    order_labels.append(corr)
                    sop_path = [self._visual_data['path'][0][i] for i in corr]
                    sop_headings = [self._visual_data['headings'][0][i] for i in corr]
                    sop_next_headings = [self._visual_data['next_headings'][0][i] for i in corr]
                    self._visual_data['path'].append(sop_path)
                    self._visual_data['headings'].append(sop_headings)
                    self._visual_data['next_headings'].append(sop_next_headings)
                    f = []
                    b = []
                    p = []
                    m =[]
                    for i in path_range:
                        f.append(path_features[0][0][corr[i]])
                        b.append(path_features[0][1][corr[i]])
                        p.append(path_features[0][2][corr[i]])
                        m.append(path_features[0][3][corr[i]])
                    # pad path lists (if needed)
                    for path_idx in range(path_length, self.args.max_path_length):
                        f.append(path_features[0][0][path_idx])
                        b.append(path_features[0][1][path_idx])
                        p.append(path_features[0][2][path_idx])
                        m.append(path_features[0][3][path_idx])
                    path_features.append([f, b, p, m])
            elif shuffle_type == 3:
                # other scan
                index = random.randint(0, len(self._vln_data) - 1)
                while(index == vln_index):
                    # avoid the same path
                    index = random.randint(0, len(self._vln_data) - 1)
                path2 = self._vln_data[index]["path"]
                scan_id2 = self._vln_data[index]["scan"]
                for _ in range(self.args.num_negatives):
                    order_labels.append(normal_path)
                    
                    min_len = min(len(pos), len(path2))
                    out_num = random.randint(1, min(max_out_num, min_len)) # number of out_listing
                    temp = path_features[0].copy()

                    temp_path = self._visual_data['path'][0].copy()
                    temp_headings = self._visual_data['headings'][0].copy()
                    temp_next_headings = self._visual_data['next_headings'][0].copy()
                    # use self to record information
                    self._temp_path=[]
                    self._temp_headings=[]
                    self._temp_next_headings=[]

                    for i in random.sample(list(range(min_len)), out_num):
                        # replace out_ix of the positive path with other path (different scan)
                        f, b, p, m = self._get_feature(scan_id2, path2[i], i)
                        temp[0][i] = f
                        temp[1][i] = b
                        temp[2][i] = p
                        temp[3][i] = m
                        temp_path[i] = self._temp_path
                        temp_headings[i] = self._temp_headings
                        temp_next_headings[i] = self._temp_next_headings
                    path_features.append(temp)

                    self._visual_data['path'].append(temp_path)
                    self._visual_data['headings'].append(temp_headings)
                    self._visual_data['next_headings'].append(temp_next_headings)

            for i in path_features:
                features.append(np.vstack(i[0]))
                boxes.append(np.vstack(i[1]))
                probs.append(np.vstack(i[2]))
                masks.append(np.hstack(i[3]))
        else:
            for path in selected_paths:
                f, b, p, m = self._get_path_features(scan_id, path, heading)
                features.append(np.vstack(f))
                boxes.append(np.vstack(b))
                probs.append(np.vstack(p))
                masks.append(np.hstack(m))


        # get the order label of trajectory
        ordering_target = []
        order_atteneded_visual_feature = 1

        prob_order = 1

        for random_order_path in range(len(order_labels)):
            if prob_order < 0.7:
                order_atteneded_visual_feature = 1
                max_length = min(self.args.max_path_length, len(order_labels[random_order_path]))

                negative_path_order = order_labels[random_order_path][:max_length]

                negative_path_order +=  [-1] * (self.args.max_path_length - len(order_labels[random_order_path]))

                ordering_target.append(negative_path_order)
            else:
                order_atteneded_visual_feature = 0
                max_length = min(self.args.max_path_length, len(selected_paths[0]))

                postive_path_order = [i for i in range(len(selected_paths[0]))][:max_length]

                postive_path_order += [-1] * (self.args.max_path_length - len(selected_paths[0]))

                ordering_target.append(postive_path_order)


        # convert data into tensors
        image_features = torch.from_numpy(np.array(features)).float()  # torch.tensor(features).float()
        image_boxes = torch.from_numpy(np.array(boxes)).float()
        image_probs = torch.from_numpy(np.array(probs)).float()
        image_masks = torch.from_numpy(np.array(masks)).long()
        instr_tokens = instr_tokens.repeat(len(features), 1).long()
        instr_mask = instr_mask.repeat(len(features), 1).long()
        segment_ids = segment_ids.repeat(len(features), 1).long()
        instr_highlights = instr_highlights.repeat(len(features), 1).long()

        # randomly mask image features
        if self._masked_vision:
            image_features, image_targets, image_targets_mask = randomize_regions(
                image_features, image_probs, image_masks
            )
        else:
            image_targets = torch.ones_like(image_probs) / image_probs.shape[-1]
            image_targets_mask = torch.zeros_like(image_masks)

        # randomly mask instruction tokens
        if self._masked_language:
            instr_tokens, instr_targets = randomize_tokens(
                instr_tokens, instr_mask, self._tokenizer, self.args
            )
        else:
            instr_targets = torch.ones_like(instr_tokens) * -1

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self.args.max_path_length * self.args.max_num_boxes, self.args.max_instruction_length
        ).long()
        instr_id = torch.tensor([path_id, instruction_index]).long()

        target = torch.tensor(target).long()
        ordering_target = torch.tensor(ordering_target)
        

        return (
            target,
            image_features,
            image_boxes,
            image_masks,
            image_targets,
            image_targets_mask,
            instr_tokens,
            instr_mask,
            instr_targets,
            instr_highlights,
            segment_ids,
            co_attention_mask,
            instr_id,
            torch.ones(image_features.shape[0]).bool(),
            ordering_target,
            order_atteneded_visual_feature,
        )

    def _get_path_success(self, scan_id, path, beam_paths, success_criteria=3):
        d = self._distances[scan_id]
        success = np.zeros(len(beam_paths))
        for idx, beam_path in enumerate(beam_paths):
            if d[path[-1]][beam_path[-1]] < success_criteria:
                success[idx] = 1
        return success

    # TODO move to utils
    def _get_path_features(self, scan_id: str, path: List[str], first_heading: float):

        path_length = min(len(path), self.args.max_path_length)
        path_features, path_boxes, path_probs, path_masks = [], [], [], []
        """ Get features for a given path. """
        headings = get_headings(self._graphs[scan_id], path, first_heading)
        # for next headings duplicate the last
        next_headings = headings[1:] + [headings[-1]]
        self._visual_data['path'].append([(scan_id,viewpoint) for viewpoint in path])
        self._visual_data['headings'].append(headings)
        self._visual_data['next_headings'].append(next_headings)

        path_length = min(len(path), self.args.max_path_length)
        path_features, path_boxes, path_probs, path_masks = [], [], [], []
        for path_idx, path_id in enumerate(path[:path_length]):
            key = scan_id + "-" + path_id

            # get image features
            features, boxes, probs = self._features_reader[
                key, headings[path_idx], next_headings[path_idx],
            ]
            num_boxes = min(len(boxes), self.args.max_num_boxes)

            # pad features and boxes (if needed)
            pad_features = np.zeros((self.args.max_num_boxes, 2048))
            pad_features[:num_boxes] = features[:num_boxes]

            pad_boxes = np.zeros((self.args.max_num_boxes, 12))
            pad_boxes[:num_boxes, :11] = boxes[:num_boxes, :11]
            pad_boxes[:, 11] = np.ones(self.args.max_num_boxes) * path_idx

            pad_probs = np.zeros((self.args.max_num_boxes, 1601))
            pad_probs[:num_boxes] = probs[:num_boxes]

            box_pad_length = self.args.max_num_boxes - num_boxes
            pad_masks = [1] * num_boxes + [0] * box_pad_length

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        # pad path lists (if needed)
        for path_idx in range(path_length, self.args.max_path_length):
            pad_features = np.zeros((self.args.max_num_boxes, 2048))
            pad_boxes = np.zeros((self.args.max_num_boxes, 12))
            pad_boxes[:, 11] = np.ones(self.args.max_num_boxes) * path_idx
            pad_probs = np.zeros((self.args.max_num_boxes, 1601))
            pad_masks = [0] * self.args.max_num_boxes

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        return (
            path_features,
            path_boxes,
            path_probs,
            path_masks,
        )

    def _get_feature(self, scan_id: str, path_id: str, path_idx: float):
        """ Get a feature for a given path_id. """
        # the return valule of np.arctan2 is [-pi / 2, pi / 2]
        headings = random.uniform(-np.pi/2,np.pi/2)
        next_headings = random.uniform(-np.pi/2,np.pi/2)

        # record the trajectory informative for visualization
        self._temp_path=(scan_id,path_id)
        self._temp_headings=headings
        self._temp_next_headings=next_headings

        key = scan_id + "-" + path_id

        # get image features
        features, boxes, probs = self._features_reader[
            key, headings, next_headings,
        ]
        num_boxes = min(len(boxes), self.args.max_num_boxes)

        # pad features and boxes (if needed)
        pad_features = np.zeros((self.args.max_num_boxes, 2048))
        pad_features[:num_boxes] = features[:num_boxes]

        pad_boxes = np.zeros((self.args.max_num_boxes, 12))
        pad_boxes[:num_boxes, :11] = boxes[:num_boxes, :11]
        pad_boxes[:, 11] = np.ones(self.args.max_num_boxes) * path_idx

        pad_probs = np.zeros((self.args.max_num_boxes, 1601))
        pad_probs[:num_boxes] = probs[:num_boxes]

        box_pad_length = self.args.max_num_boxes - num_boxes
        pad_masks = [1] * num_boxes + [0] * box_pad_length

        return (
            pad_features,
            pad_boxes,
            pad_probs,
            pad_masks,
        )

    def _get_path_id(self, beam_index: int):
        vln_index = self._beam_to_vln[beam_index]
        vln_item = self._vln_data[vln_index]
        return (vln_item['scan'], vln_item['path_id'])
    
    def get_sub_beam(self, rate_per_scan: float=0.15):
        scans = {}
        sub_indices = []

        # get all scans and paths
        for beam_index in range(self.__len__()):
            scan, path_id = self._get_path_id(beam_index)
            if scan not in scans:
                scans[scan] = {}
            if path_id not in scans[scan]:
                scans[scan][path_id] = []
            scans[scan][path_id].append(beam_index)
        
        # random sample
        for scan, paths in scans.items():
            path_key = random.sample(paths.keys(), math.ceil(len(paths)*rate_per_scan))
            sub_indices += [paths[key][0] for key in path_key]      # default to the first instruction path
        return sub_indices