import json
from typing import List, Dict, Union, TypeVar, Iterator,Tuple, Optional, Callable, Iterable
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from transformers import PreTrainedTokenizer, BertTokenizer
import math
import random
import copy
from utils.dataset.features_reader import (
    Trajectory,
    PhotoId,
)
import itertools
from itertools import groupby
import re
from scripts.video_process.gen_instructions4train import run_insertion
from tqdm import tqdm

def pad_packed(t: torch.Tensor, mask: Union[torch.Tensor, torch.BoolTensor]) -> torch.Tensor:
    mask = mask.bool()
    out = mask.clone().to(t.dtype)
    out[mask] = t
    out[~mask] = -float("inf")
    return out


def load_json_data(path):
    with open(path, "r") as fid:
        data = json.load(fid)
    return data


def save_json_data(data, path):
    with open(path, "w") as fid:
        json.dump(data, fid, indent=2)


def load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    graphs = {}
    for scan in scans:
        with open("data/connectivity/%s_connectivity.json" % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item["included"]:
                    for j, conn in enumerate(item["unobstructed"]):
                        if conn and data[j]["included"]:
                            positions[item["image_id"]] = np.array(
                                [item["pose"][3], item["pose"][7], item["pose"][11]]
                            )
                            assert data[j]["unobstructed"][
                                i
                            ], "Graph should be undirected"
                            G.add_edge(
                                item["image_id"],
                                data[j]["image_id"],
                                weight=distance(item, data[j]),
                            )
            nx.set_node_attributes(G, values=positions, name="position")
            graphs[scan] = G
    return graphs


def load_distances(scans):
    distances = {}
    for scan in scans:
        with open(f"data/distances/{scan}_distances.json", "r") as fid:
            distances[scan] = json.load(fid)
    return distances


def get_headings(g, path, first_heading):
    # get xy positions for path
    pos = nx.get_node_attributes(g, "position")
    pos = {node: pos[node][:2] for node in path}

    # calculate headings
    headings = [first_heading]
    for source, target in zip(path[:-1], path[1:]):
        dx = pos[target][0] - pos[source][0]
        dy = pos[target][1] - pos[source][1]
        # use dx/dy because heading is from north (i.e. y)
        abs_heading = np.arctan2(dx, dy)
    
        headings.append(abs_heading)
    
    return headings


def index(subseq, seq):
    i, n, m = -1, len(seq), len(subseq)
    try:
        while True:
            i = seq.index(subseq[0], i + 1, n - m + 1)
            if subseq == seq[i : i + m]:
                return i
    except ValueError:
        return -1


def tokenize(
    data: List[Dict], tokenizer: PreTrainedTokenizer, max_instruction_length: int
):
    pad, cls, sep = tokenizer.convert_tokens_to_ids(["[PAD]", "[CLS]", "[SEP]"])  # type: ignore

    for item in tqdm(data,total=len(data),):
        item["instruction_tokens"] = []
        if "highlights" in item:
            item["instruction_highlights"] = []
        if "perturbations" in item:
            item["perturbation_tokens"] = [[] for _ in item["instructions"]]
        if "perturbation_highlights" in item:
            item["perturbation_highlight_masks"] = [[] for _ in item["instructions"]]

        for i, instruction in enumerate(item["instructions"]):
            tokens = tokenizer.tokenize(instruction)

            # add a classification and seperator tokens
            tokens = [cls] + [
                tokenizer.vocab[token] for token in tokens  # type: ignore
            ]
            tokens = tokens[: max_instruction_length - 1] + [sep]
            pad_length = max_instruction_length - len(tokens)
            pad_tokens = tokens + [pad] * pad_length
            item["instruction_tokens"].append(pad_tokens)

            # create a highlight version
            if "highlights" in item:
                highlights = []
                cursor = 0
                for word in item["highlights"][i]:
                    token = tokenizer.tokenize(word)
                    token_id = [tokenizer.vocab[t] for t in token]  # type: ignore
                    increment = index(token_id, tokens[cursor:])
                    if increment == -1:
                        continue
                    highlights += [False] * increment + [True] * len(token_id)
                    cursor += increment + len(token_id)

                # pad lists
                pad_length = max_instruction_length - len(highlights)
                highlights = highlights + [False] * pad_length

                # add to data
                item["instruction_highlights"].append(highlights)

            # create a perturbation version
            if "perturbations" in item:
                for j, inst in enumerate(item["perturbations"][i]):
                    tokens = tokenizer.tokenize(inst)
                    tokens = [cls] + [
                        tokenizer.vocab[token] for token in tokens  # type: ignore
                    ]
                    tokens = tokens[: max_instruction_length - 1] + [sep]
                    pad_length = max_instruction_length - len(tokens)
                    pad_tokens = tokens + [pad] * pad_length
                    item["perturbation_tokens"][i].append(pad_tokens)

                    # create a perturbation + highlight version
                    if "perturbation_highlights" in item:
                        highlights = []
                        cursor = 0
                        for word in item["perturbation_highlights"][i][j]:
                            token = tokenizer.tokenize(word)
                            token_id = [tokenizer.vocab[t] for t in token]  # type: ignore
                            increment = index(token_id, tokens[cursor:])
                            if increment == -1:
                                continue
                            highlights += [False] * increment + [True] * len(token_id)
                            cursor += increment + len(token_id)

                        # pad lists
                        pad_length = max_instruction_length - len(highlights)
                        highlights = highlights + [False] * pad_length

                        # add to data
                        item["perturbation_highlight_masks"][i].append(highlights)


def load_tokens(
    path: Union[Path, str], tokenizer: PreTrainedTokenizer, max_instruction_length: int
) -> List[Dict]:
    ppath = Path(path)
    assert ppath.suffix == ".json", ppath

    # load and tokenize data (with caching)
    tokenized_path = (
        ppath.parent / f"{ppath.stem}_tokenized_{max_instruction_length}{ppath.suffix}"
    )

    if tokenized_path.is_file():
        data = load_json_data(tokenized_path)
    else:
        data = load_json_data(ppath)
        tokenize(data, tokenizer, max_instruction_length)
        save_json_data(data, tokenized_path)
    return data


def randomize_tokens(tokens, mask, tokenizer, args):
    """ Return tokens randomly masked using standard BERT probabilities. """
    """
        self._tokenizer.vocab['left']
        2187
        self._tokenizer.vocab['forward']
        2830
        self._tokenizer.vocab['right']
        2157
    """
    

    targets = torch.ones_like(tokens) * -1

    # get random data
    p = torch.rand_like(tokens.float()) * mask.float()
    random_tokens = torch.randint_like(tokens, len(tokenizer.vocab))

    # set targets for masked tokens
    thresh = 0.85
    
    if args.mask_action_rate > 0:
        action_tokens = [2187, 2830, 2157]
        action_positions_x = []
        action_positions_y = []
        for ac in action_tokens:
            index_xy = torch.where(tokens==ac)
            action_positions_x.append(index_xy[0])
            action_positions_y.append(index_xy[1])

        action_positions_x = torch.cat(action_positions_x,dim=0)
        action_positions_y = torch.cat(action_positions_y,dim=0)

        mask_action_rate = args.mask_action_rate

        mask_index = np.random.choice(range(len(action_positions_x)), int(mask_action_rate * len(action_positions_x)))

        for mi in mask_index:
            targets[action_positions_x[mi], action_positions_y[mi]] = tokens[action_positions_x[mi], action_positions_y[mi]]
            tokens[action_positions_x[mi], action_positions_y[mi]] = tokenizer.vocab["[MASK]"]
            p[action_positions_x[mi], action_positions_y[mi]] = thresh * 0.9

    targets[p >= thresh] = tokens[p >= thresh]

    # progressively overwrite tokens while increasing the threshold

    # replace 80% with '[MASK]' token
    tokens[p >= thresh] = tokenizer.vocab["[MASK]"]

    # replace 10% with a random word
    thresh = 0.85 + 0.15 * 0.8
    tokens[p >= thresh] = random_tokens[p >= thresh]

    # keep 10% unchanged
    thresh = 0.85 + 0.15 * 0.9
    tokens[p >= thresh] = targets[p >= thresh]

    return tokens, targets

def randomize_regions(features, probs, mask):
    """Return features after randomly masking using ViLBERT probabilities.

    Let B equal the batch size and N equal the number of regions.

    Parameters
    ----------
    features : torch.tensor, (B, N, 2048)
        The original feature vectors.
    probs : torch.tensor, (B, N, 2048)
        The target probability distribution for each region.
    mask : torch.tensor, (B, N)
        A zero-one mask where zeros represent missing regions.
    """
    targets = torch.ones_like(probs) / probs.shape[-1]
    targets_mask = torch.zeros_like(mask)

    p = torch.rand_like(mask.float()) * mask.float()

    # set targets for masked regions
    thresh = 0.85
    targets[p >= thresh] = probs[p >= thresh]
    targets_mask[p >= thresh] = 1

    # replace 90% of the masked features with zeros
    thresh = 0.85 + 0.15 * 0.1
    features[p >= thresh] = 0

    return features, targets, targets_mask

def get_viewpoints(scan_list, graphs, feature_reader):
    """ Return a list of viewpoints that are in the graphs and feature reader. """
    viewpoints = {}
    for scan in scan_list:
        graph_viewpoints = set(graphs[scan].nodes())
        feats_viewpoints = feature_reader.viewpoints[scan]
        viewpoints[scan] = feats_viewpoints.intersection(graph_viewpoints)
    return viewpoints

T = TypeVar("T")

def perm2num(p):
    n = len(p)
    num = 0
    k = 1
    i = n -2
    while(i >= 0):
        j = i + 1
        while( j < n ):
            if p[j]<p[i]:
                num += k
            j += 1
        k = math.factorial(n - i)
        i -= 1
    return num

def ytb_load_tokens(
    path: Union[Path, str], tokenizer: PreTrainedTokenizer
) -> List[Dict]:
    ppath = Path(path)
    assert ppath.suffix == ".json", ppath

    # load and tokenize data (with caching)
    tokenized_path = (
        ppath.parent / f"{ppath.stem}_tokenized_{ppath.suffix}"
    )

    if tokenized_path.is_file():
        data = load_json_data(tokenized_path)
    else:
        data = load_json_data(ppath)
        ytb_tokenize(data, tokenizer)
        save_json_data(data, tokenized_path)
    return data

def ytb_tokenize(
    data: List[Dict], tokenizer: PreTrainedTokenizer
):
    pad, cls, sep = tokenizer.convert_tokens_to_ids(["[PAD]", "[CLS]", "[SEP]"])  # type: ignore

    for item in data:
        item["instruction_tokens"] = []
        for i, instruction in enumerate(item["instructions"]):
            tokens = tokenizer.tokenize(instruction)

            # add a classification and seperator tokens
            tokens = [cls] + [
                tokenizer.vocab[token] for token in tokens  # type: ignore
            ]
            item["instruction_tokens"].append(tokens)

def random_fill(captioned_idx: List[T], captionless_idx: List[T], seq: List[T], fillers: List[T]) -> None:
    captioned_idx_copy = copy.deepcopy(captioned_idx)
    n = len(seq)
    random.shuffle(fillers)
    for idx, x in enumerate(fillers):
        insert_pos = random.randint(0, n - 1)
        captioned_idx_copy.insert(insert_pos, captionless_idx[idx])
        seq.insert(insert_pos, x)
        n += 1
    return captioned_idx_copy

def random_caption_image(video_ids: List[str], frame_ids_by_video: Dict[str, List[PhotoId]]) -> Tuple[str, int]:
    l = random.choice(video_ids)
    while(True):
        frame = random.choice(frame_ids_by_video[l])
        if frame["caption"] != "":
            p = frame["frame_id"]
            return l, p

def random_captionless_image(video_ids: List[str], frame_ids_by_video: Dict[str, List[PhotoId]]) -> Tuple[str, int]:
    l = random.choice(video_ids)
    while(True):
        frame = random.choice(frame_ids_by_video[l])
        if frame["caption"] == "":
            p = frame["frame_id"]
            return l, p
            
def random_image(listing_ids: List[int], photos_by_listing: Dict[int, List[PhotoId]]) -> Tuple[int, PhotoId]:
    l = random.choice(listing_ids)
    p = random.choice(photos_by_listing[l])
    return l, p

def is_captionless(photo_id: PhotoId, photo_id_to_caption: Dict[int, Dict]):
    if isinstance(photo_id, (list, tuple)):
        return all(is_captionless(pid, photo_id_to_caption) for pid in photo_id)
    caption = photo_id_to_caption[photo_id]
    return sum(caption["instruction_tokens"][0]) < 204
    
def generate_negative_trajectories(
    positive_path: Trajectory,
    states: Union[List[bool], List[int]],
    room_ids: Union[List[str],List[bool]],
    photo_ids_by_room: Union[Dict[str, List[PhotoId]],Dict[int, List[PhotoId]]],
    photo_id_to_caption: Union[Dict[int, Dict],Dict[str, Dict]],
    num_negatives: int,
    shuffler: Callable,
    dataset_type: str,
    traj_judge: bool,
    negative_style: str
):
    path_len = len(positive_path)
    # The default number of negative samples is 2
    # Get the index of caption and captionless
    captioned_idx: List[int] =[]
    captionless_idx: List[int] =[]
    captionless_ids: Trajectory = []
    for i, sample in enumerate(positive_path):
        if states[i]:
            captioned_idx.append(i)
        else:
            captionless_idx.append(i)
            captionless_ids.append(sample)
    
    
    normal_idx = [i for i in range(path_len)]
    candidate_num = 2

    # whether to use trajectory judge task
    if traj_judge:
        # Keep the order of caption and random the order of captionless 
        negative_captions_idx: List[List[int]] = []
        negative_captions: List[Trajectory] = []
        while(len(negative_captions) < num_negatives):
            negative_traj = [positive_path[n] for n in captioned_idx]
            order = random_fill(captioned_idx, captionless_idx, negative_traj, captionless_ids)
            if negative_traj == positive_path:
                # Avoid the positive path
                continue
            negative_captions_idx.append(order)
            negative_captions.append(negative_traj)

        # # Keep the order of caption and pick some captionless from other videos
        negative_randoms: List[Trajectory] = []
        random_captionless = []
        for i in captionless_idx:
            if dataset_type == 'ytb':
                lid, pid = random_captionless_image(room_ids, photo_ids_by_room)
            elif dataset_type == 'bnb':
                lid, pid = random_image(room_ids, photo_ids_by_room)
                found = is_captionless(pid, photo_id_to_caption) # True means captionless
                while not found:
                    lid, pid = random_image(room_ids, photo_ids_by_room)
                    found = is_captionless(pid, photo_id_to_caption)
            random_captionless.append((lid, pid))
        
        while(len(negative_randoms) < num_negatives):
            negative_traj = [positive_path[n] for n in captioned_idx]
            order = random_fill(captioned_idx, captionless_idx, negative_traj, random_captionless)
            if negative_traj == positive_path:
                # Avoid the positive path
                continue
            negative_randoms.append(negative_traj) 
        
    else:
        # Random the order of instruction
        shuffled_idx = [n for _, n in zip(range(num_negatives * candidate_num), shuffler(captioned_idx))] # 提供两倍的候选打乱数选择

        # 确保shuffled_idx的长度是num_negatives * 2
        if len(shuffled_idx) < num_negatives * candidate_num:
            shuffled_idx = (shuffled_idx*(int(num_negatives * candidate_num / len(shuffled_idx))))[:num_negatives * candidate_num]

        # Fill these shufflings to create the negative pairs
        negative_captions_idx: List[List[int]] = []
        negative_captions: List[Trajectory] = []
        for _ in range(num_negatives):
            negative = random.choice(shuffled_idx)
            shuffled_idx.remove(negative)
            negative_traj = [positive_path[n] for n in negative]
            negative_captions_idx.append(random_fill(negative, captionless_idx, negative_traj, captionless_ids))
            negative_captions.append(negative_traj)

        # Make sure to have at least 1 randomized element with a caption
        negative_randoms: List[Trajectory] = []
        num_flipped = random.randint(1, path_len - 1) # the number of out_listing
        flipped_idx = list(range(path_len))
        random.shuffle(flipped_idx)
        flipped_idx = flipped_idx[:num_flipped]

        for _ in range(num_negatives):
            path = []
            for i in range(path_len):
                if i in flipped_idx:
                    # get caption of out_listing
                    if dataset_type == 'ytb':
                        lid, pid = random_caption_image(room_ids, photo_ids_by_room)
                    elif dataset_type == 'bnb':
                        lid, pid = random_image(room_ids, photo_ids_by_room)
                        found = not is_captionless(pid, photo_id_to_caption) # True means caption
                        while not found:
                            lid, pid = random_image(room_ids, photo_ids_by_room)
                            found = not is_captionless(pid, photo_id_to_caption)
                else:
                    lid, pid = positive_path[i]
                path.append((lid, pid))
            negative_randoms.append(path)

    if negative_style == 'normal':
    # Random the order of image
        shuffled_idx = [n for _, n in zip(range(num_negatives * 2), shuffler(normal_idx))] # Provides twice the number of candidates for disruption selection
        if len(shuffled_idx) < num_negatives * candidate_num:
            # Make sure the length of shuffled_idx is num_negatives * 2
            shuffled_idx = (shuffled_idx*(int(num_negatives * candidate_num / len(shuffled_idx))))[:num_negatives * candidate_num]
        negative_images_idx: List[List[int]] = []
        negative_images: List[Trajectory] = [] 
        for _ in range(num_negatives):
            negative = random.choice(shuffled_idx)
            shuffled_idx.remove(negative)
            negative_traj = [positive_path[n] for n in negative]
            negative_images_idx.append(negative)
            negative_images.append(negative_traj)
        
    elif negative_style == 'shuffle_instruction':
        # Random the order of instruction
        shuffled_idx = [n for _, n in zip(range(num_negatives * candidate_num), shuffler(captioned_idx))] # 提供两倍的候选打乱数选择

        # 确保shuffled_idx的长度是num_negatives * 2
        if len(shuffled_idx) < num_negatives * candidate_num:
            shuffled_idx = (shuffled_idx*(int(num_negatives * candidate_num / len(shuffled_idx))))[:num_negatives * candidate_num]

        # Fill these shufflings to create the negative pairs
        negative_images_idx: List[List[int]] = []
        negative_images: List[Trajectory] = [] 
        for _ in range(num_negatives):
            negative = random.choice(shuffled_idx)
            shuffled_idx.remove(negative)
            negative_traj = [positive_path[n] for n in negative]
            negative_images_idx.append(random_fill(negative, captionless_idx, negative_traj, captionless_ids))
            negative_images.append(negative_traj)
        
    elif negative_style == 'another_path' or negative_style == 'another_destination':
        # another correct path
        negative_images_idx: List[List[int]] = []
        negative_images: List[Trajectory] = [] 
        while len(negative_images) < num_negatives:
            negative_traj = [positive_path[0]]
            idx = []
            temp = []
            for i in photo_ids_by_room[positive_path[0][0]]:
                idx.append(i['frame_id'])
                temp.append((positive_path[0][0], i['merging']))
            current = idx.index(positive_path[0][1][0])

            if len(idx) - current < path_len:
                # 轨迹长度不够时
                for _ in range(num_negatives):
                    negative_images_idx.append(normal_idx)
                    negative_images.append(sorted(random.sample(temp, path_len),key=lambda x:x[1][0]))
                # raise NotImplementedError()

            negative_traj += sorted(random.sample(temp[current + 1:], path_len-1),key=lambda x:x[1][0])
            if negative_traj != positive_path and (negative_style == 'another_path' or positive_path[-1] != negative_traj[-1]):
                negative_images_idx.append(normal_idx)
                negative_images.append(negative_traj)
    else:
        raise NotImplementedError()
    
    order_labels = {
        "normal_idx": normal_idx,
        "negative_captions_idx": negative_captions_idx,
        "negative_images_idx": negative_images_idx
    }

    return negative_captions, negative_images, negative_randoms, order_labels

def shuffle_different(seq: List[T]) -> Iterator[List[T]]:
    sequences = list(itertools.permutations(seq, len(seq)))
    random.shuffle(sequences)
    for s in sequences:
        l = list(s)
        if l != seq:
            yield l

def shuffle_two(seq: List[T]) -> Iterator[List[T]]:
    n = len(seq)
    ij = list(itertools.permutations(range(n), 2))
    random.shuffle(ij)
    for i, j in ij:
        seq2 = copy.deepcopy(seq)
        seq2[i], seq2[j] = seq2[j], seq2[i]
        yield seq2

def shuffle_non_adjacent(seq: List[T]) -> Iterator[List[T]]:
    n = len(seq)
    starting = {i: [j for j in range(n) if abs(j - i) > 1] for i in range(n)}
    keys = list(starting.keys())
    done = []
    while keys != []:
        idx_keys, start = random.choice(list(enumerate(keys)))
        idx_list, permute = random.choice(list(enumerate(starting[start])))

        del starting[start][idx_list]
        if starting[start] == []:
            del keys[idx_keys]

        if {start, permute} in done:
            continue
        done.append({start, permute})

        shuffled = copy.deepcopy(seq)
        shuffled[start], shuffled[permute] = shuffled[permute], shuffled[start]

        yield shuffled

def load_shuffler(shuffler):
    if shuffler == "different":
        return shuffle_different
    elif shuffler == "nonadj":
        return shuffle_non_adjacent
    elif shuffler == "two":
        return shuffle_two
    raise ValueError(f"Unexpected shuffling mode ({shuffler})")

def _check_enough_images(photo_ids_by_room, min_length):
    key_to_del = []
    for room_id, photo_ids in photo_ids_by_room.items():
        if len(photo_ids) < min_length:
            key_to_del.append(room_id)
    
    for l_id in key_to_del:
        del photo_ids_by_room[l_id]
            # raise ValueError(f"Not enough images for listing {video_id}")
    return photo_ids_by_room

def load_trajectories(testset_path: Union[Path, str], dataset_type: str):
    testset = load_json_data(testset_path)
    # we convert keys into int
    return {int(key) if dataset_type=='bnb' else key: seq for key, seq in testset.items()}
##########################################################
# The following functions are serviced for YouTube dataset
##########################################################
def _ytb_load_skeletons(skeleton_path: Union[Path, str], tokenizer, max_instruction_length: int ) -> Optional[Dict[int, List[Dict]]]:

    skeletons = ytb_load_tokens(skeleton_path, tokenizer)
    filter_skeletons = []
    for ins in skeletons:
        if len(ins['instruction_tokens'][0]) <= 60:
            filter_skeletons.append(ins)
    return filter_skeletons

def sort_skeletons(skeleton):
    # record the nums of the mask and omask for each template
    temps_num = [] # num of mask_omask
    temps_indexs = [[] for i in range(200)] # indexs of tempaltes for each num of mask_omask
    max_mask_num = 0
    max_omask_num = 0
    for ind, temp in enumerate(skeleton):
        instr = temp['instructions'][0]
        index = ind
        instr = re.sub('([.,!?:()])', r' \1', instr)
        words = instr.split(' ')
        mask_num = words.count('[MASK]')
        omask_num = words.count('[OMASK]')
        t_n = str(mask_num)+'_'+str(omask_num)
        if mask_num>max_mask_num:
            max_mask_num = mask_num
        if omask_num>max_omask_num:
            max_omask_num = omask_num
        if t_n not in temps_num:
            temps_num.append(t_n)
        temps_indexs[temps_num.index(t_n)].append(index)
    return temps_indexs, temps_num

def ytb_get_caption(frame_key: str, key_id_to_caption: Dict[str, Dict]) -> Tuple[str, str]:
    frame = key_id_to_caption[frame_key]
    return (frame["caption"], frame["action"])

def ytb_get_key(video_id: str, frame_id: int):
    return f"{video_id}/%04d"%frame_id

    
def ytb_generate_trajectory_from_listing(
    video_id: str,
    video_ids: List[str],
    _frame_ids_by_video: Dict[str, List[PhotoId]],
    key_id_to_caption: Dict[str, Dict],
    key_trajectory: List[int],
    min_length: int = 4,
    max_length: int = 7,
    min_captioned: int = 2,
    max_captioned: int = 7,
    ) -> Trajectory:
    """
    This function is set aside in order to be used by ytb_dataset/scripts/generate_frame_ids
    """
    # Gather all candidates from the same video_id
    frame_ids = merge_frames(_frame_ids_by_video[video_id])
    frame_ids = sorted(frame_ids,key=lambda x:x[0])
    all_candidates = [frame_id[0] for frame_id in frame_ids]
    frame_to_merge = {frame_id[0]:frame_id for frame_id in frame_ids}

    if len(key_trajectory) < min_captioned:
        raise ValueError(f"Not enough captioned frames for listing {video_id}")

    if len(all_candidates) < min_length:
        raise ValueError(f"Not enough images for listing {video_id}")

    if (all_candidates.index(key_trajectory[-1]) - all_candidates.index(key_trajectory[0])) < min_length:
        # raise ValueError(f"Not enough frames for listing {video_id}")
        # When the number of frames in the middle of a keyframe is too low, use the frames outside the keyframe

        # 
        temp = all_candidates
        candidates = all_candidates[all_candidates.index(key_trajectory[0]):all_candidates.index(key_trajectory[-1])+1]

        while(len(candidates) < min_length):
            # Avoid the same frames
            x =  random.choice(temp)
            if x in candidates:
                temp.remove(x)
                continue
            candidates.append(x)
        candidates.sort()
        candidates = [(video_id, frame_to_merge[i]) for i in candidates]
        states = [i[1][0] in key_trajectory for i in candidates]
        return candidates, states
        
    while(True):
        # Decide the number of key_frame
        num_key_frames = random.randint(min_captioned, min(max_captioned, len(key_trajectory)))

        # Decide the index of frames to start from
        start_key = random.randint(0, (len(key_trajectory)-num_key_frames))
        start = key_trajectory[start_key]
        end = key_trajectory[start_key + num_key_frames - 1]

        start_index = all_candidates.index(start)
        end_index = all_candidates.index(end)
        if (end_index - start_index + 1) >= min_length:
            path_len = random.randint(max(num_key_frames, min_length), min(end_index - start_index + 1, max_length))
            # if path_len > num_key_frames: # CHANGE
            #     # Guaranteed to have captionless
            #     break
            break
    
    candidates: Trajectory = []
    captionless = []
    for i in all_candidates[start_index:end_index+1]:
        if i in key_trajectory:
            # Classify frames with and without caption
            candidates.append(i)
        else:
            captionless.append(i)
    # random fill captionless to candidates
    candidates += random.sample(captionless, path_len-len(candidates))  
    candidates.sort()

    candidates = [(video_id, frame_to_merge[i]) for i in candidates]

    states = [i[1][0] in key_trajectory for i in candidates]
    return candidates, states

##########################################################
# The following functions are serviced for BnB dataset
##########################################################
def generate_trajectory_out_listing(
    listing_id: int,
    listing_ids: List[int],
    photos_by_listing: Dict[int, List[PhotoId]],
    photo_id_to_caption: Dict[int, Dict],
    min_length: int = 4,
    max_length: int = 7,
    min_captioned: int = 2,
    max_captioned: int = 7,
    ) -> Tuple[Trajectory, List[bool]]:
    """
    This function is set aside in order to be used by bnb_dataset/scripts/generate_photo_ids
    """
    # Gather all candidates
    path_len = random.randint(min_length, max_length)
    num_captioned = random.randint(min(min_captioned, path_len), min(max_captioned, path_len))
    assert num_captioned > 1
    num_captionless = path_len - num_captioned
    
    captioned: Trajectory = []
    captionless: Trajectory = []
    while len(captioned) < num_captioned or len(captionless) < num_captionless:
        listing_id, photo_id = random_image(listing_ids, photos_by_listing)
        if is_captionless(photo_id, photo_id_to_caption):
            if len(captionless) < num_captionless:
                captionless.append((listing_id, photo_id))
        else:
            if len(captioned) < num_captioned:
                captioned.append((listing_id, photo_id))

    candidates: Trajectory = captioned + captionless
    states: List[bool] = [True] * num_captioned + [False] * num_captionless

    together = list(zip(candidates, states))
    random.shuffle(together)
    candidates, states = list(zip(*together)) # type: ignore

    return candidates, states

def generate_trajectory_from_listing(
    listing_id: int,
    listing_ids: List[int],
    photos_by_listing: Dict[int, List[PhotoId]],
    photo_id_to_caption: Dict[int, Dict],
    min_length: int = 4,
    max_length: int = 7,
    min_captioned: int = 2,
    max_captioned: int = 7,
    ) -> Tuple[Trajectory, List[bool]]:
    """
    This function is set aside in order to be used by bnb_dataset/scripts/generate_photo_ids
    """
    # Gather all candidates from the same listing_id
    photo_ids = copy.deepcopy(photos_by_listing[listing_id])
    candidates: Trajectory = [(listing_id, photo_id) for photo_id in photo_ids]
    random.shuffle(candidates)

    # Decide the number of photos
    max_photos = len(candidates)
    path_len = random.randint(min_length, min(max_length, max_photos)) # random the length of a trajectory, here the length of the trajectory will change

    # Separe captioned from captionless 
    states: List[bool] = [not is_captionless(photo_id, photo_id_to_caption) for _, photo_id in candidates]
    captioned_ids, captionless_ids = [], []
    for i, caption in enumerate(states):
        if caption:
            captioned_ids.append(candidates[i])
        else:
            captionless_ids.append(candidates[i])

    # Take a certain number of captioned images, then fill with captionless photo
    # and then with captioned photos
    assert len(captioned_ids) > 1, listing_id
    max_captioned = min(max_captioned, len(captioned_ids), path_len)
    min_captioned = min(min_captioned, len(captioned_ids), path_len)
    assert max_captioned >= min_captioned, (len(captioned_ids), listing_id)
    num_captioned = random.randint(min_captioned, max_captioned)
    candidates = captioned_ids[:num_captioned]
    states = [True] * num_captioned
    candidates += captionless_ids[:path_len - num_captioned]
    states += [False] * (len(candidates) - num_captioned)
    num_captioned2 = max(0, path_len - len(candidates))
    candidates += captioned_ids[num_captioned: num_captioned2 + num_captioned]
    states += [True] * num_captioned2

    # Shuffle again
    together = list(zip(candidates, states))
    random.shuffle(together)
    candidates, states = list(zip(*together)) # type: ignore

    return candidates, states

def merge_images(captions: Iterable[Dict]) -> List[PhotoId]:
    return list({
        tuple(p["merging"]) if "merging" in p and len(p["merging"]) > 1 
        else p["photo_id"]
        for p in captions
    })

def merge_frames(captions: Iterable[Dict]) -> List[PhotoId]:
    return list({
        tuple(p["merging"]) if "merging" in p and len(p["merging"]) > 1 
        else tuple([p["frame_id"]])
        for p in captions
    })

def get_key(listing_id, photo_id):
    if isinstance(photo_id,int):
        return f"{listing_id}-{photo_id}"
    else:
        return f"{listing_id}/{f'%04d'%(photo_id['frame_id'])}"

def _check_in_lmdb(photo_ids_by_listing, keys):
    for listing_id, photo_ids in photo_ids_by_listing.items():
        for photo_id in photo_ids:
            if not isinstance(photo_id, (tuple, list)):
                photo_id = (photo_id, )
            for pid in photo_id:
                if get_key(listing_id, pid) not in keys:
                    photo_ids_by_listing[listing_id].remove(pid)
                    print(f"{pid, listing_id} is not the LMDB features")
                    # raise ValueError(f"{pid, listing_id} is not the LMDB features")
    return photo_ids_by_listing


def get_caption(photo_id: PhotoId, photo_id_to_caption: Dict[int, Dict]) -> List[int]:
    # We have a merged image. We pick a caption based on the Places365 score
    if isinstance(photo_id, (tuple, list)):
        # empty photo id (for mypy)
        if not photo_id:  
            raise ValueError("empty photo id")

        # select an image having a caption
        pid = None
        for pid in photo_id:
            if pid in photo_id_to_caption:
                break
        if pid is None:
            return []

        candidates = list(photo_id_to_caption[pid]["merging"])
        weights = list(photo_id_to_caption[pid]["weights"])

        # We consider only candidates having a caption
        for i, candidate in enumerate(candidates):
            if candidate not in photo_id_to_caption or is_captionless(candidate, photo_id_to_caption):
                weights[i] = 0

        photo_id = int(random.choices(candidates, weights=weights)[0])

    return photo_id_to_caption[photo_id]["instruction_tokens"][0]

def _load_skeletons(
        skeleton_path: Union[Path, str], 
        tokenizer,
        max_instruction_length: int 
        ) -> Optional[Dict[int, List[Dict]]]:
    skeletons_by_path = load_tokens(skeleton_path, tokenizer, max_instruction_length)
    skeletons_by_tokens = []
    for skelentons in skeletons_by_path:
        for i, _ in enumerate(skelentons['instructions']):
            skeletons_by_tokens.append({
                'distance':skelentons['distance'],
                'scan':skelentons['scan'],
                'path_id':skelentons['path_id'],
                'path':skelentons['path'],
                'heading':skelentons['heading'],
                'instructions':skelentons['instructions'][i],
                'instruction_tokens':skelentons['instruction_tokens'][i],
                'np':skelentons['np'][i],
                'perturbations':skelentons['perturbations'][i]
            })
    skeletons_by_tokens = sorted(skeletons_by_tokens, key=lambda s: sum(s["np"]))
    skeletons_by_length = {length: list(s) for length, s in groupby(skeletons_by_tokens, key=lambda s: sum(s["np"]))}
    return skeletons_by_length

##########################################################
# TInstructionGenerator
##########################################################
class InstructionGenerator:
    """
    Given a trajectory, it can generate an instruction
    """
    def __init__(self, tokenizer: BertTokenizer, separators: Tuple[str, ...], photo_id_to_caption: Union[Dict[str, Dict],Dict[int, Dict]], max_instruction_length: int):
        self._tokenizer = tokenizer
        self._cls, self._pad, self._sep = self._tokenizer.convert_tokens_to_ids(["[CLS]", "[PAD]", "[SEP]"])  # type: ignore

        if separators:
            self._separators: List[Optional[int]] = []
            seps: List[str] = list(separators)
            while None in seps:
                seps = seps.pop(seps.index(None)) # type: ignore
                self._separators.append(None)
            self._separators += self._tokenizer.convert_tokens_to_ids(seps) # type: ignore
        else:
            self._separators = [self._sep]

        self._max_instruction_length = max_instruction_length
        self._photo_id_to_caption = photo_id_to_caption

    def _remove_special_tokens(self, tokens: List[int]) -> List[int]:
        end = tokens.index(self._pad) - 1 if self._pad in tokens else len(tokens)
        while tokens[end - 1] in self._separators:
            end -= 1
            if end < 0:
                raise ValueError(f"Issue with tokens {tokens}")
        return tokens[1: end]


    def __call__(self,  trajectory: Trajectory) -> List[int]:
        raise NotImplementedError()

class RephraseInstructionGenerator(InstructionGenerator):
    """
    Fill the blanks on a R2R instruction using NP from Airbnb
    """
    def __init__(self, skeleton_path: Union[Path, str],   *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skeleton_path = skeleton_path
        self._skeletons_by_length = _load_skeletons(self._skeleton_path, self._tokenizer, self._max_instruction_length)
        

    def __call__(self, trajectory: Trajectory) -> List[int]:
        # gather captions
        captions: List[List[int]] = []
        photo_id: PhotoId
        for _, photo_id in trajectory:
            if is_captionless(photo_id, self._photo_id_to_caption):
                continue
            caption = get_caption(photo_id, self._photo_id_to_caption)
            caption = self._remove_special_tokens(caption)
            captions.append(caption)

        # pick a skeleton
        if self._skeletons_by_length is None:
            raise ValueError("Should not happen")

        # fill the skeleton
        skeleton = random.choice(self._skeletons_by_length[len(captions)])
        sentence = []
        counter = 0
        for np, split in zip(skeleton["np"], skeleton["instruction_tokens"]):
            if np:
                caption = captions[counter]
                counter += 1
            else:
                caption = [split]
            sentence += caption

        sentence = sentence[:self._max_instruction_length - 1]
        sentence += [self._sep] 
        sentence += [self._pad] * (self._max_instruction_length - len(sentence))

        return sentence

class ConcatenateInstructionGenerator(InstructionGenerator):
    """
    Contenate captions in order to create a fake instruction
    """
    @property
    def sep(self) -> List[int]:
        """ Select a separator token """
        _sep = random.choice(self._separators)
        if _sep is not None:
            return [_sep]
        return []

    def __call__(self, trajectory: Trajectory) -> List[int]:
        # gather captions
        captions: List[List[int]] = []
        photo_id: PhotoId
        for _, photo_id in trajectory:
            if is_captionless(photo_id, self._photo_id_to_caption):
                continue
            caption = get_caption(photo_id, self._photo_id_to_caption)
            caption = self._remove_special_tokens(caption)
            captions.append(caption)
        
        # shorten some captions
        credit = self._max_instruction_length
        credit -= 1 # CLS token
        credit -= len(captions) # connector
        quota = credit // len(captions)
        exceeding_ids = []
        exceeding_lengths = []
        for idx, caption in enumerate(captions):
            num_tokens = len(caption)
            if num_tokens > quota:
                exceeding_ids.append(idx)
                exceeding_lengths.append(num_tokens)
            else:
                credit -= num_tokens

        if exceeding_ids != []:
            exceeding_lengths, exceeding_ids = list(zip(*sorted(zip(exceeding_lengths, exceeding_ids)))) # type: ignore
            for i, idx in enumerate(exceeding_ids):
                num_tokens = credit // len(exceeding_ids[i:])
                captions[idx] = captions[idx][:num_tokens]
                credit -= len(captions[idx])
                assert credit >= 0

        # concatenate with separators
        merge: List[int] = [self._cls]
        for i, caption in enumerate(captions):
            merge += caption
            if i < len(captions) - 1:
                merge += self.sep
        merge += [self._sep]
        
        # pad sentence
        merge += [self._pad] * (self._max_instruction_length - len(merge))
        
        return merge

class YTBRephraseInstructionGenerator(InstructionGenerator):
    """
    Fill the blanks on a R2R instruction using NP and VP from YouTube
    """
    def __init__(self, skeleton_path: Union[Path, str], random_action: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skeleton_path = skeleton_path
        self._random_action = random_action
        self._skeletons_by_length = _ytb_load_skeletons(self._skeleton_path, self._tokenizer, self._max_instruction_length)
        self.temps_indexs, self.temps_num = sort_skeletons(self._skeletons_by_length)
        self.generatived_sentence = ""

    def __call__(self, trajectory: Trajectory, templete = None) -> Tuple[str, str]:
        # gather captions
        captions: List[str] = []
        actions: List[str] = []
        for vid, fid in trajectory:
            caption, action = ytb_get_caption(ytb_get_key(vid,fid[0]), self._photo_id_to_caption)
            if caption =="":
                # just use caption frame to generate instruction
                continue
            if self._random_action:
                # Pick one action at random 
                action = random.choice(["forward", "left", "right"])
            captions.append(caption)
            actions.append([action])

        # We only need the caption of last frame
        actions = actions[:-1]

        # Insert captions and actions from YouTube
        sentence, templete = run_insertion(captions, actions, self._skeletons_by_length, self.temps_indexs, self.temps_num, templete)
        self.generatived_sentence = sentence
        sentence = self._tokenizer.tokenize(sentence)
        # add a classification and seperator tokens
        sentence = [self._cls] + [
                self._tokenizer.vocab[token] for token in sentence  # type: ignore
            ]
        # fill the skeleton
        sentence = sentence[:self._max_instruction_length - 1]
        sentence += [self._sep] 
        sentence += [self._pad] * (self._max_instruction_length - len(sentence))

        return (sentence, templete)