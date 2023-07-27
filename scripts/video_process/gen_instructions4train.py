import os
import sys
sys.path.append(os.getcwd())

import json
import random
import argtyped
import re
from tqdm import tqdm
from typing import Tuple, List
from pathlib import Path
import csv
from typing import List, Union, Dict, Sequence
from os import listdir

def load_tsv(filename: Union[str, Path], fieldnames: Sequence[str]) -> List[Dict]:
    with open(filename, newline="") as fid:
        reader = csv.DictReader(fid, fieldnames=fieldnames, delimiter="\t")
        return list(reader)

class Arguments(argtyped.Arguments, underscore=True):
    sample_path: Path = "data/YouTube-VLN/trajectory/"
    template: Path = "data/task/R2R_train_templates.tsv"
    caption: Path = "data/YouTube-VLN/CLIP_captioned_images/"
    direction: Path = "data/YouTube-VLN/inverses_actions/"
    output: Path = "data/YouTube-VLN/instructions_data/"
    direction_words: Tuple[List[str], ...] = (
        ["left"],
        ["right"],
        ["upstairs", "up"],
        ["downstairs", "down"],
        ["forward", "straight"],
        ["around"]
    )


def run_insertion(captions, directions, templates, temps_indexs, temps_num, template = None):

    cap_item = captions
    dir_item = directions

    m_om = str(len(cap_item)) +'_'+ str(len(dir_item))
    if m_om not in temps_num:  #donnot have matched templates
        return
    else:
        if template is None:
            instr = templates[random.choice(temps_indexs[temps_num.index(m_om)])]['instructions'][0]
            template = instr
        else:
            #use selected tempate
            instr = template
    instr = re.sub('([.,!?:()])', r' \1', instr)
    words = instr.split(' ')
    mask_num = words.count('[MASK]')
    omask_num = words.count('[OMASK]')

    mask_indexes = None
    if mask_num > 0:
        cap_words = []
        viewpoint_indexes = []
        ## FIXME
        for i, w in enumerate(cap_item):
            room, obj = w.split(' with ')
            caps = [w,room,obj]
            cap_words.append(random.sample(caps,1)[0])
            viewpoint_indexes.append(i + 1)  # for room/object/ room with object

        mask_indexes = [i for i in range(len(words)) if words[i] == '[MASK]']

        for i, index in enumerate(mask_indexes):
            words[index] = cap_words[i]

        if len(cap_item) > len(dir_item):
            mask_indexes = mask_indexes[0:len(dir_item)]
    if omask_num > 0:
        omask_indexes = [i for i in range(len(words)) if words[i] == '[OMASK]']
        i = 0
        if mask_indexes is not None:
            for index in omask_indexes:
                while i < len(mask_indexes) and mask_indexes[i] < index:
                    # Find the mask closest to omask
                    # omask is not the last empty, mask appears before omask
                    i += 1
                # i for mask last empty or mask after omask (noun after verb)
                if i < len(mask_indexes):
                    curr_view = viewpoint_indexes[i]
                else:
                    curr_view = viewpoint_indexes[i - 1]
                if curr_view - 1 >= 0:
                    dir = dir_item[curr_view - 1]
                else:
                    dir = dir_item[curr_view]
                if 'around' in dir:
                    dir_w = 'around'
                elif index - 1 >= 0 and (words[index - 1] == 'turn' or words[index - 1] == 'Turn'):
                    for w in dir:
                        if w != 'forward':
                            dir_w = w
                            break
                    else:
                        if words[index - 1] == 'turn':
                            words[index - 1] = 'go'
                        else:
                            words[index - 1] = 'Go'
                        dir_w = random.choice(dir) # forward
                else:
                    dir_w = random.choice(dir)
                words[index] = dir_w
        else:
            if omask_num < len(viewpoint_indexes):
                sample_indexes = random.sample(viewpoint_indexes, omask_num)
                sample_indexes.sort()
            else:
                omask_indexes = random.sample(omask_indexes, len(viewpoint_indexes))
                omask_indexes.sort()
                sample_indexes = viewpoint_indexes
            for i, index in enumerate(omask_indexes):
                curr_view = sample_indexes[i]
                while curr_view > len(dir_item):
                    curr_view -= 1
                if curr_view - 1 >= 0:
                    dir = dir_item[curr_view - 1]
                else:
                    dir = dir_item[curr_view]
                if 'around' in dir:
                    dir_w = 'around'
                elif index - 1 >= 0 and words[index - 1] == 'turn':
                    for w in dir:
                        if w != 'forward':
                            dir_w = w
                            break
                    else:
                        dir_w = random.choice(dir)
                else:
                    dir_w = random.choice(dir)
                words[index] = dir_w
    new_instr = ' '.join(words)

    return new_instr, template


if __name__ == "__main__":
    args = Arguments()
    fieldnames = ["instr_id", "sentence"]
    templates = load_tsv(args.template, fieldnames)    
    videos = listdir(f'{args.sample_path}')
    for vid in tqdm(videos):
        run_insertion(args, vid, templates)

