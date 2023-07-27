import json
import logging
from typing import List
import os 
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, BertTokenizer


from lily import Lily, BERT_CONFIG_FACTORY

from utils.cli import get_parser
from utils.dataset.common import pad_packed
from utils.dataset.dataset_init import load_BeamDataset
from utils.dataset.features_reader import PanoFeaturesReader

from lily import Lily
from utils.utils_init import get_model_input, get_mask_options

from utils.misc import increment_path, get_logger, exp_saver
from datetime import datetime

def main():
    # ----- #
    # setup #
    # ----- #

    # command line parsing
    parser = get_parser()
    parser.add_argument(
        "--split",
        choices=["train", "val_seen", "val_unseen", "test"],
        required=True,
        help="Dataset split for evaluation",
    )
    # parser.add_argument("--pretrain", default=False)
    parser.add_argument("--random_testing", default=False)
    args = parser.parse_args()

    # force arguments
    args.num_beams = 1
    args.batch_size = 1
    args.gradient_accumulation_steps = 1
    args.ranking = True
    
    print(args)
            
    # create output directory
    save_folder = increment_path(os.path.join(args.output_dir, f"{args.save_name}"), increment=True, sep=f'/test_{args.split}', note=args.note).resolve()
    print(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    exp_saver(save_folder, "test", args.split)

    logger = get_logger(__name__, os.path.join(save_folder, "test.log"))

    # ------------ #
    # data loaders #
    # ------------ #

    # load a dataset
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    features_reader = PanoFeaturesReader(args.img_feature)

    vln_data = f"data/task/{args.r2r_prefix}R2R_{args.split}.json"
    print(vln_data)
    
    dataset = load_BeamDataset(args.split, args, tokenizer, features_reader, True, Train=False)


    data_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- #
    # model #
    # ----- #

    config = BERT_CONFIG_FACTORY[args.model_name].from_json_file(args.config_file)
    config.ranking = True # type: ignore
    config.masked_language = False # type: ignore
    config.masked_vision = False # type: ignore
    config.model_name = args.model_name
    config.num_negatives = args.num_negatives
    config.max_path_length = args.max_path_length
    config.max_num_boxes = args.max_num_boxes
    config.max_instruction_length = args.max_instruction_length
    config.pretrain = args.pretrain
    config.traj_judge = args.traj_judge


    with open(os.path.join(save_folder, "config.txt"), "w") as fid:
        print("args:\n{", file=fid)
        for name, value in vars(args).items():
            print(f"  '{name}': {value}", file=fid)
        print("}\n", file=fid)
        print("config:", file=fid)
        print(config, file=fid)
    
    config.args = args


    model = Lily.from_pretrained(args.from_pretrained, config, default_gpu=True)
    model.cuda()
    logger.info(f"number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ---------- #
    # evaluation #
    # ---------- #
    logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M")} begin evaluation')


    with torch.no_grad():
        all_scores = eval_epoch(model, data_loader, args)

    # save scores
    scores_path = os.path.join(save_folder, f"{args.r2r_prefix}_scores_{args.split}.json")
    json.dump(all_scores, open(scores_path, "w"))
    logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M")} saving scores: {scores_path}')

    # convert scores into results format
    all_results = convert_scores(
        all_scores=all_scores,
        beam_path=f"data/beamsearch/{args.beam_prefix}beams_{args.split}.json",
        # add_exploration_path=args.split == "test",
    )

    # save results
    results_path = os.path.join(save_folder, f"{args.r2r_prefix}_results_{args.split}.json")
    json.dump(all_results, open(results_path, "w"))
    logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M")} saving results: {results_path}')



def eval_epoch(model, data_loader, args):
    device = next(model.parameters()).device

    model.eval()
    all_scores = []
    for batch in tqdm(data_loader):
        # load batch on gpu

        instr_ids = get_instr_ids(batch)
        if args.random_testing:
            vil_logit = torch.rand(batch[0].shape).to(device)

        else:
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            # get the model output
            output = model(*get_model_input(batch))
            opt_mask = get_mask_options(batch)
            vil_logit = pad_packed(output['ranking'].squeeze(1), opt_mask)

        for instr_id, logit in zip(instr_ids, vil_logit.tolist()):
            all_scores.append((instr_id, logit))

    return all_scores


def convert_scores(all_scores, beam_path, add_exploration_path=False):
    beam_data = json.load(open(beam_path, "r"))
    instr_id_to_beams = {item["instr_id"]: item["ranked_paths"] for item in beam_data}
    instr_id_to_exploration_path = {}
    if add_exploration_path:
        instr_id_to_exploration_path = {
            item["instr_id"]: [[vp] for vp in item["exploration_path"]] for item in beam_data
        }

    output = []
    for instr_id, scores in all_scores:
        idx = np.argmax(scores)
        beams = instr_id_to_beams[instr_id]
        trajectory = []
        if add_exploration_path:
            trajectory += instr_id_to_exploration_path[instr_id]
        # perturbations -> we fake a wrong destination by stopping at the initial location
        if idx >= len(beams):
            trajectory = [beams[0][0]]
        else:
            trajectory += beams[idx]
        output.append({"instr_id": instr_id, "trajectory": trajectory})

    return output


# ------------- #
# batch parsing #
# ------------- #


def get_instr_ids(batch) -> List[str]:
    instr_ids = batch[12]
    return [str(item[0].item()) + "_" + str(item[1].item()) for item in instr_ids]


if __name__ == "__main__":
    main()
