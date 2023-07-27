# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from utils.dataset.features_reader import FeaturesReader, BnBFeaturesReader, YTbFeaturesReader, PanoFeaturesReader
from utils.dataset.all_dataset import YTbDataset, BnBDataset, BeamDataset
from torch.utils.data import Subset

def load_features_reader(args) -> FeaturesReader:
    if args.pre_dataset == 'ytb':
        return YTbFeaturesReader(args.ytb_feature)
    elif args.pre_dataset == 'bnb':
        return BnBFeaturesReader(args.bnb_feature)
    elif not args.pretrain:
        return PanoFeaturesReader(args.img_feature)

def get_path(args, task_prefix) ->str:
    return f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{task_prefix}testset{args.feather_note}.json"

def get_testset_path(args) -> str:
    testset_path = {}
    if args.ranking or args.not_traj_judge_data:
        if args.negative_style == "normal":
            negative_style = ""
        else:
            negative_style = args.negative_style + "_"
        testset_path["ranking"] = get_path(args, negative_style)
    if args.traj_judge and not args.ranking:
        # when ranking and traj_judge work simultaneously, use ranking's testset
        testset_path["traj"] =  get_path(args, "traj_")
    
    return testset_path


def load_pretrian_dataset(args, tokenizer, features_reader, caption_path, testset_path, Train=True):
    separators = ("then", "and", ",", ".") if args.separators else ("[SEP]",)
    
    if Train:
        masked_vision=args.masked_vision
        masked_language=args.masked_language
    else:
        masked_vision=False
        masked_language=False
    
    if args.pre_dataset == 'ytb':
        Dataset =  YTbDataset
    elif args.pre_dataset == 'bnb':
        Dataset =  BnBDataset

    return Dataset(
        args = args,
        caption_path=caption_path,
        tokenizer=tokenizer,
        features_reader=features_reader,
        masked_vision=masked_vision,
        masked_language=masked_language,
        training=Train,
        separators=separators,
        testset_path=testset_path,
    )


def load_BeamDataset(tag, args, tokenizer, features_reader, default_gpu, Train = True):
    
    if Train:
        num_beams=args.num_beams_train
        masked_vision=args.masked_vision
        masked_language=args.masked_language
        shuffle_visual_features=args.shuffle_visual_features
    else:
        num_beams=args.num_beams
        masked_vision=False
        masked_language=False
        shuffle_visual_features=False
    
    return BeamDataset(
        args = args,
        vln_path=f"data/task/{args.r2r_prefix}R2R_{tag}.json",
        beam_path=f"data/beamsearch/{args.beam_prefix}beams_{tag}.json",
        tokenizer=tokenizer,
        features_reader=features_reader,
        num_beams=num_beams,
        num_beams_strict=False,
        training=Train,
        masked_vision=masked_vision,
        masked_language=masked_language,
        default_gpu=default_gpu,
        ground_truth_trajectory=False,
        shuffle_visual_features=shuffle_visual_features,
    )


def load_dataloader(args, default_gpu, logger, local_rank) -> str:
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

    logger.info(f"Loading features reader...")
    features_reader = load_features_reader(args)
    
    # construct datasets
    logger.info("Loading train dataset")
    if args.pretrain:
        caption_path = f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{args.pre_dataset}_train{args.feather_note}.json"
        testset_path = get_testset_path(args)

        logger.info(f"Caption path: {caption_path}")
        logger.info(f"Testset path: {testset_path}")

        # loading training datasets
        train_dataset = load_pretrian_dataset(args, tokenizer, features_reader, caption_path, testset_path)
        
        logger.info("Loading test dataset")
        test_dataset = load_pretrian_dataset(
                args,tokenizer,features_reader,
                caption_path=f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{args.pre_dataset}_test{args.feather_note}.json",
                testset_path=testset_path,
                Train=False
        )

        # in mini mode only run on a subset of the datasets
        if args.mini:
            train_dataset = Subset(
                train_dataset,
                np.random.choice(range(len(train_dataset)), size=950, replace=False),  # type: ignore
            )
            test_dataset = Subset(
                test_dataset,
                np.random.choice(range(len(test_dataset)), size=50, replace=False),  # type: ignore
            )

        if args.debug:
            test_dataset = Subset(
                test_dataset,
                np.random.choice(range(len(test_dataset)), size=4, replace=False),  # type: ignore
            )
        logger.info(f"test_dataset length:{len(test_dataset)}")
        test_sampler = SequentialSampler(test_dataset) if local_rank == -1 else DistributedSampler(test_dataset)

    else:
        vln_path = f"data/task/{args.r2r_prefix}R2R_train.json"
        beam_path = f"data/beamsearch/{args.beam_prefix}beams_train.json"
        
        logger.info(f"VLN path: {vln_path}")
        logger.info(f"Beam path: {beam_path}")
        
        # loading training datasets
        train_dataset = load_BeamDataset("train", args, tokenizer, features_reader, default_gpu)
        test_dataset = None

    if args.debug:
        train_dataset = Subset(
            train_dataset,
            np.random.choice(range(len(train_dataset)), size=8, replace=False),  # type: ignore
        )
        
    logger.info(f"train_dataset length:{len(train_dataset)}")
    logger.info("Loading val datasets")
    
    if not args.no_test and not args.pretrain:
        val_seen_dataset = load_BeamDataset("val_seen", args, tokenizer, features_reader, default_gpu, Train=False)
        val_unseen_dataset = load_BeamDataset("val_unseen", args, tokenizer, features_reader, default_gpu, Train=False)
    
        if args.pretrain:
            # only run on a subset of the datasets
            val_seen_dataset = Subset(
                val_seen_dataset,
                val_seen_dataset.get_sub_beam(), # type: ignore
            )
            val_unseen_dataset = Subset(
                val_unseen_dataset,
                val_unseen_dataset.get_sub_beam(), # type: ignore
            )
        

        if args.debug:
            val_seen_dataset = Subset(
                val_seen_dataset,
                np.random.choice(range(len(val_seen_dataset)), size=8, replace=False),  # type: ignore
            )
            val_unseen_dataset = Subset(
                val_unseen_dataset,
                np.random.choice(range(len(val_unseen_dataset)), size=4, replace=False),  # type: ignore
            )
    else:
        val_seen_dataset = {}
        val_unseen_dataset = {}
    
    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        if not args.no_test:
            val_seen_sampler = SequentialSampler(val_seen_dataset)
            val_unseen_sampler = SequentialSampler(val_unseen_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        if not args.no_test:
            val_seen_sampler = DistributedSampler(val_seen_dataset)
            val_unseen_sampler = DistributedSampler(val_unseen_dataset)
    
    logger.info(f"val_seen_dataset length {len(val_seen_dataset)}")
    logger.info(f"val_unseen_dataset length {len(val_unseen_dataset)}")

    # adjust the batch size for distributed training
    batch_size = args.batch_size // args.gradient_accumulation_steps
    if local_rank != -1:
        batch_size = batch_size // dist.get_world_size()
    logger.info(f"batch_size: {batch_size}")

    logger.info(f"Creating dataloader")
    # create data loaders
    train_data_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    ) if args.pretrain else None
    if not args.no_test:
        val_seen_data_loader = DataLoader(
            val_seen_dataset,
            sampler=val_seen_sampler,
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_unseen_data_loader = DataLoader(
            val_unseen_dataset,
            sampler=val_unseen_sampler,
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        val_seen_data_loader = {}
        val_unseen_data_loader = {}


    return train_data_loader, test_data_loader, val_seen_data_loader, val_unseen_data_loader
