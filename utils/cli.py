import argparse


def boolean_string(s):
    if s in {"False", "0", "false"}:
        return False
    if s in {"True", "1", "true"}:
        return False
    raise ValueError("Not a valid boolean string")


def get_parser() -> argparse.ArgumentParser:
    """Return an argument parser with the standard VLN-BERT arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--traj_loss_scale",
        default=1.0,
        type=float,
        help="the weight of tp loss (default: 1.0)",
    )

    parser.add_argument(
        "--pre_dataset",
        type=str,
        choices=["","ytb", "bnb"],
        default="",
    )
    
    parser.add_argument(
        "--pretrain",
        type=bool,
        default=True,
    )
    
    parser.add_argument(
        "--mini",
        default=False,
        action="store_true",
        help="use the subset of datasets(default: False)",
    )

    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="resume model from training model",
    )

    parser.add_argument(
        "--ConstantLR",
        default=False,
        action="store_true",
        help="Constant learning rate schedule",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["vilbert", "prevalent", "oscar"],
        default="vilbert",
    )

    parser.add_argument(
        "--traj_judge",
        action='store_true',
        default=False,
        help="Determine whether the path is reasonable  (default: False)",
    )

    parser.add_argument(
        "--negative_style",
        default='normal',
        type=str,
        choices=['normal', 'shuffle_instruction', 'another_path', 'another_destination'],
        help="The style of constructing negatives (default: normal)"
    )

    parser.add_argument(
        "--no_test",
        action='store_true',
        default=False,
        help="Do not test seen and unseen during pretrain (default: False)",
    )

    parser.add_argument(
        "--no_serial",
        action='store_true',
        default=False,
        help="Shuffle correct path during training (default: False)",
    )

    # fmt: off
    # input/output data handling
    parser.add_argument(
        "--in_memory",
        default=False,
        type=boolean_string,
        help="Store the dataset in memory (default: False)",
    )
    parser.add_argument(
        "--img_feature",
        default="data/matterport-ResNet-101-faster-rcnn-genome.lmdb",
        type=str,
        help="Image features store in an LMDB file",
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        default=3,
        type=int,
        help="Number of workers per gpu (default: 3)",
    )
    parser.add_argument(
        "--output_dir",
        default="result",
        type=str,
        help="The root output directory (default: result)",
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="The name tag used for saving (default: '')",
    )
    # model configuration
    parser.add_argument(
        "--bert_tokenizer",
        default="bert-base-uncased",
        type=str,
        help="Bert tokenizer model (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--config_file",
        default="data/config/bert_base_6_layer_6_connect.json",
        type=str,
        help="Model configuration file (default: data/config/bert_base_6_layer_6_connect.json)",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Load a pretrained model (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--max_instruction_length",
        default=60,
        type=int,
        help="The maximum number of instruction tokens used by the model (default: 60)",
    )
    parser.add_argument(
        "--max_path_length",
        default=8,
        type=int,
        help="The maximum number of viewpoints tokens used by the model (default: 8)",
    )
    parser.add_argument(
        "--max_num_boxes",
        default=101,
        type=int,
        help="The maximum number of regions used from each viewpoint (default: 101)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
        help="Prefix for dataset variants (default: '')",
    )
    parser.add_argument(
        "--r2r_prefix",
        default="",
        type=str,
        help="r2r Prefix for dataset variants (default: '')",
    )
    parser.add_argument(
        "--beam_prefix",
        default="",
        type=str,
        help="Beam prefix for dataset variants (default: '')",
    )
    parser.add_argument(
        "--num_beams",
        default=30,
        type=int,
        help="The fixed number of ranked paths to use in inference (default: 30)"
    )
    parser.add_argument(
        "--num_negatives",
        default=2,
        type=int,
        help="The number of negatives per type of negatives (default: 2)"
    )
    parser.add_argument(
        "--shuffler",
        default="different",
        type=str,
        choices=["different", "nonadj", "two"],
        help="Shuffling function (default: different)",
    )
    parser.add_argument(
        "--shuffle_visual_features",
        action='store_true',
        default=False,
        help="Shuffle visual features during training (default: False)",
    )
    parser.add_argument(
        "--rank",
        default=-1,
        type=int,
        help="rank for distributed computing on gpus",
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="local_rank for distributed computing on gpus",
    )
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="Number of GPUs on which to divide work (default: -1)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="where the run code",
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="The size of one batch of training (default: 2)",
    )
    parser.add_argument(
        "--note",
        default="",
        type=str,
        help="If you want to note sth in the experiment, use this flag",
    )
    parser.add_argument(
        "--mask_action_rate",
        default=0.0,
        type=float,
        help="The masked rate of action words. (default: 0.0)"
    )

    # training configuration
    parser.add_argument(
        "--masked_vision",
        action="store_true",
        default=False,
        help="Mask image regions during training (default: False)",
    )
    parser.add_argument(
        "--masked_language",
        action="store_true",
        default=False,
        help="Mask instruction tokens during training (default: False)",
    )
    parser.add_argument(
        "--skip_val",
        action="store_true",
        default=False,
        help="Skip validation",
    )
    parser.add_argument(
        "--no_scheduler",
        action="store_true",
        default=False,
        help="Deactivate the scheduler",
    )
    parser.add_argument(
        "--ranking",
        action='store_true',
        default=False,
        help="Rank trajectories during training (default: False)",
    )
    parser.add_argument(
        "--num_epochs",
        default=20,
        type=int,
        help="Total number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--save_epochs",
        default=-1,
        type=int,
        help="THe number of epochs to save the training model (default: -1, not to save until the end)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of step before a backward pass (default: 8)",
    )
    parser.add_argument(
        "--learning_rate",
        default=4e-5,
        type=float,
        help="The initial learning rate (default: 4e-5)",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.2,
        type=float,
        help="Percentage of training to perform a linear lr warmup (default: 0.2)",
    )
    parser.add_argument(
        "--cooldown_factor",
        default=2.0,
        type=float,
        help="Multiplicative factor applied to the learning rate cooldown slope (default: 2.0)",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="The weight decay (default: 1e-2)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Train on a small subset of the dataset (default: False)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed the random number generator for training"
    )
    # Modified
    parser.add_argument(
        "--ground_truth_trajectory",
        default=False,
        type=boolean_string,
        help="Adding the ground truth trajectory in the pool of candidate (default: False)",
    )
    parser.add_argument(
        "--num_beams_train",
        default=4,
        type=int,
        help="The fixed number of ranked paths to use during training (default: 4)"
    )

    # bnb
    parser.add_argument(
        "--min_path_length",
        default=4,
        type=int,
        help="The minimum length of trajectory path (default: 4)"
    )
    parser.add_argument(
        "--min_captioned",
        default=2,
        type=int,
        help="The minimum number of captioned images (default: 2)"
    )
    parser.add_argument(
        "--max_captioned",
        default=7,
        type=int,
        help="The maximum number of captioned images (default: 7)"
    )
    parser.add_argument(
        "--combine_dataset",
        default=False,
        action="store_true",
        help="Combine a precomputed dataset with a non precompute dataset (default: False)",
    )
    parser.add_argument(
        "--out_listing",
        default=False,
        action="store_true",
        help="Using photo ids from other listings (default: False)",
    )
    parser.add_argument(
        "--separators",
        default=False,
        action="store_true",
        help="Using multiple separators when joining captions (default: False)",
    )
    parser.add_argument(
        "--bnb_feature",
        default=[
            "data/img_features/img_features_0",
            "data/img_features/img_features_1",
            "data/img_features/img_features_2",
            "data/img_features/img_features_3",
            "data/img_features/img_features_4",
            "data/img_features/img_features_5",
            "data/img_features/img_features_6",
            "data/img_features/img_features_7",
            "data/img_features/img_features_8",
            "data/img_features/img_features_9",
            "data/img_features/img_features_10",
            "data/img_features/img_features_11",
            "data/img_features/img_features_12",
            "data/img_features/img_features_13",
            "data/img_features/img_features_14",
            "data/img_features/img_features_15",
            "data/img_features/img_features_16",
            "data/img_features/img_features_17",
            "data/img_features/img_features_18",
            "data/img_features/img_features_19",
        ],
        type=str,
        nargs="+",
        help="Image features store in an LMDB file",
    )

    # speaker
    parser.add_argument(
        "--dataset",
        default="r2r",
        type=str,
        help="Type of dataset",
    )
    parser.add_argument(
        "--np",
        default=False,
        action="store_true",
        help="Add noun phrases before tokens",
    )
    parser.add_argument(
        "--window",
        default=20,
        type=int,
        help="Length for splitting a sentence",
    )

    # ytb
    parser.add_argument(
        "--skeleton_path",
        default="data/task/R2R_train_templates.json",
        type=str,
        help="Skeleton path for generating better captions (concatenation if '')"
    )
    parser.add_argument(
        "--ytb_feature",
        default=[
            "data/YouTube-VLN/youtube_img_features/img_features_0",
            "data/YouTube-VLN/youtube_img_features/img_features_1",
            "data/YouTube-VLN/youtube_img_features/img_features_2",
            "data/YouTube-VLN/youtube_img_features/img_features_3",
            "data/YouTube-VLN/youtube_img_features/img_features_4",
            "data/YouTube-VLN/youtube_img_features/img_features_5",
            "data/YouTube-VLN/youtube_img_features/img_features_6",
            "data/YouTube-VLN/youtube_img_features/img_features_7",
            "data/YouTube-VLN/youtube_img_features/img_features_8",
            "data/YouTube-VLN/youtube_img_features/img_features_9",
            "data/YouTube-VLN/youtube_img_features/img_features_10",
        ],
        type=str,
        nargs="+",
        help="Image features store in an LMDB file",
    )
    parser.add_argument(
        "--random_action",
        default=False,
        action="store_true",
        help="Random fill actions when generating instructions (default: False)",
    )
    parser.add_argument(
        "--skip_all_reduce",
        default=False,
        action="store_true",
        help="whether skip the all_reduce ops (default: False)",
    )
    parser.add_argument(
        "--not_traj_judge_data",
        default=False,
        action="store_true",
        help="whether not to use traj_judge data (default: False)",
    )
    parser.add_argument(
        "--feather_note",
        default="",
        type=str,
        help=""
    )

    return parser
