import warnings
warnings.filterwarnings("ignore",category = FutureWarning)
warnings.filterwarnings("ignore",category = UserWarning)

import os
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch

from utils.cli import get_parser
from utils.misc import get_output_dir, set_seed, NoneLogger, logo_print, exp_saver, get_logger
from utils.utils_init import val_args, get_time, train_epoch, save_model, delete_model, test_epoch, val_epoch
from utils.distributed import set_cuda, get_local_rank, wrap_distributed_model
from utils.dataset.dataset_init import load_dataloader
from lily import Lily, BERT_CONFIG_FACTORY
from vilbert.vilbert_init import get_optimization



def main():
    # command line parsing
    parser = get_parser()
    parser.add_argument("--final", default=False, action="store_true")
    args = parser.parse_args()

    save_folder = get_output_dir(args)

    # validate command line arguments
    val_args(args)

    # initialize
    set_seed(args)

    # get device settings
    default_gpu, _, device = set_cuda(args)

    # create output directory
    logger = NoneLogger()
    if default_gpu:
        logo_print()    
        print("Experiment Folder: ", save_folder)
        save_folder.mkdir(exist_ok=True, parents=True)
        exp_saver(save_folder, "train", "pretrain")
        model_save_path = os.path.join(save_folder.parent, "data")
        os.makedirs(model_save_path, exist_ok=True)
        logger = get_logger(__name__, save_folder / "train.log")

    # create data loaders
    local_rank = get_local_rank(args)
    train_data_loader, test_data_loader, val_seen_data_loader, val_unseen_data_loader = load_dataloader(args, default_gpu, logger, local_rank)

    # Loading model
    logger.info(f"Loading model")
    config = BERT_CONFIG_FACTORY[args.model_name].from_json_file(args.config_file)

    # save the parameters
    if default_gpu:
        with open(os.path.join(save_folder, "config.txt"), "w") as fid:
            print("args:\n{", file=fid)
            for name, value in vars(args).items():
                print(f"  '{name}': {value}", file=fid)
            print("}\n", file=fid)
            print("config:", file=fid)
            print(config, file=fid)

    config.args = args
    
    if len(args.from_pretrained) == 0:  # hack for catching --from_pretrained ""
        model = Lily(config)
    else:
        model = Lily.from_pretrained(
            args.from_pretrained, config, default_gpu=default_gpu
        )

    logger.info(f"number of parameters: {sum(p.numel() for p in model.parameters())}")

    # move/distribute model to device
    model.to(device)
    model = wrap_distributed_model(model, local_rank)

    if default_gpu:
        with open(save_folder / "model.txt", "w") as fid:
            fid.write(str(model))

    optimizer, scheduler, model, start_epoch = get_optimization(args, model, len(train_data_loader), logger)

    # SummaryWriter
    writer = SummaryWriter(
        logdir=os.path.join(save_folder.parent, "tb"), flush_secs=30
    ) if default_gpu else None

    # -------- #
    # training #
    # -------- #
    # run training
    logger.info(f"starting training from epoch {start_epoch}...")

    best_success_rate = {}
    if args.ranking:
        best_success_rate["ranking"] = 0
    if args.traj_judge:
        best_success_rate["traj"] = 0
        
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"epoch {epoch}")

        if isinstance(train_data_loader.sampler, DistributedSampler):
            train_data_loader.sampler.set_epoch(epoch)

        logger.info(f"Training: {save_folder} begin to train {epoch} \n time:{get_time()}")

        try:
            train_epoch(
                epoch,
                model,
                optimizer,
                scheduler,
                train_data_loader,
                writer,
                default_gpu,
                args,
                logger,
            )
        except:
            logger.info(f"Errors:{save_folder} train {epoch} failed\n time:{get_time()}")
            raise

        # save the model every epoch
        if default_gpu:
            if ((args.save_epochs != -1) and ((epoch + 1) % args.save_epochs == 0)) or (epoch == (args.num_epochs - 1)):
                save_model(model_save_path, epoch, logger, model, optimizer, scheduler, epoch)
                
                # Delete intermediate results and keep only the best
                delete_model(model_save_path, {epoch-args.save_epochs})
            else:
                logger.info(f"not to save the model")

        # run validation
        if not args.skip_val and (args.ranking or args.traj_judge):
            logger.info(f"running validation")
            global_step = (epoch) * len(train_data_loader)

            # run validation on the "test" split
            with torch.no_grad():
                stats_output = test_epoch(
                    epoch,
                    model,
                    "test",
                    test_data_loader,
                    writer,
                    default_gpu,
                    args,
                    global_step,
                    logger,
                )

            if default_gpu:
                for task in best_success_rate:
                    # save the model that performs the best on test dataset
                    if stats_output[task][2] >= best_success_rate[task]:
                        best_success_rate[task] = stats_output[task][2]
                        save_model(model_save_path, f"best_{task}", logger, model, optimizer, scheduler, epoch)
                    else:
                        logger.info(f"not to save the {task} model")

    # -------------- #
    # after training #
    # -------------- #
    if default_gpu:
        writer.close()
        tips = f"Finish: {save_folder} Finish~~~\n time:{get_time()}"
        logger.info(tips)


if __name__ == "__main__":
    main()
