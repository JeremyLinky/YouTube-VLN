from vilbert.optimization import AdamW, WarmupLinearSchedule, ConstantLRSchedule
from pathlib import Path
import os 
import torch
from torch import nn

def get_optimization(args, model, train_data_loader_length, logger):
    # set parameter specific weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0},
        {"params": [], "weight_decay": args.weight_decay},
    ]
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)
        else:
            optimizer_grouped_parameters[1]["params"].append(param)

    # optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)

    if (args.pretrain and args.no_scheduler) or args.ConstantLR:
        scheduler = ConstantLRSchedule(optimizer)
    else:
        # calculate learning rate schedule
        t_total = (
            train_data_loader_length // args.gradient_accumulation_steps
        ) * args.num_epochs
        warmup_steps = args.warmup_proportion * t_total
        adjusted_t_total = warmup_steps + args.cooldown_factor * (t_total - warmup_steps)
        scheduler = (WarmupLinearSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            t_total=adjusted_t_total,
            last_epoch=-1,
            )
            if not args.no_scheduler
            else MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0) # type: ignore
        )
    
    start_epoch = 0
    # load checkpoint of the optimizer
    if args.resume:
        checkpoint_path = Path(args.from_pretrained)
        logger.info(f"resume the training model from {checkpoint_path}")
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if 'model_state_dict' in state_dict:
                if hasattr(model, "module") and isinstance(model.module, nn.Module):
                    model.module.load_state_dict(state_dict["model_state_dict"])
                elif isinstance(model, nn.Module):
                    model.load_state_dict(state_dict["model_state_dict"])
                logger.info(f"load model_state_dict...")
            if 'optimizer_state_dict' in state_dict:
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                logger.info(f"load optimizer_state_dict...")
            if 'scheduler_state_dict' in state_dict:
                scheduler.load_state_dict(state_dict["scheduler_state_dict"])
                logger.info(f"load scheduler_state_dict...")
            if "epoch" in state_dict:
                start_epoch = state_dict["epoch"] + 1
                logger.info(f"load epoch {start_epoch}...")
        else:
            logger.info(f"resumimg the training model failed, {checkpoint_path} does not exist")
        
        # Keep the learning rate from the final result 
        if args.ConstantLR:
            scheduler.base_lrs = scheduler._last_lr
            logger.info(f"Keep the learning rate {scheduler.base_lrs} as the final result")
    
    return optimizer, scheduler, model, start_epoch