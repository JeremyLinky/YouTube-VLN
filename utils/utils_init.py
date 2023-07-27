from datetime import datetime
from tqdm import tqdm
import torch
from typing import List, Dict, Tuple
from utils.distributed import get_local_rank
from utils.dataset.common import pad_packed
import torch.nn.functional as F
import torch.distributed as dist
import os 
from torch import nn


def val_args(args):
    multi_task = args.masked_vision + args.masked_language + args.ranking + args.traj_judge
    if not multi_task:
        raise ValueError(
            "No training objective selected, add --masked_vision, --masked_language, --ranking, or --traj_judge"
        )
    if not args.pretrain and args.traj_judge and ((args.ranking or args.not_traj_judge_data) ^ args.shuffle_visual_features): 
        raise ValueError(
            "when finetune, traj_judge task don't need extra shuffle_visual_features, remove --shuffle_visual_features or\
            do two tasks at the same time, please add --shuffle_visual_features"
        )


def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


# ------------- #
# batch parsing #
# ------------- #
# TODO batch format
def get_model_input(batch):
    (
        _,
        image_features,
        image_locations,
        image_mask,
        _,
        _,
        instr_tokens,
        instr_mask,
        _,
        instr_highlights,
        segment_ids,
        co_attention_mask,
        _,
        opt_mask,
        _,
        attend_order_visual_feature,
    ) = batch

    # remove padding samples
    image_features = image_features[opt_mask]
    image_locations = image_locations[opt_mask]
    image_mask = image_mask[opt_mask]
    instr_tokens = instr_tokens[opt_mask]
    instr_mask = instr_mask[opt_mask]
    instr_highlights = instr_highlights[opt_mask]
    segment_ids = segment_ids[opt_mask]
    # transform batch shape
    co_attention_mask = co_attention_mask.view(
        -1, co_attention_mask.size(2), co_attention_mask.size(3)
    )

    return (
        instr_tokens,
        image_features,
        image_locations,
        segment_ids,
        instr_mask,
        image_mask,
        co_attention_mask,
        instr_highlights,
        attend_order_visual_feature,
    )


def get_device(batch):
    return batch[0].device

def get_mask_options(batch) -> torch.Tensor:
    return batch[13]

def get_batch_size(batch):
    return batch[1].size(0)

def get_ranking_target(batch):
    return batch[0]
    
def get_vision_target(batch):
    opt_mask = get_mask_options(batch)
    return (
        batch[4][opt_mask].flatten(0, 1),
        batch[5][opt_mask].flatten(),
    )

def get_linguistic_target(batch):
    opt_mask = get_mask_options(batch)
    return batch[8][opt_mask].flatten()



# ----- #
# train #
# ----- #
def get_loss_correct(batch: List[torch.Tensor], outputs: Dict[str, torch.Tensor], task, args, logger, training) -> torch.Tensor:
    # [bs, num_cand]
    opt_mask = get_mask_options(batch)
    batch_size = get_batch_size(batch)
    device = opt_mask.device

    # calculate the loss and correct
    loss = torch.tensor(0, device=device)
    correct = torch.tensor(0, device=device)
    if task == "vision":
        predictions = outputs[f"{task}"]
        predictions = predictions.view(-1, predictions.shape[2])
        target, target_mask = get_vision_target(batch)
        loss = F.kl_div(
            F.log_softmax(predictions, dim=-1),
            target,
            reduction="none",
        )
        loss *= target_mask.unsqueeze(-1).float()
        numel = max(1, torch.sum(target_mask).item())
        loss = torch.sum(loss) / numel
    elif task == "language":
        voc_size = outputs[f"{task}"].shape[-1]
        predictions = outputs[f"{task}"].view(-1, voc_size)
        target = get_linguistic_target(batch)
        loss = F.cross_entropy(
            predictions, target, ignore_index=-1
        )
    elif task == 'ranking':
        target = get_ranking_target(batch)
        prediction = pad_packed(outputs[f"{task}"].squeeze(1), opt_mask)
        if training:
            loss = F.cross_entropy(prediction, target, ignore_index=-1)
            correct = torch.sum(torch.argmax(prediction, 1) == target).float()
        else:
            loss = F.binary_cross_entropy_with_logits(prediction, target.float())
            correct = torch.sum(
                target.gather(1, torch.argmax(prediction, 1).view(-1, 1))
            ).float()
    elif task == 'traj':
        prediction = pad_packed(outputs[f"{task}"].squeeze(1), opt_mask)
        if not (args.ranking or args.not_traj_judge_data):
            target = torch.zeros(prediction.shape, device=device).bool()
            target[:,0] = 1
        else:
            # when traj_judge and ranking task work together
            target = torch.zeros(prediction.shape, device=device).bool()
            if args.pretrain:
                target[:,:(1+args.num_negatives)] = 1
            else:
                target[:,:-args.num_negatives] = 1 # Only the last two paths are shuffled

        pos_weight = torch.tensor([target.shape[1]/target[0].sum() - 1], device=device) # negative / positive
        loss = F.binary_cross_entropy_with_logits(prediction, target.float(), pos_weight=pos_weight)
        correct = torch.sum((prediction.sigmoid()>0.5) == target).float()/target.shape[1]
    
    return batch_size, target, loss, correct


def compute_metrics_independent(batch: List[torch.Tensor], outputs: Dict[str, torch.Tensor], task, args, logger, reduced_metrics) -> torch.Tensor:
    device = get_device(batch)
    local_rank = get_local_rank(args)
    batch_size, target, loss, correct = get_loss_correct(batch, outputs, task, args, logger, True) 

    # calculate accumulated stats
    reduced_loss = loss.detach().float()
    reduced_correct = correct.detach().float()
    reduced_batch_size = torch.tensor(batch_size, device=device).detach().float()

    # TODO: skip this `all_reduce` to speed-up runtime
    if local_rank != -1 and not args.skip_all_reduce:
        world_size = float(dist.get_world_size())
        reduced_loss /= world_size
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_correct, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_batch_size, op=dist.ReduceOp.SUM) # type: ignore
    
    reduced_metrics["loss"][f"{task}"] = reduced_loss
    if not (task == 'vision' or task == 'language'):
        reduced_metrics["accuracy"][f"{task}"] = reduced_correct / reduced_batch_size

    return loss


def train_epoch(
    epoch, model, optimizer, scheduler, data_loader, writer, default_gpu, args, logger
) -> None:
    device = next(model.parameters()).device
    model.train()   # CHANGE
    model.zero_grad()

    for step, batch in enumerate(tqdm(data_loader, disable= not (default_gpu))):
        # load batch on gpu
        batch = tuple(
            t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t
            for t in batch
        )

        # get the model output
        outputs = model(*get_model_input(batch))
    
        # calculate the final loss
        loss = torch.tensor(0, device=device).float()
        reduced_metrics = {}
        reduced_metrics["loss"] = {}
        reduced_metrics["accuracy"] = {}
        mask_task = {}

        # calculate the sum of loss
        if args.masked_vision:
            loss += compute_metrics_independent(batch, outputs, 'vision', args, logger, reduced_metrics)
        if args.masked_language:
            loss += compute_metrics_independent(batch, outputs, 'language', args, logger, reduced_metrics)
        if args.ranking:
            loss += compute_metrics_independent(batch, outputs, 'ranking', args, logger, reduced_metrics)
        if args.traj_judge:
            loss += args.traj_loss_scale * compute_metrics_independent(batch, outputs, 'traj', args, logger, reduced_metrics)
        
        reduced_metrics["loss/train"] = torch.tensor(0, device=device).detach().float()
        for item in reduced_metrics["loss"].values():
            reduced_metrics["loss/train"] += item
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # backward pass
        loss.backward()
            
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()            
            scheduler.step()
            model.zero_grad()

        # write stats to tensorboard
        if default_gpu:
            global_step = step + epoch * len(data_loader)
            reduced_metrics["learning_rate/train"] = float(scheduler.get_last_lr()[0])
            for name in ["learning_rate/train", "loss/train"]:
                writer.add_scalar(name, reduced_metrics[name], global_step=global_step)
            for task, item in reduced_metrics["accuracy"].items():
                writer.add_scalar(f'accuracy/{task}', item, global_step=global_step)
            for task, item in reduced_metrics["loss"].items():
                writer.add_scalar(f'loss/{task}', item, global_step=global_step)
            for name, item in mask_task.items():
                    writer.add_scalar(name, item, global_step=global_step)
            
            logger.info(
                f'times: {get_time()}\t'
                f'epoch: {epoch}\t'
                f'batch: {step}\t'
                f'train loss: {reduced_metrics["loss/train"]:.2f}\t'
                f'learning rate: {reduced_metrics["learning_rate/train"]}\t'
                )
            
            tips_accuracy = '\t'.join([f'{task} accuracy: {item:.2f}' for task, item in reduced_metrics["accuracy"].items()])
            tips_loss = '\t'.join([f'{task} loss: {item:.2f}' for task, item in reduced_metrics["loss"].items()])
            tips = tips_accuracy + '\t\t' + tips_loss
            logger.info(
                '===============>: \t'
                f'{tips}'
            )


# ---------- #
# save model #
# ---------- #
def get_model_path(model_save_path, save_name):
    return os.path.join(model_save_path, f"{save_name}.bin")

def save_model(model_save_path, save_name, logger, model, optimizer, scheduler, epoch):
    model_path = get_model_path(model_save_path, save_name)
    logger.info(f"saving the {save_name} model")
    if hasattr(model, "module") and isinstance(model.module, nn.Module):
        net: nn.Module = model.module
    elif isinstance(model, nn.Module):
        net = model
    else:
        raise ValueError("Can't find the Module here")
    
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        },
        model_path,
    )

def delete_model(model_save_path, save_name):
    model_path = get_model_path(model_save_path, save_name)
    if os.path.exists(model_path):
        os.unlink(model_path)


# -------- #
# val test #
# -------- #
def val_independent(batch: List[torch.Tensor], outputs: Dict[str, torch.Tensor], task, args, logger, stats) -> torch.Tensor:
    batch_size, _, loss, correct = get_loss_correct(batch, outputs, task, args, logger, False) 

    stats[f"{task}"][0] += batch_size
    stats[f"{task}"][1] += loss
    stats[f"{task}"][2] += correct
    stats[f"{task}"][3] += 1


def test_epoch(epoch: int, model, tag, data_loader, writer, default_gpu, args, global_step, logger):
    device = next(model.parameters()).device
    local_rank = get_local_rank(args)

    # validation
    model.eval()

    stats = {}
    if args.ranking:
        stats["ranking"] = torch.zeros(4, device=device).float()
    if args.traj_judge:
        stats["traj"] = torch.zeros(4, device=device).float()

    for step, batch in enumerate(tqdm(data_loader, disable= not (default_gpu))):
        # load batch on gpu
        batch = tuple(
            t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t
            for t in batch
        )

        # get the model output
        outputs = model(*get_model_input(batch))

        if args.ranking:
            val_independent(batch, outputs, 'ranking', args, logger, stats)
        if args.traj_judge:
            val_independent(batch, outputs, 'traj', args, logger, stats)


        tips = '\t'.join([f'{task} loss: {item[1]/item[3]:.2f} \t{task} success rate: {item[2]/item[0]:.2f} \t ' for task, item in stats.items()])
        logger.info(
            '===============>: \t'
            f'{tips}'
        )

    # calculate accumulated stats
    reduced_stats = {}
    for task in stats:
        reduced_stats[task] = stats[task].detach().float()
    if local_rank != -1 and not args.skip_all_reduce:
        # average loss and accumulated correct
        world_size = float(dist.get_world_size())
        for task in reduced_stats:
            dist.all_reduce(reduced_stats[task], op=dist.ReduceOp.SUM) # type: ignore
            reduced_stats[task][1] /= world_size
    for task in reduced_stats:
        reduced_stats[task][1] /= reduced_stats[task][3]
        reduced_stats[task][2] /= reduced_stats[task][0]

    # write stats to tensorboard
    if default_gpu:
        for task, item in reduced_stats.items():
            writer.add_scalar(
                f"loss/{task}_{tag}", item[1], global_step=global_step
            )
            writer.add_scalar(
                f"accuracy/{task}_{tag}", item[2], global_step=global_step
            )
        tips = '\t'.join([f'{task} loss: {item[1]:.2f} \t{task} success rate: {item[2]:.2f} \t ' for task, item in reduced_stats.items()])
        logger.info(
            f"[{tag}] epoch: {epoch} \t"
            f'{tips}'
        )
    
    return reduced_stats


def val_epoch(epoch: int, model, tag, data_loader, writer, default_gpu, args, global_step, logger, task):
    device = next(model.parameters()).device
    local_rank = get_local_rank(args)

    # validation
    model.eval()
    stats = torch.zeros(3, device=device).float() # (bs ranking_loss ranking_accuracy) 
    for step, batch in enumerate(tqdm(data_loader, disable= not (default_gpu))):
        # load batch on gpu
        batch = tuple(
            t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t
            for t in batch
        )
        batch_size = get_batch_size(batch)

        # get the model output
        outputs = model(*get_model_input(batch))
        opt_mask = get_mask_options(batch)

        target = get_ranking_target(batch)
        logit = pad_packed(outputs[f"{task}"].squeeze(1), opt_mask)

        # calculate loss
        loss = F.binary_cross_entropy_with_logits(logit, target.float())

        # calculate success rate of the top scoring beam
        correct = torch.sum(
            target.gather(1, torch.argmax(logit, 1).view(-1, 1))
    ).float()

        stats[0] += batch_size
        
        # accumulate
        stats[1] += loss
        stats[2] += correct


        logger.info(
            f"[{tag}] step: {step} "
            f"{task} loss: {stats[1] / (step + 1):0.2f} "
            f"{task} success rate: {stats[2] / stats[0]:0.2f} "
        )
    logger.info(
        f"[{tag}] step: {step} "
        f"batch size: {stats[0]} "
        f"{task} loss: {stats[1]} "
    )

    if local_rank != -1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM) # type: ignore

    # write stats to tensorboard
    success_rate = stats[2] / stats[0]
    if default_gpu:
        writer.add_scalar(
            f"loss/{task}_{tag}", stats[1] / (step + 1), global_step=global_step
        )
        writer.add_scalar(
            f"accuracy/{task}_{tag}", success_rate, global_step=global_step
        )

    logger.info(
        f"[{task}] epoch: {epoch} success_rate: {success_rate.item():.3f}"
    )
    return success_rate






