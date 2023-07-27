"""
Distributed tools
"""
from typing import Tuple, Callable, Union, List
from numbers import Number
import os
import math
import pickle
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import RandomSampler, SequentialSampler, Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from utils.misc import get_output_dir
import sys
logging.basicConfig(
    format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_world_size(args):
    if args.world_size != -1:
        world_size = args.world_size
    elif os.environ.get("WORLD_SIZE", "") != "":
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        raise RuntimeError()
    return world_size


def get_rank(args) -> int:
    if os.environ.get("RANK", "") != "":
        # pytorch.distributed.launch provide this variable no matter what
        rank = int(os.environ["RANK"])
        print("RANK from environ is", rank)
    elif os.environ.get("SLURM_PROCID", "") != "":
        # pytorch.distributed.launch provide this variable no matter what
        rank = int(os.environ["SLURM_PROCID"])
        print("RANK from SLURM is", rank)
    else:
        # WARNING: this assumes that each node has the same number of GPUs

        if os.environ.get("NODE_RANK", "") != "":
            node_rank = int(os.environ["NODE_RANK"])
        else:
            raise RuntimeError("Can't find any rank or node rank")

        local_rank = get_local_rank(args)

        n_gpus = torch.cuda.device_count()
        rank = local_rank + node_rank * n_gpus
        print("RANK from local rank is", rank)
        
    return rank


def load_init_param(args):
    """
    Load parameters for the rendezvous distributed procedure
    """
    # sync file
    sync_dir = get_output_dir(args)
    # sync_dir.mkdir(parents=True, exist_ok=True)
    sync_file = f"{sync_dir}/.torch_distributed_sync"

    return {
        "backend": "nccl",
        "init_method": "env://",
        # "init_method": f"file://{sync_file}",
        "rank": get_rank(args),
        "world_size": get_world_size(args),
    }


def init_distributed(args):
    init_param = load_init_param(args)
    rank = init_param["rank"]
    logger.info(f"Init distributed {init_param['rank']} - {init_param['world_size']}")
    # logger.info("before {} - {}\n".format(rank, pformat(init_param)))

    dist.init_process_group(**init_param)

    if rank == 0:
        logger.info("after {}".format(rank))


def is_main_proc(args) -> bool:
    return args.local_rank == -1 or dist.get_rank() == 0


def wrap_distributed_model(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    elif torch.cuda.device_count() > 1:
        logger.info("Using data parallel")
        model = torch.nn.DataParallel(model)

    return model


def get_local_rank(args):
    # using distributed.launcher
    if args.local_rank != -1:
        return args.local_rank

    # using SLURM launcher
    if args.world_size > 1:
        local_id = int(os.environ.get("SLURM_LOCALID", "0"))
        assert "SLURM_NTASKS" in os.environ
        return local_id

    # not a distributed
    return -1


def set_cuda(args) -> Tuple[bool, int, torch.device]:
    """
    Initialize CUDA for distributed computing
    """
    local_rank = get_local_rank(args)

    if not torch.cuda.is_available() or args.device == "cpu":
        assert local_rank == -1, local_rank
        return True, -1, torch.device("cpu")

    # get device settings
    if local_rank != -1:
        init_distributed(args)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        distributed = local_rank != -1        
        rank = dist.get_rank()
        main_proc = rank == 0
        if main_proc:
            logger.info(f"Found {dist.get_world_size()} GPUs")
    else:
        main_proc = True
        device = torch.device("cuda")
        rank = -1
        distributed = False

    if main_proc:
        logger.info(
            f"device: {device}, rank: {rank}, distributed training: {distributed}"
        )

    return main_proc, rank, device


def build_sampler(
    dataset: Dataset, is_train: bool, batch_size: int, local_rank: int
) -> Tuple[Sampler, Callable[[int], None]]:

    if local_rank == -1:
        if is_train:
            sampler: Sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        pre_epoch = lambda e: None

        # DataParallel: scale the batch size by the number of GPUs
        if size > 1:
            batch_size *= size

    else:
        size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, num_replicas=size, rank=dist.get_rank(), shuffle=is_train
        )
        pre_epoch = sampler.set_epoch

    return sampler, pre_epoch


def all_reduce_and_rescale_tensors(
    tensors: List[torch.Tensor], rescale_denom: Union[Number, torch.Tensor]
) -> None:
    """
    All-reduce and rescale tensors at once (as a flattened tensor)

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    sz = sum(t.numel() for t in tensors)
    buffer_t = tensors[0].new(sz).zero_()

    # copy tensors into buffer_t
    offset = 0
    for t in tensors:
        numel = t.numel()
        buffer_t[offset : offset + numel].copy_(t.view(-1))
        offset += numel

    # all-reduce and rescale
    dist.all_reduce(buffer_t[:offset])
    buffer_t.div_(rescale_denom)

    # copy all-reduced buffer back into tensors
    offset = 0
    for t in tensors:
        numel = t.numel()
        t.view(-1).copy_(buffer_t[offset : offset + numel])
        offset += numel


def all_reduce_and_rescale_tensors_chunked(
    tensors, rescale_denom, buffer_size=10485760
):
    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = (
        tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    )
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset : offset + numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        hvd.allreduce_(buffer_t[:offset])
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset : offset + numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            hvd.allreduce_(t)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def broadcast_tensors(tensors, root_rank, buffer_size=10485760):
    """broadcast tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to broadcast
        root_rank: rank to broadcast
        buffer_size: broadcast chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = (
        tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    )
    buffer = []

    def broadcast_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset : offset + numel].copy_(t.view(-1))
            offset += numel

        # broadcast
        hvd.broadcast_(buffer_t[:offset], root_rank)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset : offset + numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, broadcast directly
            hvd.broadcast_(t, root_rank)
        elif filled + sz > buffer_size:
            # buffer is full, broadcast and replace buffer with tensor
            broadcast_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        broadcast_buffer()


def _encode(enc, max_size, use_max_size=False):
    enc_size = len(enc)
    enc_byte = max(math.floor(math.log(max_size, 256) + 1), 1)
    if use_max_size:
        # this is used for broadcasting
        buffer_ = torch.cuda.ByteTensor(max_size + enc_byte)
    else:
        buffer_ = torch.cuda.ByteTensor(enc_size + enc_byte)
    remainder = enc_size
    for i in range(enc_byte):
        base = 256 ** (enc_byte - i - 1)
        buffer_[i] = remainder // base
        remainder %= base
    buffer_[enc_byte : enc_byte + enc_size] = torch.ByteTensor(list(enc))
    return buffer_, enc_byte


def _decode(buffer_, enc_byte):
    size = sum(256 ** (enc_byte - i - 1) * buffer_[i].item() for i in range(enc_byte))
    bytes_list = bytes(buffer_[enc_byte : enc_byte + size].tolist())
    shift = size + enc_byte
    return bytes_list, shift


_BUFFER_SIZE = 4096


def all_gather_list(data):
    """Gathers arbitrary data from all nodes into a list."""
    enc = pickle.dumps(data)

    enc_size = len(enc)
    max_size = hvd.allgather(torch.tensor([enc_size]).cuda()).max().item()
    in_buffer, enc_byte = _encode(enc, max_size)

    out_buffer = hvd.allgather(in_buffer[: enc_byte + enc_size])

    results = []
    for _ in range(hvd.size()):
        bytes_list, shift = _decode(out_buffer, enc_byte)
        out_buffer = out_buffer[shift:]
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


def any_broadcast(data, root_rank):
    """broadcast arbitrary data from root_rank to all nodes."""
    enc = pickle.dumps(data)

    max_size = hvd.allgather(torch.tensor([len(enc)]).cuda()).max().item()
    buffer_, enc_byte = _encode(enc, max_size, use_max_size=True)

    hvd.broadcast_(buffer_, root_rank)

    bytes_list, _ = _decode(buffer_, enc_byte)
    result = pickle.loads(bytes_list)
    return result
