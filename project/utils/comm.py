"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import functools
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger()

LOCAL_GPU_GROUP = None

GLOBAL_CPU_GROUP = None
LOCAL_CPU_GROUP = None


def get_rank(group=None):
    if group is not None:
        return dist.get_rank(group)
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.environ.get("RANK", "0"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size(group=None):
    if group is not None:
        return dist.get_world_size(group)
    elif dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_world_size():
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def get_device():
    return torch.device("cuda", get_local_rank())


def get_local_gpu_group():
    global LOCAL_GPU_GROUP
    if LOCAL_GPU_GROUP is None:
        local_world_size = get_local_world_size()
        num_machines = get_world_size() // local_world_size
        machine_rank = get_rank() // local_world_size
        for i in range(num_machines):
            ranks_on_i = list(range(i * local_world_size, (i + 1) * local_world_size))
            pg = dist.new_group(ranks_on_i, backend="nccl")
            if i == machine_rank:
                LOCAL_GPU_GROUP = pg
                logger.info(f"Created local GPU group with {local_world_size} ranks.")
    return LOCAL_GPU_GROUP


def get_cpu_group():
    world_size = get_world_size()
    global GLOBAL_CPU_GROUP
    if GLOBAL_CPU_GROUP is None:
        GLOBAL_CPU_GROUP = dist.new_group(backend="gloo")
        logger.info(f"Created global CPU group with {world_size} ranks.")
    return GLOBAL_CPU_GROUP


def get_local_cpu_group():
    global LOCAL_CPU_GROUP
    if LOCAL_CPU_GROUP is None:
        local_world_size = get_local_world_size()
        num_machines = get_world_size() // local_world_size
        machine_rank = get_rank() // local_world_size
        for i in range(num_machines):
            ranks_on_i = list(range(i * local_world_size, (i + 1) * local_world_size))
            pg = dist.new_group(ranks_on_i, backend="gloo")
            if i == machine_rank:
                LOCAL_CPU_GROUP = pg
                logger.info(f"Created local CPU group with {local_world_size} ranks.")
    return LOCAL_CPU_GROUP


def barrier(group=None):
    if group is not None:
        dist.barrier(group)
    elif dist.is_available() and dist.is_initialized():
        dist.barrier()
    else:
        return


def local_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier(get_local_gpu_group())
    else:
        return


def all_gather_object(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, data, group=get_cpu_group())
    return gather_list


def gather_object(data, dst=0):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    gather_list = [None for _ in range(world_size)] if get_rank() == dst else None
    dist.gather_object(data, gather_list, dst=dst, group=get_cpu_group())
    return gather_list


def broadcast_object(data, src=0):
    world_size = get_world_size()
    if world_size == 1:
        return data
    broadcast_list = [data] if get_rank() == src else [None]
    dist.broadcast_object_list(broadcast_list, src=src, group=get_cpu_group())
    return broadcast_list[0]


def local_broadcast_object(data, local_src=0):
    local_world_size = get_local_world_size()
    if local_world_size == 1:
        return data
    machine_rank = get_rank() // local_world_size
    src = machine_rank * local_world_size + local_src
    broadcast_list = [data] if get_local_rank() == local_src else [None]
    dist.broadcast_object_list(broadcast_list, src=src, group=get_local_gpu_group())
    return broadcast_list[0]


def main_process_first(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = get_rank()
        if rank == 0:
            result = func(*args, **kwargs)
            barrier()
            return result
        else:
            barrier()
            return func(*args, **kwargs)
    return wrapper


def main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = get_rank()
        result = None
        if rank == 0:
            logger.warning(f"Rank {rank} is running {func.__name__}.")
            result = func(*args, **kwargs)
        else:
            logger.warning(f"Rank {rank} is waiting for rank 0 to finish {func.__name__}.")
        barrier()
        result = broadcast_object(result, src=0)
        return result
    return wrapper


def local_main_process_first(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        local_rank = get_local_rank()
        if local_rank == 0:
            result = func(*args, **kwargs)
            barrier()
            return result
        else:
            barrier()
            return func(*args, **kwargs)
    return wrapper


def local_main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        local_rank = get_local_rank()
        result = None
        if local_rank == 0:
            logger.warning(f"Local rank {local_rank} is running {func.__name__}.")
            result = func(*args, **kwargs)
        else:
            logger.warning(f"Local rank {local_rank} is waiting for local rank 0 to finish {func.__name__}.")
        barrier()
        result = local_broadcast_object(result, local_src=0)
        logger.warning(f"Local rank {local_rank} finished {func.__name__}.")
        return result
    return wrapper


def all_reduce(tensor, op, **kwargs):
    if get_world_size() == 1:
        return
    return dist.all_reduce(tensor, op=op, **kwargs)


def all_gather(tensor, gather_list, **kwargs):
    if get_world_size() == 1:
        if gather_list is not None and len(gather_list) > 0:
            gather_list[0].copy_(tensor)
        return
    return dist.all_gather(tensor_list=gather_list, tensor=tensor, **kwargs)
