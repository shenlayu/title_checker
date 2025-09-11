import os
from torch import distributed as dist

def get_rank():
    """ 获取进程在分布式进程中的 rank, 非分布式进程是返回 0 """
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0

def get_world_size():
    """ 获取分布式进程总数，非分布式时返回 1 """
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1

def init_process_group(backend, init_method=None, **kwargs):
    """ 初始化 CPU, GPU 进程组 """
    global cpu_group
    global gpu_group

    dist.init_process_group(backend, init_method, **kwargs)
    gpu_group = dist.group.WORLD
    if backend == "nccl":
        cpu_group = dist.new_group(backend="gloo")
    else:
        cpu_group = gpu_group