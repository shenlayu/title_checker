import os
import torch
import wandb
import numpy as np

from omegaconf import OmegaConf
from torch import distributed as dist
from uuid import uuid4
from datetime import datetime

from utils.distributed_utils import get_rank, get_world_size, init_process_group
from utils.basic_utils import init_path

proj_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]

def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_unique_id(cfg):
    """ 为进程生成专属 UID """
    # if cfg.get('uid') is not None and cfg.wandb.id is not None:
    #     assert cfg.get('uid') == cfg.wandb.id, 'Confliction: Wandb and uid mismatch!'
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    # given_uid = cfg.wandb.id or cfg.get('uid')
    given_uid = cfg.get('uid')
    uid = given_uid if given_uid else cur_time + str(uuid4()).split('-')[0]
    return uid

def init_experiment(cfg):
    OmegaConf.set_struct(cfg, False) # 允许动态增加配置
    set_seed(cfg.seed)

    cfg.uid = generate_unique_id(cfg)

    if cfg.task_type == "train":
        # 分布式初始化
        world_size = get_world_size()
        if world_size > 1 and not dist.is_initialized():
            init_process_group("nccl", init_method="env://")
        cfg.local_rank = get_rank()
        # TODO wandb 初始化
        # TODO init_path
        pass
    elif cfg.task_type == "prepare_data":
        init_path(cfg.data.input_dir)
        if not cfg.skip_basic_data:
            init_path(cfg.data.basic_data)
        if not cfg.skip_final_data:
            init_path(cfg.data.final_data)

    return cfg, None


def wandb_init(cfg) -> None:
    pass