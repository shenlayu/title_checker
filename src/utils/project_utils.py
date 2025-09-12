import os
import torch
import wandb
import numpy as np

from omegaconf import OmegaConf
from torch import distributed as dist
from uuid import uuid4
from datetime import datetime

from utils.distributed_utils import get_rank, get_world_size, init_process_group
from utils.basic_utils import init_path, get_important_cfg, init_env_variables, save_cfg, \
    print_important_cfg, WandbExpLogger
import logging

logger = logging.getLogger(__name__)

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
    if cfg.get('uid') is not None and cfg.wandb.id is not None:
        assert cfg.get('uid') == cfg.wandb.id, 'Confliction: Wandb and uid mismatch!'
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    given_uid = cfg.wandb.id or cfg.get('uid')
    given_uid = cfg.get('uid')
    uid = given_uid if given_uid else cur_time + str(uuid4()).split('-')[0]
    return uid

def init_experiment(cfg):
    OmegaConf.set_struct(cfg, False) # 允许动态增加配置
    cfg = init_env_variables(cfg)
    set_seed(cfg.seed)

    cfg.uid = generate_unique_id(cfg)

    if cfg.task_type == "train":
        wandb_init(cfg)
        # 分布式初始化
        world_size = get_world_size()
        if world_size > 1 and not dist.is_initialized():
            init_process_group("nccl", init_method="env://")
        cfg.local_rank = get_rank()

        init_path(cfg.output_dir)
        cfg_out_file = cfg.output_dir + 'hydra_cfg.yaml'
        save_cfg(cfg, cfg_out_file, as_global=True)

        _logger = WandbExpLogger(cfg)
        _logger.save_file_to_wandb(cfg_out_file, base_path=cfg.output_dir, policy='now')
        _logger.info(f'Local_rank={cfg.local_rank}')
        print_important_cfg(cfg, _logger.debug)

        return cfg, _logger
        
    elif cfg.task_type == "prepare_data":
        init_path(cfg.data.input_dir)
        if not cfg.skip_basic_data:
            init_path(cfg.data.basic_data)
        if not cfg.skip_final_data:
            init_path(cfg.data.final_data)

        cfg_out_file = cfg.output_dir + 'hydra_cfg.yaml'
        save_cfg(cfg, cfg_out_file, as_global=True)

        return cfg, None


def wandb_init(cfg) -> None:
    os.environ["WANDB_WATCH"] = "false"
    if cfg.get("wandb") and cfg["wandb"].get("use_wandb", False) and get_rank() <= 0:
        try:
            WANDB_DIR, WANDB_PROJ, WANDB_ENTITY = (
                cfg.env.vars[k.lower()] for k in ['WANDB_DIR', 'WANDB_PROJ', 'WANDB_ENTITY'])
            wandb_dir = os.path.join(proj_path, WANDB_DIR)

            if cfg.wandb.id is None:
                # 第一次运行
                init_path([wandb_dir, cfg.get('wandb_cache_dir', '')])
                wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, dir=wandb_dir,
                           reinit=True, config=get_important_cfg(cfg), name=cfg.wandb.name)
            else:  # 恢复
                logger.critical(f'Resume from previous wandb run {cfg.wandb.id}')
                wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, reinit=True,
                           resume='must', id=cfg.wandb.id)
            cfg.wandb.id, cfg.wandb.name, cfg.wandb.sweep_id = wandb.run.id, wandb.run.name, wandb.run.sweep_id
            cfg.wandb_on = True
            return
        except Exception as e:
            logger.critical(f"Wandb 初始化遇到错误: {e}\n'. Wandb 未初始化'")
    os.environ["WANDB_DISABLED"] = "true"
    cfg.wandb_on = False
    return
    