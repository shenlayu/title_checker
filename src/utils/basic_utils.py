import os
import time

import pytz
import hydra
import logging
import wandb

from contextlib import ContextDecorator
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from functools import wraps
from datetime import datetime
from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict

UNIMPORTANT_CFG = EasyDict(
    fields=['gpus', 'debug', 'wandb', 'env', 'uid',
            'local_rank', 'cmd', 'file_prefix'],
    prefix=['_'],
    postfix=['_path', '_file', '_dir']
)
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]

install()
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=False, tracebacks_suppress=[hydra],
        console=Console(width=165),
        enable_link_path=False
    )],
)
logger = logging.getLogger(__name__)
logger.info("Rich Logger initialized.")


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'

def mkdir_p(path, enable_log=True):
    """ 带输出地创建文件夹 """
    import errno
    if os.path.exists(path): return
    try:
        os.makedirs(path)
        if enable_log:
            logger.info('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and logger:
            logger.info('Directory {} already exists.'.format(path))
        else:
            raise

def init_path(dir_or_file_list):
    """ 初始化路径，输入路径或路径列表 """
    if isinstance(dir_or_file_list, list):
        return [_init_path(_) for _ in dir_or_file_list]
    else:  # single file
        return _init_path(dir_or_file_list)

def _init_path(dir_or_file):
    """ 初始化单个路径 """
    if dir_or_file.startswith('~'):
        dir_or_file = os.path.expanduser(dir_or_file)
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file.replace('//', '/')


def time2str(t):
    """ 将时间转化为字符串 """
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)

def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    """ 获取当前时间 """
    return datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)

class time_logger(ContextDecorator):
    """ 作为修饰器输出进入、退出函数时间 """
    def __init__(self, name=None, log_func=logger.info):
        self.name = name
        self.log_func = log_func

    def __enter__(self):
        self.start_time = time.time()
        self.log_func(f'Started {self.name} at {get_cur_time()}')
        return self

    def __exit__(self, *exc):
        self.log_func(f'Finished {self.name} at {get_cur_time()}, running time = '
                      f'{time2str(time.time() - self.start_time)}.')
        return False

    def __call__(self, func):
        self.name = self.name or func.__name__
        self.start_time = None

        @wraps(func)
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator
    
def append_header_to_file(file_path, header):
    with open(file_path, 'r') as input_file:
        original_content = input_file.read()

    with open(file_path, 'w') as output_file:
        output_file.write(header + original_content)

def save_cfg(cfg: DictConfig, path, as_global=True):
    """ 保存配置文件 """
    processed_cfg = get_important_cfg(cfg)
    OmegaConf.save(config=DictConfig(processed_cfg), f=path)
    if as_global:
        append_header_to_file(path, header='# @package _global_\n')
    return cfg
    
def subset_dict_by_condition(d, is_preserve=lambda x: True):
    """ 根据条件决定保留字典哪些字段 """
    if isinstance(d, dict):
        d = {k: v for k, v in d.items() if is_preserve(k)}
        for key in d.keys():
            if isinstance(d[key], dict) and is_preserve(key):
                d[key] = subset_dict_by_condition(d[key], is_preserve)
    return d

def get_important_cfg(cfg: DictConfig, reserve_file_cfg=True, unimportant_cfg=UNIMPORTANT_CFG):
    """ 筛选值得记录的 configs """
    uimp_cfg = cfg.get('_unimportant_cfg', unimportant_cfg)
    imp_cfg = OmegaConf.to_object(cfg)

    def is_preserve(k: str):
        judge_file_setting = k == '_file_' and reserve_file_cfg
        prefix_allowed = (not any([k.startswith(_) for _ in uimp_cfg.prefix])) or judge_file_setting
        postfix_allowed = not any([k.endswith(_) for _ in uimp_cfg.postfix])
        field_allowed = k not in uimp_cfg.fields
        return prefix_allowed and postfix_allowed and field_allowed

    imp_cfg = subset_dict_by_condition(imp_cfg, is_preserve)
    return imp_cfg

def print_important_cfg(cfg, log_func=logger.info):
    log_func(OmegaConf.to_yaml(get_important_cfg(cfg, reserve_file_cfg=False)))

def init_env_variables(cfg=None, env_cfg_file=f'{root_path}configs/user/env.yaml'):
    """ 赋值环境变量相关配置 """
    if cfg is None and os.path.exists(env_cfg_file):
        cfg = OmegaConf.load(env_cfg_file)
        if 'env' in cfg and 'vars' in cfg.env:
            for k, v in cfg.env.vars.items():
                k = k.upper()
                os.environ[k] = v
            if (conda_path := os.environ.get('CONDA_EXE')) is not None:
                conda_bin_dir = conda_path.rstrip('conda')
                os.environ['PATH'] = f"{conda_bin_dir}:{os.environ['PATH']}"

    return cfg

NonPercentageFloatMetrics = ['loss', 'time']

def judge_by_partial_match(k, match_dict, case_sensitive=False):
    k = k if case_sensitive else k.lower()
    return len([m for m in match_dict if m in k]) > 0

def metric_processing(log_dict):
    for k, v in log_dict.items():
        if isinstance(v, float):
            is_percentage = not judge_by_partial_match(k, NonPercentageFloatMetrics)
            if is_percentage:
                log_dict[k] *= 100
            log_dict[k] = round(log_dict[k], 4)
    return log_dict

class WandbExpLogger:
    def __init__(self, cfg):
        self.wandb = cfg.wandb
        self.wandb_on = cfg.wandb.id is not None
        self.local_rank = cfg.local_rank
        self.logger = logger
        self.logger.setLevel(getattr(logging, cfg.logging.level.upper()))
        self.info = self.logger.info
        self.critical = self.logger.critical
        self.warning = self.logger.warning
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.error = self.logger.error
        self.log_metric_to_stdout = (not self.wandb_on and cfg.local_rank <= 0) or \
                                    cfg.logging.log_wandb_metric_to_stdout
        self.results = defaultdict(list)

    def log(self, *args, level='', **kwargs):
        if self.local_rank <= 0:
            self.logger.log(getattr(logging, level.upper()), *args, **kwargs)

    def log_fig(self, fig_name, fig_file):
        if wandb.run is not None and self.local_rank <= 0:
            wandb.log({fig_name: wandb.Image(fig_file)})
        else:
            self.error('Figure not logged to Wandb since Wandb is off.', 'ERROR')

    def wandb_metric_log(self, metric_dict, level='info'):
        metric_dict = metric_processing(metric_dict)
        for metric, value in metric_dict.items():
            self.results[metric].append(value)

        if wandb.run is not None and self.local_rank <= 0:
            wandb.log(metric_dict)
        if self.log_metric_to_stdout:
            self.log(metric_dict, level=level)

    def lookup_metric_checkpoint_by_best_eval(self, eval_metric, out_metrics=None):
        if len(self.results[eval_metric]) == 0:
            return {}
        best_val_ind = self.results[eval_metric].index(max(self.results[eval_metric]))
        out_metrics = out_metrics or self.results.keys()
        return {m: self.results[m][best_val_ind] for m in out_metrics}

    def wandb_summary_update(self, result):
        if wandb.run is not None and self.local_rank <= 0:
            wandb.summary.update(result)

    def save_file_to_wandb(self, file, base_path, policy='now', **kwargs):
        if wandb.run is not None and self.local_rank <= 0:
            wandb.save(file, base_path=base_path, policy=policy, **kwargs)