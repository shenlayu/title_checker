import os
import time

import pytz
import hydra
import logging

from contextlib import ContextDecorator
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from functools import wraps
from datetime import datetime

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