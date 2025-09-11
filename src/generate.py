import os
import sys
import json

root_path = os.path.abspath(os.path.dirname(__file__)).split("src")[0]
os.chdir(root_path)
sys.path.append(root_path + "src")

import hydra

from utils import time_logger, init_experiment
from utils.generate_utils import generate_basic_data_dict, generate_final_dict, LLM

@time_logger()
@hydra.main(config_path=f"{root_path}/configs/prepare_data", config_name="config", version_base=None)
def generate(cfg):
    cfg, _ = init_experiment(cfg)

    data_config = cfg.data

    input_dir = data_config.input_dir
    png_name = data_config.png
    svg_name = data_config.svg
    raw_data_name = data_config.raw_data
    basic_data = data_config.basic_data
    final_data = data_config.final_data

    if not cfg.skip_basic_data:
        generate_basic_data_dict(input_dir, basic_data, png_name, svg_name, raw_data_name)

    if not cfg.skip_final_data:
        llm_config = cfg.llm

        llm_kwargs = {
            "base_url": llm_config.base_url,
            "api_key": llm_config.openai_api_key,
            "model": llm_config.model,
            "pos_system_prompt": llm_config.pos_system_prompt,
            "neg_system_prompt": llm_config.neg_system_prompt,
            "pos_prompt": llm_config.pos_prompt,
            "neg_prompt": llm_config.neg_prompt,
            "raw_data_symbol": llm_config.raw_data_symbol,
            "title_symbol": llm_config.title_symbol,
            "max_tokens": llm_config.max_tokens,
            "temperature": llm_config.temperature,
            "num_pos": llm_config.num_pos,
            "num_neg": llm_config.num_neg
        }

        max_workers = llm_config.max_workers

        generate_final_dict(basic_data, final_data, llm_kwargs, max_workers)

if __name__ == "__main__":
    generate()