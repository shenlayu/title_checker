
import re
import os
import json
import shutil
from utils import time_logger

from typing import Optional, Dict
from tqdm import tqdm
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

from .llm_utils import LLM

import logging
logger = logging.getLogger(__name__)

def _font_size_of(node) -> float:
    """ 提取单个 <text> 的字体大小 """
    style = node.attrib.get("style", "")
    m = re.search(r"font-size\s*:\s*([\d.]+)", style)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0

def extract_chart_title(svg_path: str) -> Optional[str]:
    """ 从单个 SVG 文件中提取 chart 标题。 """
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_string = f.read()
    except:
        return None

    ns = {"svg": "http://www.w3.org/2000/svg"}
    try:
        root = ET.fromstring(svg_string)
    except ET.ParseError:
        return None

    # 找到标题对应的 class
    g = root.find(".//svg:g[@class='title']", ns)
    if g is None:
        return None

    # 找到标题对应 class 中所有 <text>
    texts = g.findall(".//svg:text", ns)
    if not texts:
        return ""

    sizes = [_font_size_of(t) for t in texts]
    max_size = max(sizes) if sizes else 0.0

    parts = []
    for t, s in zip(texts, sizes):
        if s == max_size:
            content = (t.text or "").strip()
            if content:
                parts.append(content)

    return " ".join(parts)

@time_logger()
def generate_basic_data_dict(root_input_dir: str, basic_data_path: str,
                             png_name: str = "chart.png", 
                             svg_name: str = "chart.svg", 
                             raw_data_name: str = "data.json"):
    """ 对于根文件夹中的每个子文件夹，生成基础数据，保存到目标位置 """
    if logger is not None:
        logger.info("开始生成基础数据")

    if os.path.exists(basic_data_path) and os.path.getsize(basic_data_path) > 0:
        logger.warning(f"基础数据文件 {basic_data_path} 已存在且非空，跳过生成基础数据")
        return
    # TODO 改为断点续跑

    subfolders = [d for d in os.listdir(root_input_dir) 
                  if os.path.isdir(os.path.join(root_input_dir, d))]

    with open(basic_data_path, "w", encoding="utf-8") as f_out:
        for subfolder in tqdm(subfolders, desc="生成基础数据"):
            subfolder_path = os.path.join(root_input_dir, subfolder)
            try:
                data_dict = _generate_single_basic_data_dict(subfolder_path, png_name, svg_name, raw_data_name)
                if data_dict:
                    f_out.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"处理 {subfolder_path} 时出错: {e}")

def _generate_single_basic_data_dict(input_dir: str, png_name: str, svg_name: str, raw_data_name: str) -> Dict:
    """
    对于单个子文件夹对应的图表，提取基础数据
    返回格式：
        {
            "image_path": str,
            "title": str,
            "raw_data_path": str
        }
    """
    image_path = os.path.abspath(os.path.join(input_dir, png_name))

    raw_data_path = os.path.abspath(os.path.join(input_dir, raw_data_name))
    
    svg_path = os.path.abspath(os.path.join(input_dir, svg_name))
    title = extract_chart_title(svg_path)
    if title is None:
        return None

    basic_data_dict = {
        "image_path": image_path,
        "title": title,
        "raw_data_path": raw_data_path
    }

    return basic_data_dict

def generate_final_dict(basic_data: str, final_data_path: str, llm_kwargs: Dict, max_workers: int=4):
    """ 根据基础数据，调用大模型添加正例、负例，生成最 """
    max_workers = min(max_workers, os.cpu_count() or 1)
    logger.info(f"开始生成最终数据，使用 {max_workers} 个进程")

    # 如果文件存在，先复制到一个 tmp 文件中，而后在扫完文件后先清空之，最后再删掉 tmp 文件
    tmp_path = final_data_path + ".tmp"

    # 以 image_path 为 key, 记录已经生成了的正负例
    generated_pos_neg = {}
    if os.path.exists(final_data_path) and os.path.getsize(final_data_path):
        shutil.copy(final_data_path, tmp_path)
        with open(final_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    image_path = obj["image_path"]
                    pos_texts = obj["pos_texts"]
                    neg_texts = obj["neg_texts"]

                    generated_pos_neg[image_path] = (pos_texts, neg_texts)
                except:
                    continue

    basic_records = []
    with open(basic_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            basic_record = json.loads(line)
            basic_record["generated_pos"], basic_record["generated_neg"] = \
                generated_pos_neg.get(basic_record["image_path"], ([], []))
            basic_records.append(basic_record)
    
    open(final_data_path, "w", encoding="utf-8").close()
    with open(final_data_path, "a", encoding="utf-8") as f:
        with ProcessPoolExecutor(max_workers=max_workers) as excutor:
            futures = [excutor.submit(_generate_final_dict_worker, record, llm_kwargs) for record in basic_records]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="生成正负例"):
                try:
                    result = fut.result()
                    if result:
                        line = json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    logger.exception(f"子任务失败：{e}")
                    continue

                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

def _generate_final_dict_worker(record: Dict, llm_kwargs: Dict):
    """
    生成单个最终数据
    返回格式：
        {
            "image_path": str,
            "pos_texts": List,
            "neg_texts": List
        }
    """
    llm = LLM(**llm_kwargs)

    with open(record["raw_data_path"], "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except Exception as e:
            logger.warning(f"raw data 解析错误：{e}")
            return None

    title = record["title"]

    generated_pos = record.get("generated_pos", [])
    if title in generated_pos:
        generated_pos_num = len(generated_pos) - 1
    else:
        generated_pos_num = len(generated_pos)
    generate_pos_num = llm.num_pos - generated_pos_num
    generated_neg = record.get("generated_neg", [])
    generated_neg_num = len(generated_neg)
    generate_neg_num = llm.num_neg - generated_neg_num

    if generate_pos_num > 0:
        pos_texts = llm.generate_pos(raw_data, title, generate_pos_num)
    else:
        pos_texts = []
    pos_texts.extend(generated_pos)

    if generate_neg_num > 0:
        neg_texts = llm.generate_neg(raw_data, title, generate_neg_num)
    else:
        neg_texts = []
    neg_texts.extend(generated_neg)

    if title not in pos_texts:
        pos_texts.append(title)
    
    final_dict = {
        "image_path": record["image_path"],
        "pos_texts": pos_texts,
        "neg_texts": neg_texts
    }

    return final_dict