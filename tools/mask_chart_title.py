import os
import json

import cairosvg
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count

def check_image(path: str) -> bool:
    """ 检查图片是否损坏 """
    if not isinstance(path, str) or not os.path.isfile(path):
        print(f"错误图像路径: {path}")
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            im2.load()
        return True
    except Exception:
        print(f"错误图像路径: {path}")
        return False

def generate_and_save_image(svg_str: str, save_path: str) -> None:
    """ 根据 SVG 生成图像并保存 """
    cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=save_path)

def mask_single_chart(svg_path: str) -> str:
    """ 将一个 SVG 文件的标题去除 """
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_string = f.read()

    ns = {"svg": "http://www.w3.org/2000/svg"}
    root = ET.fromstring(svg_string)

    parent_map = {c: p for p in root.iter() for c in p}

    for node in root.findall(".//svg:g[@class='title']", ns):
        parent = parent_map.get(node)
        if parent is not None:
            parent.remove(node)

    return ET.tostring(root, encoding="unicode")

def _process_one(task: Tuple[str, str, List, List]):
    try:
        input_svg_path, output_png_path, pos_texts, neg_texts = task

        updated_record = {
            "image_path": output_png_path,
            "pos_texts": pos_texts,
            "neg_texts": neg_texts,
        }

        # 如果输出位置不存在图像，或图像已损坏，则生成
        if not os.path.exists(output_png_path) or not check_image(output_png_path):
            if not os.path.exists(input_svg_path):
                return None
            svg_str = mask_single_chart(input_svg_path)
            generate_and_save_image(svg_str, output_png_path)
    except:
        return None
    return updated_record

def mask_title(input_path: str, output_jsonl_path: str, output_image_path: str, n_procs: int):
    """ 生成原始图像的 mask 标题版本，并更新 jsonl """
    input_path = os.path.abspath(input_path)
    output_jsonl_path = os.path.abspath(output_jsonl_path)
    output_image_path = os.path.abspath(output_image_path)

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)

    tasks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                line_dict = json.loads(line)
                image_path = line_dict["image_path"]
                pos_texts = line_dict["pos_texts"]
                neg_texts = line_dict["neg_texts"]

                chart_name = image_path.split("/")[-2]
                chart_name_base, _ = os.path.splitext(image_path)
                input_svg_path = chart_name_base + ".svg"
                output_png_path = os.path.join(output_image_path, chart_name + ".png")

                tasks.append((input_svg_path, output_png_path, pos_texts, neg_texts))
            except:
                continue
    
    if not tasks:
        return
    n_procs = min(n_procs, cpu_count() - 1) 
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        with Pool(processes=n_procs) as pool:
            for result in tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks), desc="并行生成中"):
                if not result:
                    continue
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_path = "./data/tmp/final_data.jsonl"
    output_jsonl_path = "./data/tmp/final_data_mask.jsonl"
    output_image_path = "./masked_images"
    n_procs = 8

    mask_title(input_path, output_jsonl_path, output_image_path, n_procs)