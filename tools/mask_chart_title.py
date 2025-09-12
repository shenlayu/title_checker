import os
import json
import re
from typing import Optional

from tqdm import tqdm
import xml.etree.ElementTree as ET
import cairosvg

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

def mask_title(input_path: str, output_jsonl_path: str, output_image_path: str):
    input_path = os.path.abspath(input_path)
    output_jsonl_path = os.path.abspath(output_jsonl_path)
    output_image_path = os.path.abspath(output_image_path)

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for line in tqdm(lines, desc="生成中", total=len(lines)):
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

                svg_str = mask_single_chart(input_svg_path)
                generate_and_save_image(svg_str, output_png_path)

                updated_record = {
                    "image_path": output_png_path,
                    "pos_texts": pos_texts,
                    "neg_texts": neg_texts
                }
                f.write(json.dumps(updated_record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"出现错误: {e}")
                continue

if __name__ == "__main__":
    # TODO 续跑
    input_path = "./data/tmp/final_data.jsonl"
    output_jsonl_path = "./data/tmp/final_data_mask.jsonl"
    output_image_path = "./masked_images"

    mask_title(input_path, output_jsonl_path, output_image_path)