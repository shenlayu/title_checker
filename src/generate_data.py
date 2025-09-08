import re
import os
import json
import random

import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

def _extract_font_size(el: ET.Element) -> Optional[float]:
    """ 提取字号 """
    style = el.get("style") or ""
    m = re.compile(r"font-size\s*:\s*([\d.]+)px", re.I).search(style)
    try:
        return float(m.group(1))
    except ValueError:
        return None

def _get_text_content(el: ET.Element) -> str:
    """ 获取 <text> 节点的文本 """
    texts = []
    if el.text and el.text.strip():
        texts.append(el.text.strip())
    for child in el:
        txt = _get_text_content(child)
        if txt:
            texts.append(txt)
        if child.tail and child.tail.strip():
            texts.append(child.tail.strip())
    return " ".join(texts).strip()

def extract_title_from_svg(svg_path: str) -> Optional[str]:
    """ 从 SVG 文件中提取图表标题 """
    parser = ET.XMLParser()
    tree = ET.parse(svg_path, parser=parser)
    root = tree.getroot()

    for element in root.iter():
        if element.tag.endswith('g'):
            cls = element.get("class") or ""
            if "title" in {c.strip() for c in cls.split()}:
                text_nodes = [sub_element for sub_element in element.iter() if sub_element.tag.endswith('text')]
                if not text_nodes:
                    return None
                
                sizes = [(_extract_font_size(text), text) for text in text_nodes]
                valid_sizes = [size for size, _ in sizes if size is not None]
                max_size = max(valid_sizes) if valid_sizes else None

                if max_size is not None:
                    chosen = [text for size, text in sizes if size == max_size]
                else:
                    return None

                pieces = [_get_text_content(t) for t in chosen]
                title = ' '.join(pieces).strip()
                if title:
                    return title
                
                return None

    return None

def collect_samples(root_dir: str, image_name="chart.png", svg_name="chart.svg") -> List[Dict]:
    """
    遍历 root_dir, 收集样本
    返回 item: {"image_path": str, "title": Optional[str]}
    """
    samples = []
    idx = 0
    for cur, _, files in tqdm(os.walk(root_dir)):
        files_set = set(files)
        if image_name in files_set:
            img_path = os.path.join(cur, image_name)
            title = None
            if svg_name in files_set:
                title = extract_title_from_svg(os.path.join(cur, svg_name))
            samples.append({"image_path": img_path, "title": title})

        idx += 1
        if idx >= 20:
            break
        
    return samples

def build_clip_records(samples: List[Dict], neg_k: int = 2, allow_generic_negs: bool = True) -> List[Dict]:
    """ 基于获得的图表信息和标题信息，生成其余正样本及负样本 """
    titled = [s for s in samples if s.get("title")]

    records = []
    for i, s in enumerate(titled):
        pos = [s["title"]]

        # TODO 负例逻辑
        negs = [""]

        rec = {
            "image_path": s["image_path"],
            "pos_texts": pos,
            "neg_texts": negs
        }
        records.append(rec)

    return records

def split_dataset(records: List[Dict], val_ratio=0.1, test_ratio=0.1, seed=42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """ 划分数据集 """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)

    n = len(records)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_idx = set(idxs[:n_val])
    test_idx = set(idxs[n_val:n_val+n_test])

    train, val, test = [], [], []
    for j, rec in enumerate(records):
        if j in val_idx:
            val.append(rec)
        elif j in test_idx:
            test.append(rec)
        else:
            train.append(rec)
    return train, val, test

def write_jsonl(path: str, records: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_clip_jsonl(
    root_dir: str,
    out_dir: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    neg_k: int = 2,
    seed: int = 42,
    image_name: str = "chart.png",
    svg_name: str = "chart.svg",
    info_name: str = "info.json",
):
    samples = collect_samples(root_dir, image_name=image_name, svg_name=svg_name, info_name=info_name)
    if not samples:
        raise RuntimeError(f"未在 {root_dir} 下发现任何包含 {image_name} 的子目录。")

    records = build_clip_records(samples, neg_k=neg_k)
    if not records:
        raise RuntimeError("没有可用标题 (pos_texts), 请确认 SVG 或 info.json 中存在标题。")

    train, val, test = split_dataset(records, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), test)

if __name__ == "__main__":
    ROOT = "/data/lizhen/newdata/converted"
    OUT = "./clip_data"
    generate_clip_jsonl(
        root_dir=ROOT,
        out_dir=OUT,
        val_ratio=0.1,
        test_ratio=0.1,
        neg_k=2,
        seed=42,
    )