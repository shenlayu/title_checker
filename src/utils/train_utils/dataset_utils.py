import os
import json

import torch
from torch.utils.data import Dataset
from PIL import Image

import logging
logger = logging.getLogger(__name__)


class CLIPDataset(Dataset):
    """ CLIP 使用的数据集 """
    def __init__(self, input_path: str):
        self.items = []
        bad_line, total_line = 0, 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_line += 1
                try:
                    line_json = json.loads(line)
                    assert "image_path" in line_json and "pos_texts" in line_json and "neg_texts" in line_json
                    line_json["pos_texts"] = self._clean_text(line_json["pos_texts"], "pos_texts")
                    line_json["neg_texts"] = self._clean_text(line_json["neg_texts"], "neg_texts")
                    if not (line_json["pos_texts"] and line_json["neg_texts"]):
                        bad_line += 1
                        continue
                    self.items.append(line_json)
                except:
                    bad_line += 1
                    continue
        logger.info(f"总共 {total_line} 条数据，其中 {bad_line} 条无效，已去除")

    def _clean_text(self, x, key):
        """ 清除数据格式不正确的条目 """
        if isinstance(x, str):
            x = [x]
        elif not isinstance(x, list):
            return []
        out = []
        for text in x:
            if text is None:
                return [] # 如果某一行存在 null, 直接去除这一行，以防影响训练
            if not isinstance(text, (str, int, float)):
                return []
            s = str(text).strip()
            if s:
                out.append(s)
            else:
                return []
        return out

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        return {"image": image, "pos_texts": item["pos_texts"], "neg_texts": item["neg_texts"]}

def make_collate(processor):
    def collate_fn(batch):
        B = len(batch)
        images = [b["image"] for b in batch]
        enc_img = processor(images=images, return_tensors="pt")

        texts = []
        pos_mask = []
        group_ptr = [0]
        for bi, item in enumerate(batch):
            pos_texts = item["pos_texts"]
            neg_texts = item["neg_texts"]

            for ti, t in enumerate(pos_texts + neg_texts):
                if not isinstance(t, str):
                    print(pos_texts)
                    print(neg_texts)
                    raise TypeError(f"Text at batch_index={bi}, text_index={ti} is {type(t)}: {t!r}")
                
            texts.extend(pos_texts)
            texts.extend(neg_texts)
            pos_mask.extend([True] * len(pos_texts) + [False] * len(neg_texts))
            group_ptr.append(group_ptr[-1] + len(pos_texts) + len(neg_texts))
        enc_texts = processor(text=texts, padding=True, return_tensors="pt")

        return {
            "pixel_values": enc_img["pixel_values"],
            "input_ids": enc_texts["input_ids"],
            "attention_mask": enc_texts["attention_mask"],
            "group_ptr": torch.tensor(group_ptr, dtype=torch.long),
            "pos_mask": torch.tensor(pos_mask, dtype=torch.bool) 
        }

    return collate_fn