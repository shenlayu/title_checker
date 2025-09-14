import json
from typing import Tuple, Dict, List, Set

def _read_jsonl(path: str) -> Dict[str, Tuple[List[str], List[str]]]:
    data: Dict[str, Tuple[List[str], List[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                image_path = d["image_path"]
                pos_texts = list(map(str, d.get("pos_texts", [])))
                neg_texts = list(map(str, d.get("neg_texts", [])))
                data[image_path] = (pos_texts, neg_texts)
            except (json.JSONDecodeError, KeyError) as e:
                continue
    return data

def merge(original_path: str, tmp_path: str):
    """ 将原始 jsonl 文件和其 tmp 版本融合 """
    original_pos_neg = _read_jsonl(original_path)
    tmp_pos_neg = _read_jsonl(tmp_path)

    all_pos_neg: Dict[str, Tuple[Set[str], Set[str]]] = {}

    for image_path, (pos, neg) in original_pos_neg.items():
        all_pos_neg[image_path] = (set(pos), set(neg))

    for image_path, (tpos, tneg) in tmp_pos_neg.items():
        apos, aneg = all_pos_neg.get(image_path, (set(), set()))
        apos.update(tpos)
        aneg.update(tneg)
        all_pos_neg[image_path] = (apos, aneg)

    with open(original_path, "w", encoding="utf-8") as f:
        for image_path, (apos, aneg) in all_pos_neg.items():
            if apos and aneg:
                line_dict = {
                    "image_path": image_path,
                    "pos_texts": list(apos),
                    "neg_texts": list(aneg)
                }
                f.write(json.dumps(line_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    original_path = "/home/shenyu/code/title_checker/data/tmp/final_data.jsonl"
    tmp_path = "/home/shenyu/code/title_checker/data/tmp/final_data.jsonl.tmp"

    merge(original_path, tmp_path)