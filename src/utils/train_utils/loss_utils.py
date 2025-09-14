import torch
import torch.nn.functional as F
from typing import Union, List

class InfoNCE():
    """ CLIP 使用的 loss """
    def __init__(self, temperature: float, eval_type: Union[List[str], str]):
        self.temperature = temperature
        self.eval_type = eval_type

    def __call__(self, image_embeds, text_embeds, group_ptr, pos_mask):
        images = F.normalize(image_embeds, dim=-1)
        texts = F.normalize(text_embeds, dim=-1)
        logits = (images @ texts.T) / self.temperature # logits[i][j] 表示图像 i 和文本 j 的相似度

        B = logits.shape[0] # batch size
        device = logits.device

        pos_mask_full = torch.zeros_like(logits, dtype=torch.bool)
        for row in range(B):
            start_idx = group_ptr[row].item()
            end_idx = group_ptr[row+1].item()
            pos_slice = pos_mask[start_idx: end_idx] # 这一区间的是否为正样本布尔掩码
            if pos_slice.any():
                pos_idx = torch.arange(start_idx, end_idx, device=device)[pos_slice] # 标定区间后应用布尔掩码
                pos_mask_full[row, pos_idx] = True
                
        # InfoNCE 的对数分母
        denominator = torch.logsumexp(logits, dim=1)
        # InfoNCE 的对数分子，求和时将负样本置为 -inf
        neg_mask = torch.full_like(logits, float("-inf"))
        logits_masked = torch.where(pos_mask_full, logits, neg_mask) # 若为正例，用 logits 值，否则用 -inf
        numerator = torch.logsumexp(logits_masked, dim=1)

        # 去除无正样本之行，否则 batch loss = inf
        valid_rows = pos_mask_full.any(dim=1)
        if valid_rows.any():
            diff = numerator[valid_rows] - denominator[valid_rows]
            loss = -diff.mean()
        else:
            loss = logits.new_tensor(float("nan")) # 和 logits 同设备，同 dtype 的单值 nan 张量

        eval_stats = {} # 保存评估指标
        with torch.no_grad():
            if (isinstance(self.eval_type, str) and self.eval_type == "itc_top1") or \
                isinstance(self.eval_type, list) and "itc_top1" in self.eval_type:
                preds = logits.argmax(dim=1)
                if valid_rows.any():
                    hit = pos_mask_full[torch.arange(B, device=logits.device), preds]
                    itc_top1 = hit[valid_rows].float().mean()
                else:
                    itc_top1 = logits.new_tensor(float("nan"))
                eval_stats.update({"itc_top1": itc_top1.detach()})
            if (isinstance(self.eval_type, str) and self.eval_type == "bc_acc") or \
                isinstance(self.eval_type, list) and "bc_acc" in self.eval_type: # 记原始标题（最后一个正例）和第一个负例中相似度更大者
                correct, total = 0, 0
                for row in range(B):
                    start_idx = group_ptr[row].item()
                    end_idx = group_ptr[row+1].item()
                    pos_slice = pos_mask[start_idx: end_idx]
                    if not torch.any(pos_slice):
                        continue
                    last_pos_idx_local = torch.nonzero(pos_slice, as_tuple=False)[-1].item()
                    last_pos_idx = start_idx + last_pos_idx_local

                    neg_slice = ~pos_slice
                    if not torch.any(neg_slice):
                        continue
                    first_neg_idx_local = torch.nonzero(neg_slice, as_tuple=False)[0].item()
                    first_neg_idx = start_idx + first_neg_idx_local

                    logit_pos = logits[row][last_pos_idx]
                    logit_neg = logits[row][first_neg_idx]
                    correct += int((logit_pos > logit_neg).item())
                    total += 1
                bc_acc = logits.new_tensor(float("nan")) if total == 0 else \
                    torch.tensor(correct/total, device=logits.device)
                eval_stats.update({"bc_acc": bc_acc})

        return loss, eval_stats
    

class BCE():
    """ 二分类 loss, 一个 batch 中一半用正例监督，一半用负例监督 """
    def __init__(self, eval_type: Union[List[str], str]):
        self.eval_type = eval_type

    def __call__(self, image_embeds, text_embeds, group_ptr, pos_mask):
        images = F.normalize(image_embeds, dim=-1)
        texts = F.normalize(text_embeds, dim=-1)
        logits = (images @ texts.T)

        B = logits.shape[0]
        device = logits.device

        pos_mask_full = torch.zeros_like(logits, dtype=torch.bool)
        for row in range(B):
            start_idx = group_ptr[row].item()
            end_idx = group_ptr[row+1].item()
            pos_slice = pos_mask[start_idx: end_idx]
            if pos_slice.any():
                pos_idx = torch.arange(start_idx, end_idx, device=device)[pos_slice]
                pos_mask_full[row, pos_idx] = True

        mid = B // 2
        selected_logits = []
        selected_labels = []

        for row in range(B):
            start_idx = group_ptr[row].item()
            end_idx = group_ptr[row+1].item()
            pos_slice = pos_mask[start_idx: end_idx]

            if row < mid:
                if torch.any(pos_slice):
                    last_pos_idx_local = torch.nonzero(pos_slice, as_tuple=False)[-1].item()
                    last_pos_idx = last_pos_idx_local + start_idx
                    selected_logits.append(logits[row, last_pos_idx])
                    selected_labels.append(1.0)
            else:
                neg_slice = ~pos_slice
                if torch.any(neg_slice):
                    first_neg_idx_local = torch.nonzero(neg_slice, as_tuple=False)[0].item()
                    first_neg_idx = start_idx + first_neg_idx_local
                    selected_logits.append(logits[row, first_neg_idx])
                    selected_labels.append(0.0)
            
        if len(selected_logits) > 0:
            selected_logits = torch.stack(selected_logits, dim=0)
            selected_labels = torch.tensor(selected_labels, device=device, dtype=selected_logits.dtype)
            loss = F.binary_cross_entropy_with_logits(selected_logits, selected_labels)
        else:
            loss = logits.new_tensor(float("nan"))

        eval_stats = {}
        with torch.no_grad():
            if (isinstance(self.eval_type, str) and self.eval_type == "itc_top1") or \
                isinstance(self.eval_type, list) and "itc_top1" in self.eval_type:
                preds = logits.argmax(dim=1)
                valid_rows = pos_mask_full.any(dim=1)
                if valid_rows.any():
                    hit = pos_mask_full[torch.arange(B, device=logits.device), preds]
                    itc_top1 = hit[valid_rows].float().mean()
                else:
                    itc_top1 = logits.new_tensor(float("nan"))
                eval_stats.update({"itc_top1": itc_top1.detach()})
            if (isinstance(self.eval_type, str) and self.eval_type == "bc_acc") or \
                isinstance(self.eval_type, list) and "bc_acc" in self.eval_type:
                correct, total = 0, 0
                for row in range(B):
                    start_idx = group_ptr[row].item()
                    end_idx = group_ptr[row+1].item()
                    row_pos_slice = pos_mask[start_idx: end_idx]
                    if not torch.any(row_pos_slice):
                        continue
                    last_pos_idx_local = torch.nonzero(row_pos_slice, as_tuple=False)[-1].item()
                    last_pos_idx = start_idx + last_pos_idx_local

                    row_neg_slice = ~row_pos_slice
                    if not torch.any(row_neg_slice):
                        continue
                    first_neg_idx_local = torch.nonzero(row_neg_slice, as_tuple=False)[0].item()
                    first_neg_idx = start_idx + first_neg_idx_local

                    logit_pos = logits[row, last_pos_idx]
                    logit_neg = logits[row, first_neg_idx]
                    correct += int((logit_pos > logit_neg).item())
                    total += 1

                bc_acc = logits.new_tensor(float("nan")) if total == 0 else \
                    torch.tensor(correct / total, device=device)
                eval_stats.update({"bc_acc": bc_acc})

        return loss, eval_stats