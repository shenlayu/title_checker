import torch
import torch.nn.functional as F
from typing import Union, List, Dict

class InfoNCE():
    """ CLIP 使用的 loss """
    def __init__(self, temperature: float, eval_type: Union[List[str], str]):
        self.temperature = temperature
        self.eval_type = eval_type

    def __call__(self, image_embeds, text_embeds, group_ptr, pos_mask):
        images = F.normalize(image_embeds, dim=-1)
        texts = F.normalize(text_embeds, dim=-1)
        logits = (images @ texts.T) / self.temperature # logits[i][j] 表示图像 i 和文本 j 的相似度

        B, T = logits.shape # batch size, text count
        device = logits.device

        pos_mask_full = _Helper.build_pos_mask_full(B, T, group_ptr, pos_mask, device)
                
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

        eval_stats = _Helper.eval_metrics(logits, group_ptr, pos_mask, pos_mask_full, self.eval_type)

        return loss, eval_stats
    

class BCE():
    """ 二分类 loss, 一个 batch 中一半用正例监督，一半用负例监督 """
    def __init__(self, temperature: float, eval_type: Union[List[str], str]):
        self.temperature = temperature
        self.eval_type = eval_type

    def __call__(self, image_embeds, text_embeds, group_ptr, pos_mask):
        images = F.normalize(image_embeds, dim=-1)
        texts = F.normalize(text_embeds, dim=-1)
        logits = (images @ texts.T) / self.temperature

        B, T = logits.shape
        device = logits.device

        pos_mask_full = _Helper.build_pos_mask_full(B, T, group_ptr, pos_mask, device)

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

        eval_stats = _Helper.eval_metrics(logits, group_ptr, pos_mask, pos_mask_full, self.eval_type)

        return loss, eval_stats

class _Helper:
    @staticmethod
    def build_pos_mask_full(B: int, T: int, group_ptr: torch.Tensor, pos_mask: torch.Tensor, device) -> torch.Tensor:
        pos_mask_full = torch.zeros((B, T), dtype=torch.bool, device=device)
        for row in range(B):
            start_idx = group_ptr[row].item()
            end_idx = group_ptr[row+1].item()
            pos_slice = pos_mask[start_idx: end_idx] # 这一区间的是否为正样本布尔掩码
            if pos_slice.any():
                pos_idx = torch.arange(start_idx, end_idx, device=device)[pos_slice] # 标定区间后应用布尔掩码
                pos_mask_full[row, pos_idx] = True
        return pos_mask_full

    @staticmethod
    def eval_metrics(
        logits: torch.Tensor,
        group_ptr: torch.Tensor,
        pos_mask: torch.Tensor,
        pos_mask_full: torch.Tensor,
        eval_type: Union[List[str], str]
    ) -> Dict[str, torch.Tensor]:
        B, T = logits.shape
        device = logits.device

        eval_stats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            valid_rows = pos_mask_full.any(dim=1)

            if (isinstance(eval_type, str) and eval_type == "itc_top1") or \
                (isinstance(eval_type, list) and "itc_top1" in eval_type):
                preds = logits.argmax(dim=1)
                if valid_rows.any():
                    hit = pos_mask_full[torch.arange(B, device=device), preds]
                    itc_top1 = hit[valid_rows].float().mean()
                else:
                    itc_top1 = logits.new_tensor(float("nan"))
                eval_stats["itc_top1"] = itc_top1.detach()

            if (isinstance(eval_type, str) and eval_type == "bc_acc") or \
                (isinstance(eval_type, list) and "bc_acc" in eval_type):  # 记原始标题（最后一个正例）和第一个负例中相似度更大者
                correct, total = 0, 0
                for row in range(B):
                    start_idx, end_idx = group_ptr[row].item(), group_ptr[row+1].item()
                    if end_idx <= start_idx:
                        continue
                    row_pos = pos_mask[start_idx: end_idx]
                    if not row_pos.any():
                        continue
                    last_pos_local = torch.nonzero(row_pos, as_tuple=False)[-1].item()
                    last_pos = start_idx + last_pos_local

                    row_neg = ~row_pos
                    if not row_neg.any():
                        continue
                    first_neg_local = torch.nonzero(row_neg, as_tuple=False)[0].item()
                    first_neg = start_idx + first_neg_local

                    correct += int((logits[row, last_pos] > logits[row, first_neg]).item())
                    total += 1

                bc_acc = logits.new_tensor(float("nan")) if total == 0 else \
                    torch.tensor(correct / total, device=device)
                eval_stats["bc_acc"] = bc_acc

            if (isinstance(eval_type, str) and eval_type == "auc") or \
                (isinstance(eval_type, list) and ("auc" in eval_type)):
                pos_in, neg_in, pos_cross, neg_cross = [], [], [], []

                for row in range(B):
                    start_idx, end_idx = group_ptr[row].item(), group_ptr[row+1].item()
                    if end_idx <= start_idx:
                        continue
                    labels = pos_mask[start_idx: end_idx]
                    if labels.any():
                        scores_in = logits[row, start_idx: end_idx]
                        pos_in.append(scores_in[labels])
                        neg_in.append(scores_in[~labels])

                        if (end_idx - start_idx) < T:
                            pos_cross.append(scores_in[labels])
                            left  = torch.arange(0, start_idx, device=device)
                            right = torch.arange(end_idx, T, device=device)
                            if left.numel() + right.numel() > 0:
                                neg_idx = torch.cat([left, right], dim=0)
                                neg_cross.append(logits[row, neg_idx])

                def _calc_auc(pos_list, neg_list):
                    if len(pos_list) == 0 or len(neg_list) == 0:
                        return logits.new_tensor(float("nan"))
                    pos = torch.cat(pos_list, dim=0)
                    neg = torch.cat(neg_list, dim=0)
                    if pos.numel() == 0 or neg.numel() == 0:
                        return logits.new_tensor(float("nan"))
                    diff = pos[:, None] - neg[None, :]
                    return (diff > 0).float().mean() + 0.5 * (diff == 0).float().mean()

                eval_stats["auc_in_group"] = _calc_auc(pos_in, neg_in)
                eval_stats["auc_cross_batch"] = _calc_auc(pos_cross, neg_cross)

        return eval_stats