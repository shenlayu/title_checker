import torch
import torch.nn.functional as F

class InfoNCE():
    """ CLIP 使用的 loss """
    def __init__(self, temperature: float):
        self.temperature = temperature
        self.eps = 1e-8

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
        loss = torch.mean(-(numerator-denominator))

        # 检验指标
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            hit = pos_mask_full[torch.arange(B, device=device), preds] # 第 i 行的 pred 是否 hit
            itc_top1 = hit.float().mean()

        return loss, {
            "loss": loss.detach(),
            "itc_top1": itc_top1.detach(),
        }