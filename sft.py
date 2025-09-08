# train_clip_multipos.py
import os, json, math, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup


# =========================
# 1) 数据集：一图多正样本 & 若干负样本
# =========================
class MultiPosCLIPDataset(Dataset):
    """
    每个样本:
      {
        "image_path": "...",
        "pos_texts": ["...", "..."],         # 至少1个
        "neg_texts": ["...", "..."]          # 可为0个
      }
    """
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    assert "image_path" in obj and "pos_texts" in obj
                    self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "image_path": item["image_path"],
            "pos_texts": item["pos_texts"],
            "neg_texts": item.get("neg_texts", []),
        }


# =========================
# 2) Collator：把变长文本打平
# =========================
@dataclass
class MultiPosCollator:
    processor: CLIPProcessor
    max_texts_per_image: Optional[int] = None   # 限制每图选入多少文本（正+负）以防爆显存
    image_aug: bool = True                      # 是否启用图像增强（可在 processor.image_processor 内调）

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 组装图像列表
        images = [Image.open(x["image_path"]).convert("RGB") for x in batch]

        # 打平所有文本，同时记录每个文本的“owner”图像索引及其是否为正样本
        flat_texts = []
        text_owner_img = []   # len == T，标记此文本属于哪一张图（正样本时有效）
        text_is_positive = [] # 与 text_owner_img 对齐，正文本 True，负文本 False；负文本 owner 设置为 -1

        for img_idx, item in enumerate(batch):
            pos_texts = item["pos_texts"]
            neg_texts = item.get("neg_texts", [])

            # 可选限幅：从正/负里随机抽取，控制每图文本总量
            if self.max_texts_per_image is not None:
                # 至少保留一个正样本
                keep_pos = min(len(pos_texts), max(1, self.max_texts_per_image // 2))
                keep_neg = max(0, self.max_texts_per_image - keep_pos)
                pos_texts = random.sample(pos_texts, keep_pos) if len(pos_texts) > keep_pos else pos_texts
                neg_texts = random.sample(neg_texts, keep_neg) if len(neg_texts) > keep_neg else neg_texts

            # 添加正文本
            for t in pos_texts:
                flat_texts.append(t)
                text_owner_img.append(img_idx)
                text_is_positive.append(True)

            # 添加负文本（显式负样本）
            for t in neg_texts:
                flat_texts.append(t)
                text_owner_img.append(-1)        # -1 表示它不是任何图像的正样本
                text_is_positive.append(False)

        # 可能所有样本都没给负文本，此时也没问题——用“跨图的正文本”互为负样本

        # 处理图像与文本张量
        proc_inputs = self.processor(images=images, text=flat_texts, return_tensors="pt", padding=True)
        # 组装辅助映射
        meta = {
            "text_owner_img": torch.tensor(text_owner_img, dtype=torch.long),   # T
            "text_is_positive": torch.tensor(text_is_positive, dtype=torch.bool),
        }
        return {**proc_inputs, **meta}


# =========================
# 3) 多正样本 InfoNCE（对称）
# =========================
class MultiPositiveCLIPLoss(nn.Module):
    """
    对称式：
      L = (L_img2txt + L_txt2img) / 2

    - img2txt：对每张图 i，soft label 在其正文本集合 P_i 上均匀分布；
               负样本为“同batch其它文本 + 显式负文本”
    - txt2img：只对拥有正图像的文本计算（显式负文本 owner = -1 不参与正项）
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_embeds: torch.Tensor,   # [B, D]
        text_embeds: torch.Tensor,    # [T, D]
        logit_scale: torch.Tensor,    # scalar
        text_owner_img: torch.Tensor, # [T], -1 表示该文本不是任何图像的正样本（显式负）
    ):
        device = image_embeds.device
        B, D = image_embeds.shape
        T, D2 = text_embeds.shape
        assert D == D2

        # 相似度 logits: [B, T]  与  [T, B]
        logits_img2txt = logit_scale * image_embeds @ text_embeds.t()     # [B, T]
        logits_txt2img = logit_scale * text_embeds @ image_embeds.t()     # [T, B]

        # ---------- img -> txt：多正样本 soft labels ----------
        # 对于每个 i ∈ [0..B-1]，找出 P_i = {t | text_owner_img[t] == i}
        # 构造 mask [B, T] 表示哪些位置是正样本
        pos_mask_img2txt = (text_owner_img.unsqueeze(0) == torch.arange(B, device=device).unsqueeze(1))  # [B, T]
        pos_counts = pos_mask_img2txt.sum(dim=1).clamp(min=1)  # 每行至少为1，避免除0（若某图没有正文本，会退化为只用 in-batch 负样本）

        log_prob_img2txt = torch.log_softmax(logits_img2txt, dim=1)  # [B, T]
        # 对于每张图 i，对其所有正文本的 log 概率取平均： - (1/|P_i|) * sum_{t in P_i} log p(t|i)
        loss_img2txt = -(pos_mask_img2txt.float() * log_prob_img2txt).sum(dim=1) / pos_counts
        loss_img2txt = loss_img2txt.mean()

        # ---------- txt -> img：只对有正图像的文本计算 ----------
        has_owner = (text_owner_img >= 0)  # [T]
        if has_owner.any():
            # 对每个文本 t，正图像索引为 text_owner_img[t]
            # 构造 one-hot 目标 [T, B]
            tgt_rows = torch.arange(T, device=device)[has_owner]
            tgt_cols = text_owner_img[has_owner]
            pos_mask_txt2img = torch.zeros((T, B), device=device, dtype=torch.float)
            pos_mask_txt2img[tgt_rows, tgt_cols] = 1.0

            log_prob_txt2img = torch.log_softmax(logits_txt2img, dim=1)  # [T, B]
            loss_txt2img = -(pos_mask_txt2img * log_prob_txt2img).sum(dim=1)
            loss_txt2img = loss_txt2img[has_owner].mean()
        else:
            # 极端情况：全是显式负文本
            loss_txt2img = torch.tensor(0.0, device=device)

        return (loss_img2txt + loss_txt2img) * 0.5


# =========================
# 4) 冻结/解冻策略
# =========================
def set_trainable_parameters(model: CLIPModel, freeze_vision_until: Optional[int] = None,
                             freeze_text_until: Optional[int] = None):
    """
    freeze_*_until: 冻结到第几层（含），None 表示不冻结。
    ViT 层名一般是 model.vision_model.encoder.layers.{i}
    文本层名一般是 model.text_model.encoder.layers.{i}
    """
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 解冻投影与 logit_scale
    for p in model.visual_projection.parameters():
        p.requires_grad = True
    for p in model.text_projection.parameters():
        p.requires_grad = True
    model.logit_scale.requires_grad = True

    # 视觉
    if freeze_vision_until is None:
        for p in model.vision_model.parameters():
            p.requires_grad = True
    else:
        # 仅解冻 encoder 的后半部分
        for name, p in model.vision_model.named_parameters():
            if ".encoder.layers." in name:
                # 抓层号
                idx = int(name.split(".encoder.layers.")[1].split(".")[0])
                if idx > freeze_vision_until:
                    p.requires_grad = True
            elif any(k in name for k in ["post_layernorm", "embeddings.position_embedding"]):
                p.requires_grad = True  # 可选：解冻后层归一化等
    # 文本
    if freeze_text_until is None:
        for p in model.text_model.parameters():
            p.requires_grad = True
    else:
        for name, p in model.text_model.named_parameters():
            if ".encoder.layers." in name:
                idx = int(name.split(".encoder.layers.")[1].split(".")[0])
                if idx > freeze_text_until:
                    p.requires_grad = True
            elif any(k in name for k in ["final_layer_norm", "embeddings.position_embedding"]):
                p.requires_grad = True


# =========================
# 5) 简单评估：Img->Txt top-1 正例命中率（仅供 sanity check）
# =========================
@torch.no_grad()
def evaluate_img2txt_top1(
    model: CLIPModel,
    dl: DataLoader,
    device: torch.device,
):
    model.eval()
    correct, total = 0, 0
    for batch in dl:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            output_hidden_states=False,
            return_dict=True,
        )
        # L2 归一化后的嵌入
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        sims = logit_scale * (image_embeds @ text_embeds.t())     # [B, T]

        # 找正文本集合
        text_owner_img = batch["text_owner_img"]                   # [T]
        for i in range(sims.size(0)):
            # 正文本索引
            pos_idx = (text_owner_img == i).nonzero(as_tuple=True)[0]
            if pos_idx.numel() == 0:
                continue
            pred = sims[i].argmax().item()
            correct += int(pred in pos_idx.tolist())
            total += 1
    return correct / max(total, 1)


# =========================
# 6) 训练主程
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True, help="训练集 jsonl")
    parser.add_argument("--val_jsonl", type=str, default=None, help="验证集 jsonl（可选）")
    parser.add_argument("--pretrained", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_images", type=int, default=8, help="每批图像数（文本数量变长）")
    parser.add_argument("--max_texts_per_image", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)  # 微调通常较小
    parser.add_argument("--wd", type=float, default=0.02)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--freeze_vision_until", type=int, default=9, help="冻结到第几层（含），base大概12层；-1 表示不冻结")
    parser.add_argument("--freeze_text_until", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 模型与处理器
    model = CLIPModel.from_pretrained(args.pretrained)
    processor = CLIPProcessor.from_pretrained(args.pretrained)

    # 冻结策略
    fv = None if args.freeze_vision_until < 0 else args.freeze_vision_until
    ft = None if args.freeze_text_until < 0 else args.freeze_text_until
    set_trainable_parameters(model, freeze_vision_until=fv, freeze_text_until=ft)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    model.to(device)

    # 数据
    train_ds = MultiPosCLIPDataset(args.train_jsonl)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_images,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=MultiPosCollator(processor, max_texts_per_image=args.max_texts_per_image),
    )

    val_dl = None
    if args.val_jsonl:
        val_ds = MultiPosCLIPDataset(args.val_jsonl)
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_images,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=MultiPosCollator(processor, max_texts_per_image=args.max_texts_per_image),
        )

    # 优化器 & 调度器
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * math.ceil(len(train_dl) / args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    loss_fn = MultiPositiveCLIPLoss()

    # 训练
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    output_hidden_states=False,
                    return_dict=True,
                )
                # L2 normalize
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds  = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
                logit_scale  = model.logit_scale.exp()
                loss = loss_fn(
                    image_embeds=image_embeds,
                    text_embeds=text_embeds,
                    logit_scale=logit_scale,
                    text_owner_img=batch["text_owner_img"],
                )

            loss.backward()
            if (global_step + 1) % args.grad_accum == 0:
                optim.zero_grad(set_to_none=True)
                scheduler.step()

            if (global_step + 1) % 20 == 0:
                print(f"epoch {epoch} step {step} | loss {loss.item():.4f} | logit_scale {logit_scale.item():.3f}")

            if val_dl and (global_step + 1) % max(1, args.eval_every) == 0:
                acc = evaluate_img2txt_top1(model, val_dl, device)
                print(f"[Eval] Img->Txt top1 acc: {acc:.4f}")

            global_step += 1

        # 每个 epoch 结束也评估一下
        if val_dl:
            acc = evaluate_img2txt_top1(model, val_dl, device)
            print(f"[Epoch {epoch}] Img->Txt top1 acc: {acc:.4f}")

    # 保存
    os.makedirs("ckpt", exist_ok=True)
    model.save_pretrained("ckpt/clip_multipos_finetuned")
    processor.save_pretrained("ckpt/clip_multipos_finetuned")
    print("Saved to ckpt/clip_multipos_finetuned")


if __name__ == "__main__":
    main()