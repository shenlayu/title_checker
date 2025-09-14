import os
import sys

root_path = os.path.abspath(os.path.dirname(__file__)).split("src")[0]
os.chdir(root_path)
sys.path.append(os.path.join(root_path, "src"))

import torch
import hydra
from tqdm import tqdm
from omegaconf import ListConfig
from torch.utils.data import DataLoader, random_split
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup

from utils import time_logger, init_experiment, wandb_finish
from utils.train_utils import CLIPDataset, make_collate, InfoNCE, BCE

def split_dataset(dataset, train_ratio: float, valid_ratio: float, test_ratio: float, seed: int):
    """ 切分数据集 """
    assert 0 < train_ratio < 1 and 0 <= valid_ratio < 1 and 0 <= test_ratio < 1
    assert abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-6, "train/val/test 比例之和必须为 1"

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_valid, n_test], generator=g)

@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    """ 评估模型 """
    model.eval()
    sums = {} # 任意指标的加和
    count = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_embeds = model.get_image_features(pixel_values=pixel_values)
        text_embeds = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        loss, stats = loss_fn(
            image_embeds = image_embeds,
            text_embeds = text_embeds,
            group_ptr = batch["group_ptr"].to(device),
            pos_mask = batch["pos_mask"].to(device)
        )
        stats = {"loss": loss.detach(), **stats}

        B = pixel_values.shape[0]
        for k, v in stats.items():
            val = float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
            sums[k] = sums.get(k, 0.0) + val * B
        count += B

    if count == 0:
        return {"loss": float("nan"), **({k: float("nan") for k in sums.keys()} if sums else {})}
    averaged = {k: (v / count) for k, v in sums.items()}
    return averaged

def freeze_model(model, freeze_vision: bool, freeze_text: bool):
    """ 冻结模型的一部分 """
    if freeze_vision:
        for p in model.vision_model.parameters():
            p.requires_grad = False
    if freeze_text:
        for p in model.text_model.parameters():
            p.requires_grad = False
    return model

@time_logger()
@hydra.main(config_path=f"{root_path}/configs/train", config_name="config", version_base=None)
def train(cfg):
    """ 模型训练 """
    cfg, logger = init_experiment(cfg)

    assert cfg.get("output_dir")
    assert cfg.get("input_path")
    assert cfg.get("checkpoint_path") or cfg.get("model_name")

    freeze_vision = cfg.freeze.get("freeze_vision", False)
    freeze_text = cfg.freeze.get("freeze_text", False)
    batch_size = cfg.dataset.get("batch_size", 32)
    shuffle = cfg.dataset.get("shuffle", True)
    pin_memory = cfg.dataset.get("pin_memory", torch.cuda.is_available())
    num_workers = cfg.dataset.get("num_workers", 4)
    train_ratio = cfg.dataset.get("train_ratio", 0.8)
    valid_ratio = cfg.dataset.get("valid_ratio", 0.1)
    test_ratio = cfg.dataset.get("test_ratio", 0.1)
    loss_type = cfg.loss.get("loss_type", "InfoNCE")
    temperature = cfg.loss.get("temperature", 0.07)
    lr = cfg.optim.get("lr", 1e-5)
    weight_decay = cfg.optim.get("weight_decay", 0.01)
    warmup_ratio = cfg.optim.get("warmup_ratio", 0.05)
    num_epochs = cfg.get("num_epochs", 3)
    seed = cfg.get("seed", 42)
    eval_type = cfg.get("eval_type", "itc_top1")
    if isinstance(eval_type, ListConfig):   # 👈 关键
        eval_type = list(eval_type)
    
    output_dir = cfg.output_dir
    checkpoint_path = cfg.checkpoint_path
    model_name = cfg.model_name
    input_path = cfg.input_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None:
        processor = CLIPProcessor.from_pretrained(checkpoint_path)
        model = CLIPModel.from_pretrained(checkpoint_path)
    elif model_name is not None:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    else:
        raise ValueError("Config 中需要 checkpoint_path 或 model_name")
    model = freeze_model(model, freeze_vision, freeze_text)
    model.to(device)

    full_dataset = CLIPDataset(input_path)
    train_ds, valid_ds, test_ds = split_dataset(
        full_dataset, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, seed=seed
    )
    collate_fn = make_collate(processor)
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        dataset=valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    if loss_type == "InfoNCE":
        loss_fn = InfoNCE(temperature, eval_type)
    elif loss_type == "BCE":
        loss_fn = BCE(temperature, eval_type)
    else:
        raise NotImplementedError(f"不支持的 loss: {loss_type}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr = lr, weight_decay = weight_decay
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_valid_loss = None
    best_epoch = None

    logger.info(f"验证集评估 epoch=0")
    valid_stats = evaluate(model, valid_loader, device, loss_fn)
    logger.wandb_metric_log({
        "valid/epoch": 0,
        **{f"valid/{k}": (v.item() if hasattr(v, "item") else float(v))
            for k, v in (valid_stats or {}).items()}
        }, level="info")

    for epoch in range(1, num_epochs + 1):
        model.train()
        global_step_base = (epoch - 1) * steps_per_epoch

        for step, batch in tqdm(enumerate(train_loader, start=1), 
                                 desc=f"训练 epoch={epoch}/{num_epochs}", total=steps_per_epoch):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad() if (freeze_text and freeze_vision) else torch.enable_grad():
                image_embeds = model.get_image_features(pixel_values=pixel_values)
                text_embeds  = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            loss, stats = loss_fn(
                image_embeds = image_embeds,
                text_embeds = text_embeds,
                group_ptr = batch["group_ptr"].to(device),
                pos_mask = batch["pos_mask"].to(device)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
            metric_dict = {
                "train/loss": float(loss.detach().cpu()),
                "train/lr": float(lr),
                "train/grad_norm": float(getattr(grad_norm, "item", lambda: grad_norm)()),
                "train/epoch": epoch,
                "train/step_in_epoch": step,
                "train/global_step": global_step_base + step,
                **{f"train/{k}": (v.item() if hasattr(v, "item") else float(v))
                    for k, v in (stats or {}).items()}
            }
            logger.wandb_metric_log(metric_dict, level="info")
        
        logger.info(f"验证集评估 epoch={epoch}")
        valid_stats = evaluate(model, valid_loader, device, loss_fn)
        logger.wandb_metric_log({
            "valid/epoch": epoch,
            **{f"valid/{k}": (v.item() if hasattr(v, "item") else float(v))
                for k, v in (valid_stats or {}).items()}
            }, level="info")
        if best_valid_loss is None or valid_stats["loss"] <= best_valid_loss:
            best_valid_loss = valid_stats["loss"]
            best_epoch = epoch

        save_dir = os.path.join(output_dir, f"epoch{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        logger.info(f"Epoch{epoch} 结果保存到 {save_dir}.")

    logger.info(f"测试集评估 epoch={epoch}")
    best_model_path = os.path.join(output_dir, f"epoch{best_epoch}")
    if os.path.isdir(best_model_path):
        best_model = CLIPModel.from_pretrained(best_model_path).to(device)
    else:
        best_model = model
    test_stats = evaluate(best_model, test_loader, device, loss_fn)
    logger.wandb_metric_log({
        **{f"test/{k}": (v.item() if hasattr(v, "item") else float(v))
            for k, v in (test_stats or {}).items()}
        }, level="info")

    wandb_finish()

if __name__ == "__main__":
    train()