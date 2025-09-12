import os
import sys

root_path = os.path.abspath(os.path.dirname(__file__)).split("src")[0]
os.chdir(root_path)
sys.path.append(os.path.join(root_path, "src"))

import torch
import hydra
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from utils import time_logger, init_experiment
from utils.train_utils import CLIPDataset, make_collate, InfoNCE

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

    output_dir = cfg.output_dir
    checkpoint_path = cfg.checkpoint_path
    model_name = cfg.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None:
        processor = CLIPProcessor.from_pretrained(checkpoint_path)
        model = CLIPModel.from_pretrained(checkpoint_path)
    elif model_name is not None:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    else:
        raise ValueError("Config 中需要 checkpoint_path 或 model_name")
    model = freeze_model(model, cfg.freeze.freeze_vision, cfg.freeze.freeze_text)
    model.to(device)

    dataset = CLIPDataset(cfg.input_path)
    collate_fn = make_collate(processor)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = cfg.dataset.batch_size, 
        shuffle = cfg.dataset.shuffle,
        collate_fn=collate_fn
    )

    loss_fn = InfoNCE(cfg.loss.temperature)

    num_epochs = cfg.num_epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr = cfg.optim.lr, weight_decay = cfg.optim.weight_decay
    )
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * cfg.optim.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(1, num_epochs + 1):
        model.train()
        global_step_base = (epoch - 1) * steps_per_epoch

        for step, batch in tqdm(enumerate(dataloader, start=1), 
                                 desc=f"训练 epoch={epoch}/{num_epochs}", total=steps_per_epoch):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad() if (cfg.freeze.freeze_text and cfg.freeze.freeze_vision) else torch.enable_grad():
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
            # TODO evaluate

        save_dir = os.path.join(output_dir, f"epoch{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        logger.info(f"Epoch{epoch} 结果保存到 {save_dir}.")


if __name__ == "__main__":
    train()