#!/usr/bin/env python3
"""
finetune_biovil_multimodal_lora.py

Finetune microsoft/BiomedVLP-BioViL-T on HuggingFace CheXpert (danjacobellis/chexpert)
using LoRA adapters applied to both the image transformer's attention and the text encoder.

Saves:
 - PEFT adapters (save_pretrained)
 - classifier head weights (torch.save)

Example:
python finetune_biovil_multimodal_lora.py --batch_size 8 --epochs 4 --device cuda
"""
import os
import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score

# ---------- CheXpert labels ----------
CHEXPERT_LABELS = [
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"
]

# --------- dataset wrapper for HF chexpert ----------
class HF_CheXpert(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform=None, text_key_candidates=("report","Report","findings","Findings","Report_text","RadiologyReport")):
        self.ds = hf_ds
        self.transform = transform
        self.text_key_candidates = text_key_candidates

    def __len__(self):
        return len(self.ds)

    def _get_report_text(self, row):
        for k in self.text_key_candidates:
            if k in row and row[k] is not None:
                return row[k]
        # fallback empty string
        return ""

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        # build label vector (some entries may be None -> map to np.nan)
        labs = []
        for lbl in CHEXPERT_LABELS:
            v = row.get(lbl, None)
            if v is None:
                # try lowercase/csv-style variants
                v = row.get(lbl.lower(), None) or row.get(lbl.replace(" ", "_"), None)
            labs.append(np.nan if v is None else float(v))
        labs = np.array(labs, dtype=float)
        report = self._get_report_text(row)
        return img, report, labs

# ---------- helper: AUROC per label ----------
def compute_auroc_per_label(y_true, y_score):
    L = y_true.shape[1]
    aurocs = []
    for i in range(L):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() < 10:
            aurocs.append(np.nan)
            continue
        try:
            au = roc_auc_score(y_true[mask, i], y_score[mask, i])
        except Exception:
            au = np.nan
        aurocs.append(au)
    return aurocs

# ---------- multimodal model wrapper ----------
class MultiModalHead(nn.Module):
    def __init__(self, base_model, embed_dim, num_labels=len(CHEXPERT_LABELS), dropout=0.1):
        super().__init__()
        self.base = base_model  # peft-enabled base
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_labels)
        )

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        # Use base model helpers for projected embeddings
        img_emb = self.base.get_image_embeddings(pixel_values=pixel_values)
        txt_emb = self.base.get_projected_text_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.cat([img_emb, txt_emb], dim=1)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/BiomedVLP-BioViL-T")
    parser.add_argument("--chexpert_dataset", default="danjacobellis/chexpert")
    parser.add_argument("--chexpert_train_split", default="train")
    parser.add_argument("--chexpert_val_split", default="validation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="finetune_lora_out")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj","k_proj","v_proj","o_proj","out_proj","proj","dense"])
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
    ])

    # load HF dataset splits
    print("Loading HuggingFace CheXpert dataset...")
    ds_train = load_dataset(args.chexpert_dataset, split=args.chexpert_train_split)
    ds_val = load_dataset(args.chexpert_dataset, split=args.chexpert_val_split)

    train_ds = HF_CheXpert(ds_train, transform=transform)
    val_ds = HF_CheXpert(ds_val, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # load base model + tokenizer
    print("Loading base model and tokenizer:", args.model_name)
    base = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # determine embed dim (projection_dim or hidden_size fallback)
    embed_dim = getattr(base.config, "projection_dim", None) or getattr(base.config, "hidden_size", 768)
    print("Detected embed_dim:", embed_dim)

    # Prepare LoRA config (PEFT)
    print("Configuring LoRA (r=%d, alpha=%d, dropout=%g) for target modules: %s" %
          (args.lora_r, args.lora_alpha, args.lora_dropout, ", ".join(args.target_modules)))
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM  # best-effort; model is multimodal but PEFT expects a TaskType
    )

    # Wrap base model with PEFT LoRA adapters
    print("Applying PEFT LoRA to base model (this may print module-name warnings).")
    base_peft = get_peft_model(base, lora_config)

    # Build multimodal model with the peft-enabled base
    model = MultiModalHead(base_peft, embed_dim=embed_dim).to(device)

    # By design PEFT marks adapter params trainable; ensure only adapter + head are trainable
    trainable = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append((name, p.numel()))
    print("Trainable parameter name snippets (nonzero):")
    for n, cnt in trainable:
        print("  ", n, cnt)
    total_trainable = sum([cnt for _, cnt in trainable])
    print("Total trainable params:", total_trainable)

    # Optimizer: head + any LoRA parameters (PEFT params have 'lora' in their name)
    params = []
    head_params = list(model.head.parameters())
    params.append({"params": head_params, "lr": args.lr_head})
    # collect lora params
    lora_params = [p for n,p in model.base.named_parameters() if "lora" in n.lower() and p.requires_grad]
    if len(lora_params) > 0:
        params.append({"params": lora_params, "lr": args.lr_lora})
    else:
        print("Warning: no LoRA params found by substring 'lora' - check target_modules or model internals.")

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # training loop
    print("Starting training on device:", device)
    best_macro_auroc = -1.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        it = 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for imgs, reports, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device)
            # tokenize reports
            tok = tokenizer(list(reports), truncation=True, padding=True, max_length=args.max_text_len, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items()}

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(pixel_values=imgs, input_ids=tok["input_ids"], attention_mask=tok.get("attention_mask", None))
                loss_mat = criterion(logits, labels)  # B x L
                mask = (labels != -1.0).float()
                loss = (loss_mat * mask).sum() / (mask.sum() + 1e-8)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # optional: gradient clipping
            torch.nn.utils.clip_grad_norm_( [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            it += 1
            pbar.set_postfix(loss=epoch_loss / it)

        avg_loss = epoch_loss / max(1, it)
        print(f"Epoch {epoch} training loss: {avg_loss:.4f}")

        # validation
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for imgs, reports, labels in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.numpy()
                tok = tokenizer(list(reports), truncation=True, padding=True, max_length=args.max_text_len, return_tensors="pt")
                tok = {k: v.to(device) for k, v in tok.items()}
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(pixel_values=imgs, input_ids=tok["input_ids"], attention_mask=tok.get("attention_mask", None))
                    probs = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(probs)
                all_labels.append(labels)
        y_score = np.vstack(all_scores)
        y_true = np.vstack(all_labels)

        aurocs = compute_auroc_per_label(y_true, y_score)
        # macro average (mean of non-nan aurocs)
        valid_aurocs = [a for a in aurocs if not np.isnan(a)]
        macro_auroc = np.mean(valid_aurocs) if len(valid_aurocs) > 0 else np.nan
        print(f"Epoch {epoch} AUROC (per label):")
        for lbl, a in zip(CHEXPERT_LABELS, aurocs):
            print(f"  {lbl:25}: {a}")
        print(f"Epoch {epoch} macro AUROC: {macro_auroc:.4f}")

        # save checkpoint (PEFT adapters + head)
        ckpt_dir = os.path.join(args.out_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            # base_peft is model.base
            model.base.save_pretrained(ckpt_dir)
            print("Saved PEFT adapters to", ckpt_dir)
        except Exception as e:
            print("Warning: failed to save peft adapters via save_pretrained:", e)
        # save head
        head_path = os.path.join(ckpt_dir, "head.pth")
        torch.save(model.head.state_dict(), head_path)
        print("Saved classifier head to", head_path)

        # update best
        if not np.isnan(macro_auroc) and macro_auroc > best_macro_auroc:
            best_macro_auroc = macro_auroc
            best_dir = os.path.join(args.out_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            # copy latest peft + head (or save again)
            try:
                model.base.save_pretrained(best_dir)
            except:
                pass
            torch.save(model.head.state_dict(), os.path.join(best_dir, "head.pth"))
            print("Saved new best model to", best_dir)

    print("Training complete. Best macro AUROC:", best_macro_auroc)
