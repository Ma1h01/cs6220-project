#!/usr/bin/env python3
import os, argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

CHEXPERT_LABELS = [
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"
]

def compute_metrics(y_true, y_score):
    L = y_true.shape[1]
    aurocs, auprcs = [], []
    for i in range(L):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() < 10:
            aurocs.append(np.nan)
            auprcs.append(np.nan)
            continue
        try:
            au = roc_auc_score(y_true[mask, i], y_score[mask, i])
            ap = average_precision_score(y_true[mask, i], y_score[mask, i])
        except:
            au = np.nan
            ap = np.nan
        aurocs.append(au)
        auprcs.append(ap)
    return np.array(aurocs), np.array(auprcs)

class HF_CheXpert(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img)
        labels = np.array([row.get(label, np.nan) for label in CHEXPERT_LABELS], dtype=float)
        return img, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/BiomedVLP-BioViL-T")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="baseline_biovil_out")
    parser.add_argument("--max_text_len", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading HuggingFace CheXpert dataset (validation split)…")
    dset = load_dataset("danjacobellis/chexpert", split="validation")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.229]*3)
    ])

    val_ds = HF_CheXpert(dset, transform)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Loading BioViL multi-modal model…")
    vlp = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        force_download=True
    ).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.model_max_length = args.max_text_len

    # Build label prompts
    label_prompts = [f"There is {lbl.lower()}." for lbl in CHEXPERT_LABELS]

    print("Encoding label prompts…")
    vlp.eval()
    with torch.no_grad():
        tok = tokenizer(
            label_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_text_len
        ).to(args.device)
        text_emb = vlp.get_projected_text_embeddings(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"]
        )  # shape (L, D)
        text_emb = torch.nn.functional.normalize(text_emb, dim=1)

    all_scores = []
    all_labels = []

    print("Running zero-shot inference…")
    vlp.eval()
    for imgs, labels in tqdm(loader):
        imgs = imgs.to(args.device)
        with torch.no_grad():
            img_emb = vlp.get_image_embeddings(pixel_values=imgs)
            img_emb = torch.nn.functional.normalize(img_emb, dim=1)
            sims = (img_emb @ text_emb.T).cpu().numpy()
        all_scores.append(sims)
        all_labels.append(labels.numpy())

    y_score = np.vstack(all_scores)
    y_true = np.vstack(all_labels)

    np.save(os.path.join(args.out_dir, "preds_val.npy"), y_score)
    np.save(os.path.join(args.out_dir, "labels_val.npy"), y_true)

    print("Saved predictions.")

    print("Computing AUROC and AUPRC…")
    aurocs, auprcs = compute_metrics(y_true, y_score)

    for lbl, a, p in zip(CHEXPERT_LABELS, aurocs, auprcs):
        print(f"{lbl:25} AUROC={np.nan_to_num(a):.4f}  AUPRC={np.nan_to_num(p):.4f}")

    # Plot ROC + PR curves
    for i, lbl in enumerate(CHEXPERT_LABELS):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() < 10:
            continue
        fpr, tpr, _ = roc_curve(y_true[mask, i], y_score[mask, i])
        prec, rec, _ = precision_recall_curve(y_true[mask, i], y_score[mask, i])

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC - {lbl}")
        plt.savefig(os.path.join(args.out_dir, f"roc_{lbl.replace(' ','_')}.png"))
        plt.close()

        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR - {lbl}")
        plt.savefig(os.path.join(args.out_dir, f"pr_{lbl.replace(' ','_')}.png"))
        plt.close()

    print("Zero-shot baseline completed successfully.")
