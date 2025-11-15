import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms

# CheXpert pathologies (14 labels)
CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.transform = transform
        
        # Handle uncertain labels (CheXpert uses -1 for uncertain)
        # Common strategy: map -1 to 0 (negative) or 1 (positive)
        # Here we'll map -1 to 0 and keep NaN as NaN
        self.labels = self.df[CHEXPERT_LABELS].fillna(0).replace(-1, 0).values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = self.img_root / self.df.iloc[idx]['Path'].replace('CheXpert-v1.0-small/', '')
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return img, labels

def load_biovilt_model(device='cuda'):
    """Load BioViL-T model from HuggingFace"""
    print("Loading BioViL-T model...")
    model = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-BioViL-T")
    model = model.to(device)
    model.eval()
    return model, tokenizer

def get_image_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def compute_similarity_scores(model, tokenizer, dataloader, device='cuda'):
    """
    Compute zero-shot classification scores using text prompts
    """
    # Create text prompts for each pathology
    text_prompts = [f"chest x-ray showing {label.lower()}" for label in CHEXPERT_LABELS]
    
    # Tokenize prompts
    text_inputs = tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    # Get text embeddings
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    all_predictions = []
    all_labels = []
    
    print("Computing predictions...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            
            # Get image embeddings
            image_features = model.get_image_features(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            similarity = (image_features @ text_features.T)
            
            all_predictions.append(similarity.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    
    return predictions, labels

def calculate_auroc(predictions, labels):
    """Calculate AUROC for each class"""
    aurocs = {}
    
    for i, label_name in enumerate(CHEXPERT_LABELS):
        # Only compute AUROC if we have both positive and negative examples
        if len(np.unique(labels[:, i])) > 1:
            try:
                auroc = roc_auc_score(labels[:, i], predictions[:, i])
                aurocs[label_name] = auroc
            except Exception as e:
                print(f"Could not compute AUROC for {label_name}: {e}")
                aurocs[label_name] = np.nan
        else:
            aurocs[label_name] = np.nan
    
    # Calculate mean AUROC (excluding NaN values)
    valid_aurocs = [v for v in aurocs.values() if not np.isnan(v)]
    mean_auroc = np.mean(valid_aurocs) if valid_aurocs else np.nan
    
    return aurocs, mean_auroc

def plot_roc_curves(predictions, labels, save_path='roc_curves.png'):
    """Plot ROC curves for all pathologies"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i, label_name in enumerate(CHEXPERT_LABELS):
        ax = axes[i]
        
        if len(np.unique(labels[:, i])) > 1:
            try:
                fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
                auroc = roc_auc_score(labels[:, i], predictions[:, i])
                
                ax.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(label_name)
                ax.legend()
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                ax.set_title(label_name)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(label_name)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.show()

def main():
    # Configuration
    csv_path = './CheXpert-v1.0-small/valid.csv'
    img_root = './CheXpert-v1.0-small'
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_biovilt_model(device)
    
    # Create dataset and dataloader
    transform = get_image_transform()
    dataset = CheXpertDataset(csv_path, img_root, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Compute predictions
    predictions, labels = compute_similarity_scores(model, tokenizer, dataloader, device)
    
    # Calculate AUROC
    aurocs, mean_auroc = calculate_auroc(predictions, labels)
    
    # Print results
    print("\n" + "="*50)
    print("AUROC Results:")
    print("="*50)
    for label_name, auroc in aurocs.items():
        if not np.isnan(auroc):
            print(f"{label_name:30s}: {auroc:.4f}")
        else:
            print(f"{label_name:30s}: N/A")
    print("="*50)
    print(f"{'Mean AUROC':30s}: {mean_auroc:.4f}")
    print("="*50)
    
    # Plot ROC curves
    plot_roc_curves(predictions, labels)
    
    # Save results
    results_df = pd.DataFrame({
        'Pathology': list(aurocs.keys()),
        'AUROC': list(aurocs.values())
    })
    results_df.to_csv('biovilt_chexpert_results.csv', index=False)
    print("\nResults saved to biovilt_chexpert_results.csv")

if __name__ == "__main__":
    main()