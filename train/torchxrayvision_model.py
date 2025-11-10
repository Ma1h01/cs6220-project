import torch
import torchxrayvision as xrv
import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

# ---
# 1. Define the 14 Pathologies in the HF Dataset
# ---
# I've listed these from the HF dataset card
# We need this list to map them to the torchxrayvision labels.
CHEXPERT_HF_LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]


# ---
# 2. Create the Custom PyTorch Dataset
# ---
class CheXpertHFDataset(Dataset):
    def __init__(self, split='train', transform=None):
        # Load the specific dataset from Hugging Face
        self.dataset = datasets.load_dataset("danjacobellis/chexpert", split=split)
        self.transform = transform
        
        # Get the 18 "standard" pathologies from torchxrayvision
        # This is what the model will output
        self.all_pathologies = xrv.datasets.default_pathologies

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Get the item from the HF dataset
        item = self.dataset[idx]
        
        # 2. Process the image
        # The HF dataset provides a PIL Image object
        # We MUST convert it to grayscale ("L") for x-ray models
        image = item['image'].convert("L")
        if self.transform:
            image_tensor = self.transform(image)
        
        # 3. Process the labels (The "Hard Part")
        # We need to map the 14 HF labels to the 18 torchxrayvision labels
        
        # Create an 18-element tensor filled with 'nan'
        label_tensor = torch.full((len(self.all_pathologies),), float('nan'))
        
        for pathology_name in CHEXPERT_HF_LABELS:
            # Get the label value (e.g., 0.0, 1.0, or -1.0)
            label_value = item[pathology_name]
            
            # Find the correct index for this pathology in the 18-label list
            if pathology_name in self.all_pathologies:
                target_idx = self.all_pathologies.index(pathology_name)
                label_tensor[target_idx] = label_value
        
        return image_tensor, label_tensor

# ---
# 3. Define Transforms and Create DataLoaders
# ---
# torchxrayvision models handle their own normalization
# We just need to resize, crop, and convert to a tensor
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    xrv.datasets.XRayCenterCrop()
])

# Create the Dataset
# Using 'train[:100]' for a small sample, remove '[:100]' for the full dataset
train_dataset = CheXpertHFDataset(split='train[:100]', transform=data_transform)

# Create the DataLoader
# A batch_size of 16 or 32 is a good start
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print(f"Loaded {len(train_dataset)} images.")


# ---
# 4. Load Model and Set Up Training Loop
# ---
# Load the pre-trained DenseNet121 model
# model = xrv.models.ResNet(weights="resnet50-res512-all")
model = xrv.models.DenseNet(weights="densenet121-res224-chex")
model.train()

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# CRITICAL: We need a loss function that can handle masked labels
# We set reduction='none' to get the loss for every label,
# then we will manually filter out the ones we don't care about.
criterion = nn.BCEWithLogitsLoss(reduction='none')

print("Starting training loop...")

# Run for one epoch (a few batches) as a test
for i, (images, labels) in enumerate(train_loader):
    # (images, labels) = (batch_size, 1, 224, 224), (batch_size, 18)
    
    optimizer.zero_grad()
    
    # Get model outputs (logits)
    outputs = model(images)  # Shape: (batch_size, 18)
    
    # --- This is the most important part ---
    
    # 1. Create the mask for valid labels (0.0 or 1.0)
    # This is correct as before.
    mask = (labels == 0.0) | (labels == 1.0)
    
    # 2. **NEW:** Create a "clean" label tensor
    # We clone 'labels' and replace all invalid entries (nan, -1.0)
    # with 0.0. This prevents the loss function from seeing 'nan'.
    labels_clean = labels.clone()
    labels_clean[~mask] = 0.0  # ~mask means "not mask"
    
    # 3. Calculate loss using the "clean" labels. 
    # This will no longer produce 'nan'.
    loss_all = criterion(outputs, labels_clean)
    
    # 4. Apply the mask as before. This zeros out the loss
    # for all the 'nan' and '-1.0' labels.
    loss_masked = loss_all * mask.float()
    
    # 5. Calculate the final mean loss
    # We divide by the *number of valid labels* to get a correct average
    loss = loss_masked.sum() / mask.sum()
    
    # --- End of fix ---
    
    loss.backward()
    optimizer.step()
    
    print(f"  Batch {i+1}, Loss: {loss.item():.4f}")
    
    # Stop after a few batches for this demo
    if i > 5:
        break

print("Training demo complete.")