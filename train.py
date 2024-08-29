import torch
import os, sys
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import torchvision.transforms as transforms
from PIL import Image
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
env = os.path.dirname(os.path.abspath(__file__))

# dependencies 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'load_model'))

# Paths
model_path = "F:/CODE/ME621/models/furception_vae_1-0.safetensors"
dataset_path = "D:/DATA/E621/dataset.csv"
image_path = "D:/DATA/E621/images/"


# Load model
from modules import timer
from modules import initialize_util
from modules import initialize
initialize.imports()
initialize.check_versions()
initialize.initialize()
from modules import sd_models
model = sd_models.model_data.get_sd_model()

# Extract Encoder
vae = model.first_stage_model
vae_encoder = vae.encoder


class FurryClassifier(nn.Module):
    def __init__(self, vae_encoder):
        super().__init__()
        self.encoder = vae_encoder
        
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # The encoder's output shape is (batch_size, 8, 64, 64)
        self.flattened_size = 8 * 64 * 64
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        # Pass input through the encoder
        x = x.half()
        encoded = self.encoder(x)
        pred = self.fc(encoded)

        return pred

# Custom dataset
class FurryDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, split='train'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        
        # Filter data based on split
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        return self.data['label'].values


class BalancedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_indices = np.where(dataset.data['label'] == 1)[0]
        self.negative_indices = np.where(dataset.data['label'] == 0)[0]
        
    def __iter__(self):
        pos_idx = self.positive_indices.copy()
        np.random.shuffle(pos_idx)
        
        batches = []
        for i in range(0, len(pos_idx), self.batch_size // 2):
            pos_batch = pos_idx[i:i + self.batch_size // 2]
            neg_batch = np.random.choice(self.negative_indices, size=self.batch_size // 2, replace=False)
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            batches.append(batch.tolist())  # Convert to list and append as a batch
        
        return iter(batches)
    
    def __len__(self):
        return len(self.positive_indices) * 2 // self.batch_size  # Number of batches per epoch

    
# Data loading
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = FurryDataset(csv_file=dataset_path, img_dir=image_path, transform=transform, split='train')
val_dataset = FurryDataset(csv_file=dataset_path, img_dir=image_path, transform=transform, split='val')

train_sampler = BalancedSampler(train_dataset, batch_size=16)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model and optimizer
model = FurryClassifier(vae_encoder)
print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}") 
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss().cuda()
scaler = GradScaler()


# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0
    
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_train_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        
        
    
    train_loss = total_train_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # Validation
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validating'):
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_val_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss = total_val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val

    # Print epoch results
    print(f'[{epoch+1}/{num_epochs}] Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}')
    print(f'[{epoch+1}/{num_epochs}] Val Loss:   {val_loss:.5f}, Val Acc: {val_acc:.5f}')

# Save the model
output_dir = f"{env}/models/ME621_Fav_2.pth"
torch.save(model.state_dict(), output_dir)