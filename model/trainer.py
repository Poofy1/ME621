import torch
import os, sys, csv
from torchvision.utils import save_image
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import torchvision.transforms as transforms
from PIL import Image
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# Get the directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
SAVE_DIR = os.path.join(parent_dir, 'data')


class FurryClassifier(nn.Module):
    def __init__(self, vae_encoder, img_size, dropout = 0.75):
        super().__init__()
        self.encoder = vae_encoder
        
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        
        encoder_dim = img_size / 8
        self.flattened_size = (int) (8 * encoder_dim * encoder_dim)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),  # First hidden layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # Output layer
        )
    
    def forward(self, x):
        # Pass input through the encoder
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

        # Precompute all IDs
        self.image_ids = self.data['image_name'].apply(lambda x: x.split('.')[0]).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[index, 1]
        image_id = self.image_ids[index]

        if self.transform:
            image = self.transform(image)

        return image, label, image_id

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



def evaluate_and_save_worst_images(model, dataset, output_dir, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    all_losses = []
    all_image_ids = []
    all_labels = []
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():
        for images, labels, image_ids in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                losses = criterion(outputs, labels)
            
            # Ensure losses is always a list or 1D tensor
            if isinstance(losses, float):
                losses = [losses]
            elif isinstance(losses, torch.Tensor):
                losses = losses.view(-1).cpu().tolist()
            
            all_losses.extend(losses)
            all_image_ids.extend(image_ids)
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate the number of images to save (5% of total)
    num_worst = int(len(all_losses) * 0.05)

    # Sort losses and get indices of worst performing images
    sorted_indices = np.argsort(all_losses)[::-1]  # Sort in descending order
    worst_indices = sorted_indices[:num_worst]

    # Prepare data for CSV
    worst_data = [(all_image_ids[idx], all_labels[idx], all_losses[idx]) for idx in worst_indices]

    # Write to CSV
    csv_path = os.path.join(output_dir, 'worst_performing.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['image_name', 'label', 'loss'])  # Write header
        for image_id, label, loss in worst_data:
            csvwriter.writerow([f"{image_id}.png", int(label), f"{loss:.4f}"])

    print(f"Saved {num_worst} worst performing instances (5% of total) to {csv_path}")
    

def create_val_split(csv_path, val_ratio=0.2):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Add a 'split' column, initialize all to 'train'
    df['split'] = 'train'
    
    # Separate positive and negative instances
    positive_instances = df[df['label'] == 1]
    negative_instances = df[df['label'] == 0]
    
    # Calculate the number of validation samples (20% of positive instances)
    n_val_samples = int(len(positive_instances) * val_ratio)
    
    # Randomly select validation samples from positive instances
    val_positive = positive_instances.sample(n=n_val_samples, random_state=42)
    
    # Randomly select an equal number of validation samples from negative instances
    val_negative = negative_instances.sample(n=n_val_samples, random_state=42)
    
    # Combine validation samples
    val_samples = pd.concat([val_positive, val_negative])
    
    # Update the 'split' column for validation samples
    df.loc[val_samples.index, 'split'] = 'val'
    
    # Save the updated DataFrame back to CSV
    df.to_csv(csv_path, index=False)
    
    print(f"Updated {csv_path} with 'split' column.")
    print(f"Train samples: {len(df[df['split'] == 'train'])}")
    print(f"Validation samples: {len(df[df['split'] == 'val'])}")
    

def load_model():
    sys.path.append(os.path.join(parent_dir, 'load_model'))
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
    
    return vae_encoder
    
def train_model():
    print("Starting Training")
    img_size = 512
    batch_size = 8

    # Paths
    dataset_path = f"{SAVE_DIR}/dataset.csv"
    image_path = f"{SAVE_DIR}/images/"


    # Load model
    vae_encoder = load_model()

    # Data loading
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    create_val_split(dataset_path)
    train_dataset = FurryDataset(csv_file=dataset_path, img_dir=image_path, transform=train_transform, split='train')
    val_dataset = FurryDataset(csv_file=dataset_path, img_dir=image_path, transform=val_transform, split='val')
        
        
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    train_sampler = BalancedSampler(train_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_sampler = BalancedSampler(val_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, shuffle=False)

    # Initialize model and optimizer
    model = FurryClassifier(vae_encoder, img_size)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}") 
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = nn.BCEWithLogitsLoss().cuda()
    scaler = GradScaler()


    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    patience = 5
    counter = 0
    epochs_without_improvement = 0

    # Create the directory path
    model_dir = f"{SAVE_DIR}/models"
    os.makedirs(model_dir, exist_ok=True)

    # Define the full path for the .pth file
    output_path = f"{model_dir}/ME621.pth"

    epoch = 0
    while True:
        epoch += 1
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        print(f'Epoch {epoch} - Training')
        for i, (images, labels, image_ids) in enumerate(train_loader):
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
            
            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.5f}')
            
            
        
        train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        print(f'Epoch {epoch} - Validating')
        with torch.no_grad():
            for i, (images, labels, image_ids) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.float().to(device).unsqueeze(1)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
                if (i + 1) % 10 == 0:  # Print every 10 batches
                    print(f'Batch {i+1}/{len(val_loader)}, Loss: {loss.item():.5f}')

        val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        # Print epoch results
        scheduler.step(val_loss)
        print(f'\n[Epoch {epoch}] Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f}')
        print(  f'[Epoch {epoch}] Val Loss:   {val_loss:.5f  } | Val Acc: {val_acc:.5f}')

        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            epochs_without_improvement = 0
            print(f"Validation loss improved. Saving model to {output_path}")
            torch.save(model.state_dict(), output_path)
        else:
            counter += 1
            epochs_without_improvement += 1

        # Check for early stopping
        if counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

        print(f"Epochs without improvement: {epochs_without_improvement}")

    print("Training completed")
    
    
    # Load the best performing model
    best_model = FurryClassifier(vae_encoder, img_size)
    best_model.load_state_dict(torch.load(output_path))
    best_model.to(device)

    # Combine train and val datasets
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    # Evaluate and save worst performing images
    evaluate_and_save_worst_images(best_model, full_dataset, SAVE_DIR, device)