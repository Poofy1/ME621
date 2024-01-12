import torch
import os
import random
import torch.nn as nn
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm as tqdm_loop
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import classification_report
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from collections import Counter
from efficientnet_pytorch import EfficientNet
from torch.utils.data import WeightedRandomSampler
import multiprocessing
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torchinfo import summary
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class StaticNoise:
    def __init__(self, intensity_min=0.0, intensity_max=0.2):
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, img):
        intensity = random.uniform(self.intensity_min, self.intensity_max)
        noise = torch.randn(*img.size()) * intensity
        img = torch.clamp(img + noise, 0, 1)

        return img
    
class SquareResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Calculate the dimensions of the padded image
        max_dim = max(img.size)
        h_pad = max_dim - img.size[1]
        w_pad = max_dim - img.size[0]

        # Pad the image with zeros
        padded_img = transforms.functional.pad(img, (w_pad//2, h_pad//2, w_pad - w_pad//2, h_pad - h_pad//2), 0)

        # Resize the image to the desired size
        resized_img = transforms.Resize((self.size, self.size))(padded_img)

        return resized_img

def create_datasets(image_dir, data_csv, image_size):
    # Load the dataset
    dataset_df = pd.read_csv(data_csv, usecols=['ID', 'Upvotes', 'Downvotes', 'Favorites', 'Valid'])

    # Split into train and val
    train_df = dataset_df[dataset_df['Valid'] == 0]
    val_df = dataset_df[dataset_df['Valid'] == 1]

    # Create image paths
    train_image_paths = [os.path.join(image_dir, f"{image_id}.png") for image_id in train_df['ID']]
    val_image_paths = [os.path.join(image_dir, f"{image_id}.png") for image_id in val_df['ID']]

    # Convert DataFrame columns to lists
    train_labels = train_df[['Upvotes', 'Downvotes', 'Favorites']].values.tolist()
    val_labels = val_df[['Upvotes', 'Downvotes', 'Favorites']].values.tolist()

    # Define transforms
    train_transform = transforms.Compose([
                transforms.RandomRotation(degrees=90),
                transforms.RandomResizedCrop(size=image_size, scale=(1, 1.5)),
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.RandomHorizontalFlip(),
                #transforms.GaussianBlur(kernel_size=3, sigma=(0.0001, 0.3)),
                transforms.ColorJitter(brightness=(0.7, 1.1), contrast=(0.35, 1.15), saturation=(0, 1.5), hue=(-0.1, 0.1)),
                SquareResize(image_size),
                transforms.ToTensor(),
                StaticNoise(intensity_min=0.0, intensity_max=0.03),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
    val_transform = transforms.Compose([
                SquareResize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])


    # Create datasets
    train_dataset = ME621_Dataset(train_image_paths, train_labels, transform=train_transform)
    val_dataset = ME621_Dataset(val_image_paths, val_labels, transform=val_transform)

    return train_dataset, val_dataset



class ME621_Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        # Transform
        img = self.transform(img)

        return img, label
    
    

class ME621_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(ME621_Model, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b0", dropout_rate=dropout_rate)
    
        # Freeze all layers in EfficientNet
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two blocks
        num_blocks = len(self.efficientnet._blocks)
        for block_idx in range(num_blocks - 5, num_blocks):
            for param in self.efficientnet._blocks[block_idx].parameters():
                param.requires_grad = True
        
        num_ftrs = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_ftrs, num_classes)

        
        # Unfreeze the fc layer
        for param in self.efficientnet._fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.efficientnet(x)
        return x
    
    def extract_features(self, x):
        # Temporarily remove the classifier
        classifier = self.efficientnet._fc
        self.efficientnet._fc = nn.Identity()

        # Extract features
        x = self.efficientnet.extract_features(x)

        # Restore the classifier
        self.efficientnet._fc = classifier

        return x



if __name__ == "__main__":
    
    # Config
    name = 'test'
    image_size = 350
    dropout_rate = 0.5
    batch_size = 64
    epochs = 100
    image_path = 'D:/DATA/E621/'
    data_csv = 'D:/DATA/E621/source_images.csv'


    # Load the dataset
    print('Preparing Dataset...')
    train_dataset, val_dataset = create_datasets(image_path, data_csv, image_size)


    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)


    # Define model, loss function, and optimizer
    model = ME621_Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    summary(model, input_size=(batch_size, 3, image_size, image_size))
    
    
    # Continue Training?
    os.makedirs(f"{env}/models/", exist_ok=True)
    model_path = f"{env}/models/{name}.pt"
    if os.path.exists(model_path):
        model = torch.load(model_path)
        
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_samples = 0
        y_pred = []
        y_true = []

        for images, labels in tqdm_loop(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, pred = torch.max(outputs, 1)
            correct = (pred == labels).sum().item()
            train_acc += correct
            y_pred += pred.cpu().numpy().tolist()
            y_true += labels.cpu().numpy().tolist()
            total_samples += len(labels)

        train_loss = train_loss / total_samples
        train_acc = train_acc / total_samples

        print(f'Epoch: {epoch} Training Loss: {train_loss:.6f} Accuracy: {train_acc:.6f}')
        print(classification_report(y_true, y_pred))
        
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        total_samples = 0
        y_pred = []
        y_true = []

        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, pred = torch.max(outputs, 1)
            correct = (pred == labels).sum().item()
            val_acc += correct
            y_pred += pred.cpu().numpy().tolist()
            y_true += labels.cpu().numpy().tolist()
            total_samples += len(labels)

        val_loss = val_loss / total_samples
        val_acc = val_acc / total_samples

        print(f'Val Loss: {val_loss:.6f} Val Accuracy: {val_acc:.6f}')
        print(classification_report(y_true, y_pred))
            

    # Save the final model  
    torch.save(model, f"{env}/models/{name}.pt")
    
    

    
        
