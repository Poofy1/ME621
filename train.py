import torch
import os
from torchvision.transforms import ToPILImage
import plotly.express as px
import plotly.offline as pyo
import cv2
import random
import torch.nn as nn
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm as tqdm_loop
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import classification_report
from gradcam.utils import visualize_cam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from collections import Counter
from efficientnet_pytorch import EfficientNet
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import functional as F
import multiprocessing
import umap
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torchinfo import summary
from efficientnet_pytorch import EfficientNet


# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# Config
image_size = 224
weight_decay = 0 #1e-5
dropout_rate = 0.5
train_path = 'D:/DATA/E621/train/'
val_path = 'D:/DATA/E621/val/'

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

# Define transforms for data augmentation
transform = transforms.Compose([
    SquareResize(image_size),
    transforms.RandomRotation(degrees=90),
    transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.3)),
    transforms.RandomPerspective(distortion_scale=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.0001, 0.3)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.7, 1.1), contrast=(0.5, 1.15), saturation=(0.25, 1.5), hue=(-0.5, 0.5)),
    transforms.ToTensor(),
    StaticNoise(intensity_min=0.0, intensity_max=0.025),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# Define transforms for data augmentation
val_transform = transforms.Compose([
    SquareResize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths, self.labels = self._get_data()

    def _get_data(self):
        image_paths = []
        labels = []
        for label, class_name in enumerate(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image_paths.append(image_path)
                labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[index]
        
        if self.transform:
            img = self.transform(img)
            
            ###
            """# Reverse normalization
            denorm = lambda t: (t * 0.5) + 0.5
            inv_img = denorm(img.clone())

            # Save the image
            transformed_img_path = os.path.join("D:/DATA/E621", f"{index}.png")
            img_save = transforms.ToPILImage()(inv_img)
            img_save.save(transformed_img_path)"""
            ###

        return img, label


class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b0", dropout_rate=dropout_rate)
    
        # Freeze all layers in EfficientNet
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two blocks
        num_blocks = len(self.efficientnet._blocks)
        for block_idx in range(num_blocks - 2, num_blocks):
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


def train_model(name, continue_training, batch_size, num_epochs, ):
    multiprocessing.freeze_support()
 
    # Define model, loss function, and optimizer
    model = MyModel().to(device)

    # print out the model summary
    summary(model, input_size=(batch_size, 3, image_size, image_size))
    
        
    # Load the dataset
    train_dataset = MyDataset(train_path, transform = transform)
    val_dataset = MyDataset(val_path, transform = val_transform)

    # Calculate class counts and weights for the training set
    train_class_counts = Counter(train_dataset.labels)
    train_total_samples = len(train_dataset.labels)
    train_class_weights = [train_total_samples / train_class_counts[label] for label in train_dataset.labels]
    train_sampler = WeightedRandomSampler(train_class_weights, len(train_dataset))

    # Create the DataLoaders for the training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    # Define the weighted loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    
    # Continue Training?
    if continue_training:
        model = torch.load(f"{env}/models/{name}.pt")
        
    print("Starting Training...")
    for epoch in range(num_epochs):
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
    
    

    
        
