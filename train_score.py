import torch
import os
import random
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import ast
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from multiprocessing import Pool
import warnings
from torchinfo import summary
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


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

def file_exists(row, image_dir):
    img_path = os.path.join(image_dir, f"{row}.png")
    return os.path.exists(img_path), img_path


def process_single_image(img_path, output_dir, resize_and_pad):
    try:
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)
        image = Image.open(img_path)
        image = resize_and_pad(image)
        image.save(output_path)
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

def preprocess_and_save_images(data, output_dir, image_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    resize_and_pad = SquareResize(image_size)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, img_name, output_dir, resize_and_pad): img_name for img_name in data}

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                future.result() 
                pbar.update()

def create_datasets(image_dir, data_csv, ready_images_path, image_size):
    # Load the dataset
    dataset_df = pd.read_csv(data_csv, usecols=['ID', 'Upvotes', 'Downvotes', 'Favorites', 'Valid'])

    
    # Prepare ID list for multiprocessing
    ids = dataset_df['ID'].tolist()

    # Use multiprocessing Pool to check file existence in parallel
    with Pool() as pool:
        results = list(tqdm(pool.starmap(file_exists, [(id, image_dir) for id in ids]), total=len(ids)))

    # Assign results to DataFrame
    dataset_df['exists'], dataset_df['path'] = zip(*results)

    # Split into train and val based on 'Valid' and 'exists' columns
    train_df = dataset_df[(dataset_df['Valid'] == 0) & (dataset_df['exists'])]
    val_df = dataset_df[(dataset_df['Valid'] == 1) & (dataset_df['exists'])]
    
    # Create image paths and labels
    train_image_paths = train_df['path'].tolist()
    val_image_paths = val_df['path'].tolist()
    
    train_labels = train_df[['Upvotes', 'Downvotes', 'Favorites']].values.tolist()
    val_labels = val_df[['Upvotes', 'Downvotes', 'Favorites']].values.tolist()
    
    #Cropping images
    preprocess_and_save_images(train_image_paths, ready_images_path, image_size)
    preprocess_and_save_images(val_image_paths, ready_images_path, image_size)


    # Define transforms
    train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=(0.8, 1.1), contrast=(0.8, 1.1), saturation=(0.8, 1.1), hue=(-0.1, 0.1)),
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
    
    #train_dataset.show_image(0)
    #train_dataset.show_image(1)
    #train_dataset.show_image(2)

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
        label = torch.tensor(label, dtype=torch.float32).to(device)

        # Transform
        img = self.transform(img).to(device)

        return img, label
    
    def unnormalize(self, tensor):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unnormalized = tensor.clone()
        for t, m, s in zip(unnormalized, mean, std):
            t.mul_(s).add_(m)  # Undo the normalization
        return unnormalized
    
    def show_image(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")

        # Apply the transform
        if self.transform:
            img_transformed = self.transform(img)
            img_transformed = self.unnormalize(img_transformed)  # Unnormalize
            img_show = transforms.ToPILImage()(img_transformed.cpu()).convert("RGB")
        else:
            img_show = img

        plt.imshow(img_show)
        plt.title(f"Image at index {index}")
        plt.pause(5) 
    

class ME621_Model(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ME621_Model, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b0", dropout_rate=dropout_rate)
    
        """# Freeze all layers in EfficientNet
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two blocks
        num_blocks = len(self.efficientnet._blocks)
        for block_idx in range(num_blocks - 5, num_blocks):
            for param in self.efficientnet._blocks[block_idx].parameters():
                param.requires_grad = True"""
        
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
    image_size = 300
    dropout_rate = 0.0
    batch_size = 32
    epochs = 100
    image_path = 'D:/DATA/E621/images/'
    ready_images_path = f'D:/DATA/E621/images_{image_size}/'
    data_csv = 'D:/DATA/E621/source_images.csv'


    # Load the dataset
    print('Preparing Dataset...')
    train_dataset, val_dataset = create_datasets(image_path, data_csv, ready_images_path, image_size)


    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, persistent_workers=True)

    
    # Define model, loss function, and optimizer
    model = ME621_Model(num_classes=3, dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    summary(model, input_size=(batch_size, 3, image_size, image_size))
    
    
    # Continue Training?
    os.makedirs(f"{env}/models/", exist_ok=True)
    model_path = f"{env}/models/{name}.pt"
    start_epoch = 0  # Default starting epoch
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
        lowest_val_loss = checkpoint['val_loss']
        print(f"Model and optimizer loaded with validation loss: {lowest_val_loss:.6f}, resuming from epoch {start_epoch}")

        
    lowest_val_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        total_mse = 0
        train_acc = 0
        total_samples = 0
        y_pred = []
        y_true = []

        for images, labels in tqdm(train_dataloader):
            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            # For regression, calculate metrics like MSE instead of accuracy
            mse = nn.MSELoss()
            current_mse = mse(outputs, labels).item()
            total_mse += current_mse * images.size(0)

        total_samples = len(train_dataloader.dataset)
        train_loss = train_loss / total_samples
        train_mse = total_mse / total_samples

        print(f'Epoch: {epoch} Training Loss: {train_loss:.6f} MSE: {train_mse:.6f}')

        
        # Validation loop
        model.eval()
        val_loss = 0
        total_mse = 0
        total_samples = 0

        with torch.no_grad():  # No need to calculate gradients during validation
            for images, labels in tqdm(val_dataloader):
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                # For regression, calculate metrics like MSE instead of accuracy
                mse = nn.MSELoss()
                current_mse = mse(outputs, labels).item()
                total_mse += current_mse * images.size(0)

        total_samples = len(val_dataloader.dataset)
        val_loss = val_loss / total_samples
        val_mse = total_mse / total_samples

        print(f'Epoch: {epoch} Val Loss: {val_loss:.6f} Val MSE: {val_mse:.6f}')




        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': lowest_val_loss
            }
            torch.save(state, f"{env}/models/{name}.pt")
            print(f"Model, optimizer, and epoch saved with improved validation loss: {lowest_val_loss:.6f}")
