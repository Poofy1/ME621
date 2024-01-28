import torch
import os, ast, sys
import numpy as np
import random
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from data.image_proc import *
import torchvision.models as models
from tqdm import tqdm
import warnings
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
torch.backends.cudnn.benchmark = False
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ME621_Dataset(Dataset):
    def __init__(self, image_ids, image_paths, favorite_ids_list, max_bag_size=100):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.favorite_ids_list = favorite_ids_list
        self.max_bag_size = max_bag_size
        self.current_bag_index = 0  # To keep track of the current bag

    def __len__(self):
        return len(self.favorite_ids_list)

    def __getitem__(self, index):
        # Select a favorites list based on the current index
        current_favorite_set = set(self.favorite_ids_list[self.current_bag_index])

        # Initialize list for bag image paths and labels
        bag_img_paths = []
        labels = []

        # Fill the bag
        while len(bag_img_paths) < self.max_bag_size and current_favorite_set:
            if random.random() < 0.5:  # Assuming 50% chance for a favorite
                favorite_id = current_favorite_set.pop()
                favorite_index = self.image_ids.index(favorite_id)
                img_path = self.image_paths[favorite_index]
                bag_img_paths.append(img_path)  # Append path directly
                labels.append(1)  # Label 1 for favorites
            else:
                # Select non-favorite avoiding any in the current favorite set
                non_favorites = [img_id for img_id in self.image_ids if img_id not in current_favorite_set]
                if non_favorites:
                    non_favorite_id = random.choice(non_favorites)
                    non_favorite_index = self.image_ids.index(non_favorite_id)
                    img_path = self.image_paths[non_favorite_index]
                    bag_img_paths.append(img_path)  # Append path directly
                    labels.append(0)

        # Increment the bag index for the next call
        self.current_bag_index += 1

        return bag_img_paths, labels
    
def custom_collate(batch):
    img_list = []
    label_list = []
    for imgs, labels in batch:
        img_list.extend(imgs)  # Concatenate all image paths
        label_list.extend(labels)  # Concatenate all labels
    return img_list, label_list


def create_datasets(image_dir, user_dir, data_csv, output_dir, image_size):
    # Load the image dataset
    image_df = pd.read_csv(data_csv, usecols=['ID', 'Created Date', 'Saved Date', 'Source Width', 'Source Height', 'Upvotes', 'Downvotes', 'Favorites', 'Valid'])
    
    # Preprocess images and update DataFrame
    image_df = preprocess_and_save_images(image_df, image_dir, output_dir, image_size)
    
    # Create a set of valid image IDs for efficient lookup
    valid_image_ids = set(image_df['ID'].tolist())
    
    # Load the user dataset
    user_csv = pd.read_csv(user_dir, usecols=['ID', 'favorites', 'Valid'])
    
    # Convert 'favorites' from string representation of list to actual list
    user_csv['favorites'] = user_csv['favorites'].apply(ast.literal_eval)

    # Filter user favorites to only include valid image IDs
    user_csv['favorites'] = user_csv['favorites'].apply(lambda favs: [fav for fav in favs if fav in valid_image_ids])

    # Further filter out rows where 'favorites' has less than 5 items
    user_csv = user_csv[user_csv['favorites'].map(len) >= 5]
    print(f'Found {len(user_csv)} unique user favorites')

    # Split into train and val sets
    user_train = user_csv[user_csv['Valid'] == 0]
    user_val = user_csv[user_csv['Valid'] == 1]

    
    # Extract labels for train and validation sets
    train_labels = user_train['favorites'].values.tolist()
    val_labels = user_val['favorites'].values.tolist()

    # Create image paths and labels
    image_ids = image_df['ID'].tolist()
    image_paths = image_df['path'].tolist()
    
    
    

    


    # Create datasets
    train_dataset = ME621_Dataset(image_ids, image_paths, train_labels)
    val_dataset = ME621_Dataset(image_ids, image_paths, val_labels)

    
    #train_dataset.show_image(0)
    #train_dataset.show_image(1)
    #train_dataset.show_image(2)
    
    return train_dataset, val_dataset




class ME621_Model(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, lstm_hidden_size=128, lstm_layers=1):
        super(ME621_Model, self).__init__()
        self.efficientnet = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')

        # Remove the original classifier from EfficientNet
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # FC layer to transform features for LSTM
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.InstanceNorm1d(num_ftrs),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_ftrs,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,  # Assuming your input tensor to LSTM is of shape (batch, seq, feature)
            dropout=dropout_rate
        )
        
        # Initialize hidden state parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # Output layer
        self.output_layer = nn.Linear(lstm_hidden_size, num_classes)
        
    def init_hidden(self, batch_size):
        # Initialize hidden and cell states to zeros
        # For LSTM, hidden state is a tuple of (hidden_state, cell_state)
        hidden_state = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(next(self.parameters()).device)
        cell_state = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(next(self.parameters()).device)
        return (hidden_state, cell_state)
        

    def forward(self, x, hidden=None):
        # Process images through EfficientNet
        features = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(features)
        x = torch.flatten(x, 1)

        # Pass through FC layer
        fc_out = self.fc(x)

        # Reshape for LSTM
        lstm_out, hidden = self.lstm(fc_out.unsqueeze(1), hidden)

        # LSTM output to final layer
        lstm_out_last_step = lstm_out[:, -1, :]

        # Final output
        out = self.output_layer(lstm_out_last_step)
        out = torch.sigmoid(out).squeeze(-1)

        return out, hidden 





if __name__ == "__main__":
    
    # Config
    name = 'ME621_Fav_1'
    image_size = 350
    dropout_rate = 0.0
    batch_size = 1
    epochs = 50
    check_interval = 1000
    raw_image_path = 'D:/DATA/E621/images/'
    ready_images_path = f'F:/Temp_SSD_Data/ME621/'
    image_csv = 'D:/DATA/E621/source_images.csv'
    user_csv = 'D:/DATA/E621/source_users_testing.csv'


    # Load the dataset
    print('Preparing Dataset...')
    train_dataset, val_dataset = create_datasets(raw_image_path, user_csv, image_csv, ready_images_path, image_size)


    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=2, pin_memory=True)

    
    # Define model, loss function, and optimizer
    model = ME621_Model(num_classes=1, dropout_rate=dropout_rate).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #summary(model, input_size=[(batch_size, 3, image_size, image_size), (batch_size, 1)])
    
    
    # Continue Training?
    os.makedirs(f"{env}/models/", exist_ok=True)
    model_path = f"{env}/models/{name}.pt"
    iteration = 0
    lowest_val_loss = float('inf')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        iteration = checkpoint['iteration'] + 1  # Continue from the next epoch
        lowest_val_loss = checkpoint['val_loss']
        print(f"Model with val loss: {lowest_val_loss:.6f}, resuming from epoch {iteration}")
        
        
    # Define transforms
    train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
    val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])

    
    total_train_samples = len(train_dataloader)
    total_val_samples = len(val_dataloader)
    print(f'Total Train Samples: {total_train_samples}')
    print(f'Total Val Samples: {total_val_samples}')


    print("Starting Training...")
    model.train()
    train_loss = 0
    total_train_samples = 0
    local_loss = 0
    

    # Train loop
    for img_list, label_list in tqdm(train_dataloader):
        local_count = 0
        local_loss = 0
        hidden_layer = model.init_hidden(batch_size)

        for img_path, label in zip(img_list, label_list):
            
            # Format Data
            img = Image.open(img_path)
            img = train_transform(img).to(device).unsqueeze(0)
            label = torch.tensor([label], dtype=torch.float32).to(device)


            output, hidden_layer = model(img, hidden_layer)
            hidden_layer = tuple([state.detach() for state in hidden_layer])
            
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img.size(0)
            total_train_samples += img.size(0)
        
            local_loss += loss.item()
            local_count += 1
        
        print(f'Set of {local_count}, Loss: {(local_loss/local_count):.3f}')













    # Perform validation check at specified interval
    if iteration % check_interval == 0:
        model.eval()
        val_loss = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_img, val_label, reset_set in val_dataloader:
                val_img = val_img.to(device)
                val_label = val_label.to(device)

                val_output, hidden_layer = model(val_img, hidden_layer, reset_state=False)
                batch_loss = criterion(val_output, val_label)
                val_loss += batch_loss.item() * val_img.size(0)
                total_val_samples += val_img.size(0)

        val_loss /= total_val_samples
        print(f'\Iteration: [{iteration}] [{int((iteration / total_train_samples) * 100)}%] | Train Loss: {train_loss / total_train_samples:.3f} | Val Loss: {val_loss:.3f}')
        train_loss = 0
        total_train_samples = 1
        model.train()

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            state = {
                'iteration': iteration,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': lowest_val_loss
            }
            torch.save(state, f"{env}/models/{name}.pt")
            print(f"Saved Model")
                
                