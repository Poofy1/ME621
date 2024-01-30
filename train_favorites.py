import torch
import os, ast
import numpy as np
import random
import torch.nn as nn
import pandas as pd
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

        # Find the maximum ID value in the current favorite list
        max_favorite_id = max(current_favorite_set)

        # Initialize list for bag image paths and labels
        bag_img_paths = []
        labels = []

        # Fill the bag
        while len(bag_img_paths) < self.max_bag_size and current_favorite_set:
            if random.random() < 0.5:  # Assuming 50% chance for a favorite
                favorite_id = current_favorite_set.pop()
                favorite_index = self.image_ids.index(favorite_id)
                img_path = self.image_paths[favorite_index]
                bag_img_paths.append(img_path)
                labels.append(1)  # Label 1 for favorites
            else:
                # Select non-favorite avoiding any in the current favorite set and ensuring ID is less than max_favorite_id
                non_favorites = [img_id for img_id in self.image_ids if img_id not in current_favorite_set and img_id < max_favorite_id]
                if non_favorites:
                    non_favorite_id = random.choice(non_favorites)
                    non_favorite_index = self.image_ids.index(non_favorite_id)
                    img_path = self.image_paths[non_favorite_index]
                    bag_img_paths.append(img_path)
                    labels.append(0)

        # Increment the bag index for the next call
        self.current_bag_index += 1

        return bag_img_paths, labels
    
    
def custom_collate(batch):
    img_list = []
    label_list = []
    for imgs, labels in batch:
        img_list.append(imgs)
        label_list.append(labels)
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



class ME621_Model_LSTM(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, lstm_hidden_size=256, lstm_layers=2, interpreter_fc_size=128):
        super(ME621_Model_LSTM, self).__init__()
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
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Initialize hidden state parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # Intermediate FC layer to interpret LSTM memory
        self.interpreter_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size + num_ftrs, interpreter_fc_size), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output_layer = nn.Linear(interpreter_fc_size, num_classes)
        
    def init_hidden(self, batch_size):
        # Initialize hidden and cell states to zeros
        hidden_state = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(next(self.parameters()).device)
        cell_state = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(next(self.parameters()).device)
        return (hidden_state, cell_state)

    def forward(self, x, ground_truth, hidden=None):
        # Process images through EfficientNet
        features = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(features)
        x = torch.flatten(x, 1)

        # Incorporate the additional binary input
        # Ensure the additional_input is in the right shape (batch_size, 1)
        ground_truth = ground_truth.unsqueeze(1)
        
        # Concatenate the CNN output and the additional binary input
        combined_input = torch.cat([x, ground_truth], dim=1)

        # Pass the combined input through the FC layer
        # Modify the input size of the first layer of self.fc to match the new combined_input size
        fc_out = self.fc(combined_input)

        # Reshape for LSTM
        lstm_out, hidden = self.lstm(fc_out.unsqueeze(1), hidden)

        # LSTM output to interpreter FC layer
        lstm_out_last_step = lstm_out[:, -1, :]
        
        # Concatenate LSTM output and FC output
        combined_out = torch.cat((lstm_out_last_step, fc_out), dim=1)

        # Pass the concatenated output to the interpreter FC layer
        interpreter_out = self.interpreter_fc(combined_out)

        # Final output
        out = self.output_layer(interpreter_out)
        out = torch.sigmoid(out).squeeze(-1)

        return out, hidden





class ME621_Model(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, embedding_dim=512, interpreter_fc_size=128):
        super(ME621_Model, self).__init__()

        # CNN feature extractor
        self.efficientnet = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')
        
        # Remove the original classifier from EfficientNet
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        
        # New layer to match feature size to embedding size
        self.feature_transform = nn.Linear(num_ftrs, embedding_dim)

        # Preference Embedding
        self.preference_embedding = nn.Parameter(torch.zeros(embedding_dim))

        # Intermediate FC layer
        self.intermediate_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, interpreter_fc_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Prediction Layer
        self.prediction_layer = nn.Linear(interpreter_fc_size, num_classes)

    def forward(self, image, image_label=None):
        # Extract features from the image using the CNN
        image_features = self.efficientnet.features(image)
        x = self.efficientnet.avgpool(image_features)
        x = torch.flatten(x, 1)


        # Transform the image features to match embedding size
        x_transformed = self.feature_transform(x)

        # Make sure preference embedding is compatible for concatenation
        preference_expanded = self.preference_embedding.unsqueeze(0).expand(x_transformed.size(0), -1)

        # Combine transformed image features with user's preference embedding
        combined = torch.cat((preference_expanded, x_transformed), dim=1) 

        # Pass the combined features through the intermediate FC layer
        intermediate_output = self.intermediate_fc(combined)

        # Predict and return the likelihood of the user liking the image
        prediction = torch.sigmoid(self.prediction_layer(intermediate_output)).squeeze(-1)

        # Update preference embedding if required
        if image_label is not None:
            self.update_preference_embedding(x_transformed, image_label)  # Use flattened features for update

        return prediction
    
    def reset_preference_embedding(self):
        """
        Reset the preference embedding to its initial state (e.g., all zeros).
        """
        self.preference_embedding.data.fill_(0.0)

    def update_preference_embedding(self, image_features, image_label, alpha=0.1):
        """
        Update the preference embedding based on the labels of the images.
        - image_features: extracted features of the current images (with batch size)
        - image_label: the labels of the images (1 for liked, 0 for not liked)
        - alpha: learning rate for updating the embedding
        """

        # Calculate update direction (1 if liked, -1 if not liked)
        update_direction = (2 * image_label - 1)

        # Calculate the average update across the batch
        average_update = alpha * (update_direction * (image_features - self.preference_embedding.data)).mean(dim=0)

        # Apply the average update to the preference embedding
        self.preference_embedding.data += average_update





if __name__ == "__main__":
    
    # Config
    name = 'ME621_Fav_1'
    image_size = 350
    dropout_rate = 0.0
    batch_size = 4
    epochs = 50
    check_interval = 100
    raw_image_path = 'D:/DATA/E621/images/'
    ready_images_path = f'F:/Temp_SSD_Data/ME621/'
    image_csv = 'D:/DATA/E621/source_images.csv'
    user_csv = 'D:/DATA/E621/source_users_testing.csv'


    # Load the dataset
    print('Preparing Dataset...')
    train_dataset, val_dataset = create_datasets(raw_image_path, user_csv, image_csv, ready_images_path, image_size)


    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=2, pin_memory=True)

    
    # Define model, loss function, and optimizer
    model = ME621_Model(num_classes=1, dropout_rate=dropout_rate).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #summary(model, input_size=[(batch_size, 3, image_size, image_size), (batch_size,)])
    
    
    # Continue Training?
    os.makedirs(f"{env}/models/", exist_ok=True)
    model_path = f"{env}/models/{name}.pt"
    start_epoch = 0
    lowest_val_loss = float('inf')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
        lowest_val_loss = checkpoint['val_loss']
        print(f"Model with val loss: {lowest_val_loss:.6f}, resuming from epoch {start_epoch}")
        
        
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
    
    total_samples = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size
    total_batches = -(-total_samples // 1)


    print("Starting Training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        total_train_samples = 0
        batch_counter = 0
        

        # Train loop
        for img_list, label_list in tqdm(train_dataloader):
            img_list = img_list[0]
            label_list = label_list[0]
            
            model.reset_preference_embedding()
            seen_labels = 1
            
            # Process in mini-batches
            for i in range(0, len(img_list), batch_size):
                # Slice the img_list and label_list to get a mini-batch
                mini_batch_img_list = img_list[i:i + batch_size]
                mini_batch_label_list = label_list[i:i + batch_size]

                # Process the mini-batch
                batch_imgs = []
                batch_labels = []
                for img_path, label in zip(mini_batch_img_list, mini_batch_label_list):
                    # Format Data
                    img = Image.open(img_path)
                    img = train_transform(img).unsqueeze(0)
                    label = torch.tensor([label], dtype=torch.float32)
                    batch_imgs.append(img)
                    batch_labels.append(label)
                
                # Convert lists to tensors
                batch_imgs = torch.cat(batch_imgs, dim=0).to(device) 
                batch_labels = torch.cat(batch_labels, dim=0).to(device)
                
                if seen_labels > 0:
                    input_label = label
                else:
                    input_label = None

                output = model(img, input_label)
                
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * img.size(0)
                total_train_samples += img.size(0)
                seen_labels -= 1

            batch_counter += 1


            # Perform validation check at specified interval
            if batch_counter % check_interval == 0:
                model.eval()
                val_loss = 0
                total_val_samples = 0

                with torch.no_grad():
                    for val_img_list, val_label_list in val_dataloader:
                        
                        model.reset_preference_embedding()
                        seen_labels = 5
                        
                        for img_path, label in zip(val_img_list, val_label_list):
                            # Format Data
                            img = Image.open(img_path)
                            img = train_transform(img).to(device).unsqueeze(0)
                            label = torch.tensor([label], dtype=torch.float32).to(device)

                            if seen_labels > 0:
                                input_label = label
                            else:
                                input_label = None
                    
                            output = model(img, input_label)

                            loss = criterion(output, label)

                            val_loss += loss.item() * img.size(0)
                            total_val_samples += img.size(0)
                            
                            seen_labels -= 1

                val_loss /= total_val_samples
                print(f'\nEpoch: [{epoch}] [{int((batch_counter / total_batches) * 100)}%] | Train Loss: {train_loss / total_train_samples:.5f} | Val Loss: {val_loss:.5f}')
                train_loss = 0
                total_train_samples = 1
                model.train()

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    state = {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_loss': lowest_val_loss
                    }
                    torch.save(state, f"{env}/models/{name}.pt")
                    print(f"Saved Model")
                    
                    