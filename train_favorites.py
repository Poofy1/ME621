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
        self.current_bag_index = 0

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

        # Fill the bag with alternating negative and positive images
        while len(bag_img_paths) < self.max_bag_size and current_favorite_set:
            # Add a negative image (not in the current favorite set)
            non_favorites = [img_id for img_id in self.image_ids if img_id not in current_favorite_set and img_id < max_favorite_id]
            if non_favorites:
                non_favorite_id = random.choice(non_favorites)
                non_favorite_index = self.image_ids.index(non_favorite_id)
                img_path = self.image_paths[non_favorite_index]
                bag_img_paths.append(img_path)
                labels.append(0)  # Label 0 for non-favorites

                # Add a positive image (from the current favorite set)
                favorite_id = current_favorite_set.pop()
                favorite_index = self.image_ids.index(favorite_id)
                img_path = self.image_paths[favorite_index]
                bag_img_paths.append(img_path)
                labels.append(1)  # Label 1 for favorites

        # Increment the bag index for the next call
        self.current_bag_index = (self.current_bag_index + 1) % len(self.favorite_ids_list)

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

    # Further filter out rows where 'favorites' has less than x items
    user_csv = user_csv[user_csv['favorites'].map(len) >= 16]
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
    
    #train_labels = [train_labels[0]]
    #train_labels = [val_labels[0]]

    # Create datasets
    train_dataset = ME621_Dataset(image_ids, image_paths, train_labels)
    val_dataset = ME621_Dataset(image_ids, image_paths, val_labels)

    
    #train_dataset.show_image(0)
    #train_dataset.show_image(1)
    #train_dataset.show_image(2)
    
    return train_dataset, val_dataset






class ME621_Model(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, embedding_dim=512, interpreter_fc_size=128):
        super(ME621_Model, self).__init__()

        # CNN feature extractor
        self.efficientnet = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')
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

    def forward(self, image, mode='prediction'):
        # Extract features from the image using the CNN
        image_features = self.efficientnet.features(image)
        x = self.efficientnet.avgpool(image_features)
        x = torch.flatten(x, 1)

        # Transform the image features to match embedding size
        x_transformed = self.feature_transform(x)

        if mode == 'siamese':
            # Assuming images come in pairs: [neg, pos, neg, pos, ...]
            neg_features = x_transformed[0::2]  # Negative features
            pos_features = x_transformed[1::2]  # Positive features

            # Update the preference embedding
            self.update_preference_embedding(pos_features, neg_features)
            
            return neg_features, pos_features

        elif mode == 'prediction':
            # Combine transformed image features with user's preference embedding
            preference_expanded = self.preference_embedding.unsqueeze(0).expand(x_transformed.size(0), -1)
            combined = torch.cat((preference_expanded, x_transformed), dim=1) 

            # Pass the combined features through the intermediate FC layer
            intermediate_output = self.intermediate_fc(combined)

            # Predict and return the likelihood of the user liking the image
            prediction = torch.sigmoid(self.prediction_layer(intermediate_output)).squeeze(-1)

            return prediction
    
    def reset_preference_embedding(self):
        """
        Reset the preference embedding to its initial state (e.g., all zeros).
        """
        self.preference_embedding.data.fill_(0.0)

    def update_preference_embedding(self, pos_image_features, neg_image_features, alpha=0.01):
        """
        Update the preference embedding based on a pair of images.
        - pos_image_features: extracted features of the liked (positive) images
        - neg_image_features: extracted features of the not liked (negative) images
        - alpha: learning rate for updating the embedding
        """

        # Update direction is +1 for positive (liked) and -1 for negative (not liked)
        pos_update_direction = 1
        neg_update_direction = -1

        # Calculate updates for both positive and negative images
        pos_update = alpha * (pos_update_direction * (pos_image_features - self.preference_embedding))
        neg_update = alpha * (neg_update_direction * (neg_image_features - self.preference_embedding))

        # Combine updates and apply to the preference embedding
        total_update = pos_update + neg_update
        self.preference_embedding.data += total_update

        # Normalize the preference embedding
        norm = self.preference_embedding.data.norm(p=2, dim=0, keepdim=True)
        self.preference_embedding.data = self.preference_embedding.data.div(norm.clamp(min=1e-8))



def contrastive_loss(neg_features, pos_features, margin=1.0):
    """
    Calculates a more complete contrastive loss between negative and positive features.

    Args:
    - neg_features (Tensor): Features from the negative (dissimilar) images.
    - pos_features (Tensor): Features from the positive (similar) images.
    - margin (float): The margin for dissimilar images.

    Returns:
    - Tensor: The contrastive loss.
    """
    # Calculate Euclidean distances
    positive_loss = torch.norm(pos_features - neg_features, p=2, dim=1)
    negative_distance = torch.norm(pos_features + neg_features, p=2, dim=1)

    # Loss for negative pairs (dissimilar) - should be large
    negative_loss = torch.relu(margin - negative_distance)

    # Combine losses
    combined_loss = positive_loss + negative_loss

    # Average loss over the batch
    return combined_loss.mean()



def val_check(model, total_train_samples, total_train_correct):
    model.eval()
    val_loss = 0
    total_val_samples = 0
    total_val_correct = 0

    with torch.no_grad():
        for val_img_list, val_label_list in val_dataloader:
            model.reset_preference_embedding()
            val_img_list = val_img_list[0]
            val_label_list = val_label_list[0]
            
            # Select a random number for siamese mode
            siamese_size = random.randint(4, 8) * 2
            siamese_imgs = val_img_list[0][:siamese_size]
            siamese_labels = val_label_list[0][:siamese_size]
            pred_imgs = val_img_list[0][siamese_size:]
            pred_labels = val_label_list[0][siamese_size:]
            
            # Siamese Phase

            for i in range(0, len(siamese_imgs), batch_size):
                # Get mini-batch
                mini_batch_img_list = siamese_imgs[i:i + batch_size]
                mini_batch_label_list = siamese_labels[i:i + batch_size]

                # Process mini-batch
                batch_imgs = [train_transform(Image.open(img_path)).unsqueeze(0) for img_path in mini_batch_img_list]
                batch_labels = torch.tensor(mini_batch_label_list, dtype=torch.float32)

                # Convert lists to tensors
                batch_imgs = torch.cat(batch_imgs, dim=0).to(device) 
                batch_labels = batch_labels.to(device)
                
                # Model output and loss calculation
                neg_features, pos_features = model(batch_imgs, mode='siamese')
                loss = contrastive_loss(neg_features, pos_features)

                # Update training statistics
                val_loss += loss.item() * batch_imgs.size(0)
                total_val_samples += batch_imgs.size(0)
            
            # Pred Phase
            
            for i in range(0, len(pred_imgs), batch_size):
                # Get mini-batch
                mini_batch_img_list = pred_imgs[i:i + batch_size]
                mini_batch_label_list = pred_labels[i:i + batch_size]

                # Process mini-batch
                batch_imgs = [train_transform(Image.open(img_path)).unsqueeze(0) for img_path in mini_batch_img_list]
                batch_labels = torch.tensor(mini_batch_label_list, dtype=torch.float32)

                # Convert lists to tensors
                batch_imgs = torch.cat(batch_imgs, dim=0).to(device) 
                batch_labels = batch_labels.to(device)
                
                # Model output and loss calculation
                output = model(batch_imgs, mode='prediction')
                loss = criterion(output, batch_labels)
                
                # Update training statistics
                val_loss += loss.item() * batch_imgs.size(0)
                total_val_samples += batch_imgs.size(0)

    val_loss /= total_val_samples
    train_accuracy = total_train_correct / total_train_samples
    val_accuracy = total_val_correct / total_val_samples

    print(f'\nEpoch: [{epoch}] | Train Loss: {train_loss / total_train_samples:.5f} | Train Acc: {train_accuracy:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {val_accuracy:.5f}')

    return val_loss
    
        


if __name__ == "__main__":
    
    # Config
    name = 'ME621_Fav_1'
    image_size = 350
    dropout_rate = 0.0
    batch_size = 8
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
    optimizer = optim.Adam(model.parameters())
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


    print("Starting Training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        total_train_samples = 0
        total_train_correct = 0
        
        batch_counter = 0
        

        # Train loop
        for img_list, label_list in tqdm(train_dataloader):
            model.reset_preference_embedding()
            img_list = img_list[0]
            label_list = label_list[0]
            
            # Select a random number for siamese mode
            siamese_size = random.randint(4, 8) * 2
            siamese_imgs = img_list[0][:siamese_size]
            siamese_labels = label_list[0][:siamese_size]
            pred_imgs = img_list[0][siamese_size:]
            pred_labels = label_list[0][siamese_size:]
            
            # Siamese Phase

            for i in range(0, len(siamese_imgs), batch_size):
                # Get mini-batch
                mini_batch_img_list = siamese_imgs[i:i + batch_size]
                mini_batch_label_list = siamese_labels[i:i + batch_size]

                # Process mini-batch
                batch_imgs = [train_transform(Image.open(img_path)).unsqueeze(0) for img_path in mini_batch_img_list]
                batch_labels = torch.tensor(mini_batch_label_list, dtype=torch.float32)

                # Convert lists to tensors
                batch_imgs = torch.cat(batch_imgs, dim=0).to(device) 
                batch_labels = batch_labels.to(device)
                
                # Model output and loss calculation
                neg_features, pos_features = model(batch_imgs, mode='siamese')
                loss = contrastive_loss(neg_features, pos_features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update training statistics
                train_loss += loss.item() * batch_imgs.size(0)
                total_train_samples += batch_imgs.size(0)
            
            # Pred Phase
            
            for i in range(0, len(pred_imgs), batch_size):
                # Get mini-batch
                mini_batch_img_list = pred_imgs[i:i + batch_size]
                mini_batch_label_list = pred_labels[i:i + batch_size]

                # Process mini-batch
                batch_imgs = [train_transform(Image.open(img_path)).unsqueeze(0) for img_path in mini_batch_img_list]
                batch_labels = torch.tensor(mini_batch_label_list, dtype=torch.float32)

                # Convert lists to tensors
                batch_imgs = torch.cat(batch_imgs, dim=0).to(device) 
                batch_labels = batch_labels.to(device)
                
                # Model output and loss calculation
                output = model(batch_imgs, mode='prediction')
                loss = criterion(output, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update training statistics
                train_loss += loss.item() * batch_imgs.size(0)
                total_train_samples += batch_imgs.size(0)
                
                
                
                
                


            batch_counter += 1


            # Perform validation check at specified interval
            if batch_counter % check_interval == 0:
                val_loss = val_check(model, total_train_samples, total_train_correct)
                model.train()
                total_train_correct = 0
                train_loss = 0
                total_train_samples = 0
                
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
                
                
        val_loss = val_check(model, total_train_samples, total_train_correct)
        model.train()
        total_train_correct = 0
        train_loss = 0
        total_train_samples = 0
        
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
                    