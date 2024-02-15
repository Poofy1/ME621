import torch
import os, ast
import numpy as np
from arch.model_favorite import *
import random
import torch.nn as nn
import pandas as pd
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from data.image_proc import *
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

class ME621_Siamese_Dataset(Dataset):
    def __init__(self, image_ids, image_paths, favorite_ids_list, transform=None):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.favorite_ids_list = favorite_ids_list
        self.transform = transform
        # Pre-calculate the cumulative length of favorite lists for indexing
        self.cumulative_lengths = [len(fav_list) for fav_list in favorite_ids_list]
        self.total_length = sum(self.cumulative_lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        # Determine which favorite list the index refers to
        list_index = 0
        while index >= self.cumulative_lengths[list_index]:
            index -= self.cumulative_lengths[list_index]
            list_index += 1

        # Now index is the position within the selected favorite list
        selected_list = self.favorite_ids_list[list_index]
        positive_id = selected_list[index]
        positive_index = self.image_ids.index(positive_id)
        positive_img_path = self.image_paths[positive_index]

        # Prepare for negative image selection
        random_favorite_set = set(selected_list)
        max_favorite_id = max(random_favorite_set)

        non_favorites = [img_id for img_id in self.image_ids if img_id not in random_favorite_set and img_id < max_favorite_id]
        non_favorite_id = random.choice(non_favorites)
        non_favorite_index = self.image_ids.index(non_favorite_id)
        negative_img_path = self.image_paths[non_favorite_index]

        # Apply transformations if any
        positive_img = self.transform(Image.open(positive_img_path)) if self.transform else Image.open(positive_img_path)
        negative_img = self.transform(Image.open(negative_img_path)) if self.transform else Image.open(negative_img_path)

        return positive_img, negative_img


def custom_collate(batch):
    img_list = []
    label_list = []
    for imgs, labels in batch:
        img_list.append(imgs)
        label_list.append(labels)
    return img_list[0], label_list[0]


def Siamese_Collate(batch):
    imgs = []
    for pos_img, neg_img in batch:
        imgs.append(pos_img.unsqueeze(0))  # Add batch dimension
        imgs.append(neg_img.unsqueeze(0))  # Add batch dimension
    imgs_tensor = torch.cat(imgs, dim=0)  # Concatenate along the batch dimension
    return imgs_tensor

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
    
    return image_ids, image_paths, train_labels, val_labels








class LossMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cumulative_loss = 0.0
        self.total_samples = 0
        self.total_correct = 0

    def update(self, loss, outputs=None, labels=None, batch_size=None):
        # Determine batch size
        if batch_size is None:
            if outputs is not None:
                batch_size = outputs.size(0)
            elif labels is not None:
                batch_size = len(labels)
            else:
                raise ValueError("Batch size could not be determined. Please provide outputs, labels, or batch_size explicitly.")
        
        self.cumulative_loss += loss.item() * batch_size
        self.total_samples += batch_size

        # Update accuracy only if outputs and labels are provided
        if outputs is not None and labels is not None:
            predicted_labels = (outputs > 0.5).float()
            self.total_correct += (predicted_labels == labels).sum().item()

    def average_loss(self):
        return self.cumulative_loss / self.total_samples if self.total_samples > 0 else 0

    def accuracy(self):
        return self.total_correct / self.total_samples if self.total_samples > 0 else 0

    
    
def PackageSiameseBatch(siamese_imgs, i):
    # Get mini-batch
    mini_batch_img_list = siamese_imgs[i:i + batch_size]
    # Process mini-batch
    batch_imgs = [train_transform(Image.open(img_path)).unsqueeze(0) for img_path in mini_batch_img_list]
    # Convert lists to tensors
    return torch.cat(batch_imgs, dim=0).to(device) 


def val_check(model, train_siamese_metrics, train_pred_metrics, training_mode):
    model.eval()
    val_siamese_metrics = LossMetrics()
    val_pred_metrics = LossMetrics()

    with torch.no_grad():
        for val_img_list, val_label_list in val_dataloader:
            
            if training_mode == 'Siamese':
                # Siamese Phase
                for i in range(0, len(val_img_list), batch_size):
                    # Get mini-batch
                    batch_imgs = PackageSiameseBatch(val_img_list, i)
                    
                    # Model output and loss calculation
                    neg_features, pos_features = model(batch_imgs, mode='siamese', train = True)
                    loss = contrastive_loss(neg_features, pos_features)

                    # Update val statistics
                    val_siamese_metrics.update(loss, None, None, len(batch_imgs))

            elif training_mode == 'Pred':
                # Siamese Phase
                model.reset_preference_embedding()
                
                # Select a random number for siamese mode
                siamese_size = random.randint(4, 8) * 2
                siamese_imgs = val_img_list[:siamese_size]
                pred_imgs = val_img_list[siamese_size:]
                pred_labels = val_label_list[siamese_size:]

                for i in range(0, len(siamese_imgs), batch_size):
                    # Get mini-batch
                    batch_imgs = PackageSiameseBatch(siamese_imgs, i)
                    
                    # Model output and loss calculation
                    _, _ = model(batch_imgs, mode='siamese')
                
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
                    
                    # Update val statistics
                    val_pred_metrics.update(loss, output, batch_labels, len(batch_imgs))

    # Prepare values for printing
    train_siamese_loss = train_siamese_metrics.average_loss()
    train_pred_loss = train_pred_metrics.average_loss()
    train_accuracy = train_pred_metrics.accuracy()
    val_siamese_loss = val_siamese_metrics.average_loss()
    val_pred_loss = val_pred_metrics.average_loss()
    val_accuracy = val_pred_metrics.accuracy()

    # Print metrics
    print(f'nEpoch: {epoch} \nTS Loss: {train_siamese_loss:.5f} | TP Loss: {train_pred_loss:.5f} | T Acc: {train_accuracy:.5f} \nVS Loss: {val_siamese_loss:.5f} | VP Loss: {val_pred_loss:.5f} | V Acc: {val_accuracy:.5f}')

    return val_pred_loss
    
        


if __name__ == "__main__":
    
    # Config
    name = 'ME621_Fav_1'
    image_size = 350
    dropout_rate = 0.0
    batch_size = 8
    epochs = 50
    check_interval = 1000
    raw_image_path = 'D:/DATA/E621/images/'
    ready_images_path = f'F:/Temp_SSD_Data/ME621/'
    image_csv = 'D:/DATA/E621/source_images.csv'
    user_csv = 'D:/DATA/E621/source_users_testing.csv'

    training_mode = 'Siamese'
    
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

    # Load the dataset
    print('Preparing Dataset...')
    image_ids, image_paths, train_labels, val_labels = create_datasets(raw_image_path, user_csv, image_csv, ready_images_path, image_size)

    if training_mode == 'Siamese':
        train_dataset = ME621_Siamese_Dataset(image_ids, image_paths, train_labels, train_transform)
        val_dataset = ME621_Siamese_Dataset(image_ids, image_paths, val_labels, val_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Siamese_Collate, num_workers=2, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Siamese_Collate, num_workers=2, pin_memory=True)
    else:
        train_dataset = ME621_Dataset(image_ids, image_paths, train_labels)
        val_dataset = ME621_Dataset(image_ids, image_paths, val_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate, num_workers=2, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=2, pin_memory=True)

    
    # Define model, loss function, and optimizer
    model = ME621_Model(num_classes=1, dropout_rate=dropout_rate).to(device)
    criterion = nn.BCELoss()
    
    siamese_params = list(model.efficientnet.parameters()) + list(model.feature_transform.parameters())
    prediction_params = list(model.intermediate_fc.parameters()) + list(model.prediction_layer.parameters())
    optimizer_siamese = optim.Adam(siamese_params)
    optimizer_prediction = optim.Adam(prediction_params)
    #summary(model, input_size=[(batch_size, 3, image_size, image_size), (batch_size,)])
    
    
    # Continue Training?
    os.makedirs(f"{env}/models/", exist_ok=True)
    model_path = f"{env}/models/{name}.pt"
    start_epoch = 0
    lowest_val_loss = float('inf')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer_siamese.load_state_dict(checkpoint['optimizer_siamese'])
        optimizer_prediction.load_state_dict(checkpoint['optimizer_prediction'])
        start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
        lowest_val_loss = checkpoint['val_loss']
        print(f"Model with val loss: {lowest_val_loss:.6f}, resuming from epoch {start_epoch}")
        
        
    

    
    total_train_samples = len(train_dataloader)
    total_val_samples = len(val_dataloader)
    print(f'Total Train Samples: {total_train_samples}')
    print(f'Total Val Samples: {total_val_samples}')
    
    total_samples = len(train_dataloader.dataset)


    
    
    print("Starting Training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_siamese_metrics = LossMetrics()
        train_pred_metrics = LossMetrics()
        batch_counter = 0
        

        if training_mode == 'Siamese':
        
            for batch_imgs in tqdm(train_dataloader):
                batch_imgs = batch_imgs.to(device) 
                
                # Model output and loss calculation
                optimizer_siamese.zero_grad()
                neg_features, pos_features = model(batch_imgs, mode='siamese', train = True)
                loss = contrastive_loss(neg_features, pos_features)
                print(loss)
                loss.backward()
                optimizer_siamese.step()

                # Update training statistics
                train_siamese_metrics.update(loss, None, None, len(batch_imgs)) 
                    
        elif training_mode == 'Pred':
            
            for img_list, label_list in tqdm(train_dataloader):
                # Siamese Phase
                model.reset_preference_embedding()
                
                # Select a random number for siamese mode
                siamese_size = random.randint(4, 16) * 2
                siamese_imgs = img_list[:siamese_size]
                pred_imgs = img_list[siamese_size:]
                pred_labels = label_list[siamese_size:]
            
                for i in range(0, len(siamese_imgs), batch_size):
                    # Get mini-batch
                    batch_imgs = PackageSiameseBatch(siamese_imgs, i)
                    
                    # Model output and loss calculation
                    _, _ = model(batch_imgs, mode='siamese_pred')
                
                # Pred Phase 
                total_train_samples = 0
                total_train_correct = 0
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
                    optimizer_prediction.zero_grad()
                    output = model(batch_imgs, mode='prediction')
                    loss = criterion(output, batch_labels)
                    loss.backward()
                    optimizer_prediction.step()

                    # Update training statistics
                    train_pred_metrics.update(loss, output, batch_labels, len(batch_imgs))

            
            # Perform validation check at specified interval
            batch_counter += 1
            if batch_counter % check_interval == 0:
                val_loss = val_check(model, train_siamese_metrics, train_pred_metrics, training_mode)
                model.train()
                total_train_correct = 0
                train_loss = 0
                total_train_samples = 0
                
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    state = {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_siamese': optimizer_siamese.state_dict(),
                        'optimizer_prediction': optimizer_prediction.state_dict(),
                        'val_loss': lowest_val_loss
                    }
                    torch.save(state, f"{env}/models/{name}.pt")
                    print(f"Saved Model")
