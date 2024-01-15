import torch
import os
import gc
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, floor
from PIL import Image
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
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    
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
    return img_path


def process_single_image(img_path, output_dir, resize_and_pad):
    try:
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)
        
        if not os.path.exists(output_path):
            image = Image.open(img_path)
            image = resize_and_pad(image)
            image.save(output_path)
        
        try:
            with Image.open(output_path) as img:
                img.load()
            return output_path, img_path
        except: 
            print("Removing: ", img_name)
            os.remove(output_path)
            os.remove(img_path)
            return 0
    except:
        return 0


def preprocess_and_save_images(df, output_dir, image_size):
    csv_file_path = os.path.join(output_dir, f'images_{image_size}.csv')
    folder_output = f'{output_dir}/images_{image_size}'
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    # Load processed images and create a mapping from ID to processed path
    processed_images = {}
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as file:
            for line in file:
                new_path, original_path = line.strip().split(',')
                processed_images[original_path] = new_path

    # Split the dataframe
    processed_df = df[df['path'].isin(processed_images)]
    unprocessed_df = df[~df['path'].isin(processed_images)]
    print(f'Preprocessed: {len(processed_df)}')
    print(f'Processing: {len(unprocessed_df)}')

    # Map original paths to new paths using a dictionary
    processed_images_map = df['path'].map(processed_images)
    df.loc[df['path'].isin(processed_images), 'path'] = processed_images_map
    
    resize_and_pad = SquareResize(image_size)
    failed_images = []
    batch_size = 10000  # Define your batch size

    unprocessed_df.reset_index(inplace=True)
    unprocessed_df = unprocessed_df.to_dict('records')

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        total_batches = len(unprocessed_df) // batch_size + (len(unprocessed_df) % batch_size > 0)
        with tqdm(total=total_batches, desc="Resizing Images") as pbar:
            for i in range(0, len(unprocessed_df), batch_size):
                print(i)
                batch = unprocessed_df[i:i+batch_size]
                futures = {executor.submit(process_single_image, row['path'], folder_output, resize_and_pad): row for row in batch}

                with open(csv_file_path, 'a' if os.path.exists(csv_file_path) else 'w') as csv_file:
                    for future in as_completed(futures):
                        row = futures[future]
                        result = future.result()
                        if result == 0: 
                            failed_images.append(row['index'])
                        else:
                            new_path, original_path = result
                            csv_file.write(f"{new_path},{original_path}\n")
                            df.at[row['index'], 'path'] = new_path

                pbar.update()

    print("Failed Images: ", len(failed_images))
    df = df.drop(failed_images)
    df = df.dropna(subset=['path'])

    # After processing, trigger garbage collection
    gc.collect()
    
    return df



def create_datasets(image_dir, data_csv, output_dir, image_size):
    # Load the dataset
    dataset_df = pd.read_csv(data_csv, usecols=['ID', 'Created Date', 'Saved Timestamp', 'Upvotes', 'Downvotes', 'Favorites', 'Valid'])
    
    # Keep only the first rows of the existing images
    #dataset_df = dataset_df.head(10000) # DEBUG

    # Find age
    dataset_df['Created Date'] = pd.to_datetime(dataset_df['Created Date'], utc=True)
    dataset_df['Saved Timestamp'] = pd.to_datetime(dataset_df['Saved Timestamp']).dt.tz_localize('Etc/GMT+6').dt.tz_convert('UTC')
    dataset_df['age'] = (dataset_df['Saved Timestamp'] - dataset_df['Created Date']).dt.total_seconds() / (24 * 60 * 60)
    dataset_df = dataset_df.drop(columns=['Created Date', 'Saved Timestamp'])

    # Prepare ID list for multiprocessing
    ids = dataset_df['ID'].tolist()

    # Use multiprocessing Pool to check file existence in parallel
    with Pool() as pool:
        results = list(tqdm(pool.starmap(file_exists, [(id, image_dir) for id in ids]), total=len(ids)))

    # Assign results to DataFrame
    dataset_df['path'] = results
    
    # Preprocess images and update DataFrame
    updated_df = preprocess_and_save_images(dataset_df, output_dir, image_size)

    # Split into train and val sets
    train_df = updated_df[updated_df['Valid'] == 0]
    val_df = updated_df[updated_df['Valid'] == 1]
    
    # Extract labels for train and validation sets
    train_labels = train_df[['Upvotes', 'Downvotes', 'Favorites']].values.tolist()
    val_labels = val_df[['Upvotes', 'Downvotes', 'Favorites']].values.tolist()
    
    # Create image paths and labels
    train_image_paths = train_df['path'].tolist()
    val_image_paths = val_df['path'].tolist()
    
    # Prepare float values (e.g., 'age') for training and validation
    train_age = train_df['age'].tolist()
    val_age = val_df['age'].tolist()

    

    # Define transforms
    train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=(0.8, 1.1), contrast=(0.8, 1.1), saturation=(0.8, 1.1), hue=(-0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
    val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])


    # Create datasets
    train_dataset = ME621_Dataset(train_image_paths, train_labels, train_age, transform=train_transform)
    val_dataset = ME621_Dataset(val_image_paths, val_labels, val_age, transform=val_transform)

    
    #train_dataset.show_image(0)
    #train_dataset.show_image(1)
    #train_dataset.show_image(2)
    
    del updated_df
    gc.collect()

    return train_dataset, val_dataset



class ME621_Dataset(Dataset):
    def __init__(self, image_paths, labels, age_values, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.age_values = age_values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img).to(device)

        age_input = torch.tensor([self.age_values[index]], dtype=torch.float32).to(device)
        label = torch.tensor(self.labels[index], dtype=torch.float32).to(device)  # Ensure labels are integers

        return (img, age_input), label

    
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
        
        num_ftrs = self.efficientnet._fc.in_features

        # Adjust the fully connected layer to take additional float input
        self.efficientnet._fc = nn.Identity()  # Remove original classifier

        # Define new classifier with additional input for age
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 1, 256),  # +1 for the additional float input
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, age_input):
        # Extract features from EfficientNet
        features = self.efficientnet.extract_features(x)
        features = nn.functional.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)

        # Concatenate the additional float input
        # Since age_input is already in the correct shape (torch.Size([32, 1])), no need to unsqueeze
        combined_input = torch.cat((features, age_input), dim=1)

        # Pass through the classifier
        out = self.classifier(combined_input)
        return out


if __name__ == "__main__":
    
    # Config
    name = 'ME621_Score'
    image_size = 300
    dropout_rate = 0.0
    batch_size = 32
    epochs = 50
    check_interval = 1000
    raw_image_path = 'D:/DATA/E621/images/'
    ready_images_path = f'K:/Temp_SSD_Data/'
    data_csv = 'D:/DATA/E621/source_images.csv'


    # Load the dataset
    print('Preparing Dataset...')
    train_dataset, val_dataset = create_datasets(raw_image_path, data_csv, ready_images_path, image_size)


    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    # Define model, loss function, and optimizer
    model = ME621_Model(num_classes=3, dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    summary(model, input_size=[(batch_size, 3, image_size, image_size), (batch_size, 1)])
    
    
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
    total_samples = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size
    total_batches = -(-total_samples // batch_size)

    print("Starting Training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        total_train_samples = 0
        batch_counter = 0

        # Train loop
        for inputs, labels in tqdm(train_dataloader):
            img, age_input = inputs
            outputs = model(img, age_input)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img.size(0)
            total_train_samples += img.size(0)
            batch_counter += 1

            # Perform validation check at specified interval
            if batch_counter % check_interval == 0:
                model.eval()
                val_loss = 0
                total_val_samples = 0

                with torch.no_grad():
                    for val_inputs, val_labels in val_dataloader:
                        val_img, val_age_input = val_inputs
                        val_outputs = model(val_img, val_age_input)
                        val_loss += criterion(val_outputs, val_labels).item() * val_img.size(0)
                        total_val_samples += val_img.size(0)

                val_loss /= total_val_samples
                print(f'Epoch: [{epoch}] [{int((batch_counter / total_batches) * 100)}%] | Train Loss: {train_loss / total_train_samples:.3f} | Val Loss: {val_loss:.3f}')
                train_loss = 0
                total_train_samples = 1

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
                    
                model.train()
