import torch
import os
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchvision.models as models
from tqdm import tqdm
from multiprocessing import Pool
import warnings
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
torch.backends.cudnn.benchmark = False
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"






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
        img = Image.open(img_path)
        img = self.transform(img)

        age_input = torch.tensor([self.age_values[index]], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)

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
    def __init__(self, num_classes=1, dropout_rate=0.5, lstm_hidden_size=128, lstm_layers=1):
        super(ME621_Model, self).__init__()
        self.efficientnet = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')

        # Remove the original classifier from EfficientNet
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # FC layer to transform features for LSTM
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
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

        # Output layer
        self.output_layer = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x, hidden=None):
        # Process images through EfficientNet
        features = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(features)
        x = torch.flatten(x, 1)

        # Pass through FC layer
        fc_out = self.fc(x)

        # Reshape for LSTM
        # If you have a single image at a time, you might need to add a dimension: fc_out.unsqueeze(1)
        lstm_out, hidden = self.lstm(fc_out.unsqueeze(1), hidden)

        # LSTM output to final layer
        # We take the output of the last time step
        lstm_out_last_step = lstm_out[:, -1, :]

        # Final output
        out = torch.sigmoid(self.output_layer(lstm_out_last_step))

        return out, hidden 




def evaluate_model(model, val_dataloader):
    model.eval()  # Set the model to evaluation mode

    # Variables to store total absolute differences and counts
    total_diff = np.zeros(3)  # Assuming there are 3 output variables
    count = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_dataloader:
            val_inputs = tuple(input_tensor.to(device) for input_tensor in val_inputs)
            val_labels = val_labels.to(device)

            val_img, val_age_input = val_inputs
            val_outputs, _ = model(val_img, val_age_input)

            # Calculate the absolute difference
            diff = torch.abs(val_outputs - val_labels)
            total_diff += diff.sum(dim=0).cpu().numpy()
            count += val_labels.size(0)

    # Calculate the average absolute differences
    avg_diff = total_diff / count
    return avg_diff


if __name__ == "__main__":
    
    # Config
    name = 'ME621_Score_2'
    image_size = 350
    dropout_rate = 0.0
    batch_size = 16
    epochs = 50
    check_interval = 10000
    raw_image_path = 'D:/DATA/E621/images/'
    ready_images_path = f'F:/Temp_SSD_Data/ME621/'
    data_csv = 'D:/DATA/E621/source_images.csv'


    # Load the dataset
    print('Preparing Dataset...')
    train_dataset, val_dataset = create_datasets(raw_image_path, data_csv, ready_images_path, image_size)


    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    
    # Define model, loss function, and optimizer
    model = ME621_Model(num_classes=3, dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #summary(model, input_size=[(batch_size, 3, image_size, image_size), (batch_size, 1)])
    
    
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
        
        
    # Evaluate the model
    avg_diff = evaluate_model(model, val_dataloader)
    print(f"Average Distance Off for Each Variable: {avg_diff}")
        
    
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
            inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            labels = labels.to(device)

            img, age_input = inputs
            outputs, _ = model(img, age_input)
            #print(f"Age: {int(age_input[0].item())} Out: {outputs[0].detach().cpu().numpy().round().astype(int)}, True: {labels[0].cpu().numpy().round().astype(int)}")
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
                        val_inputs = tuple(input_tensor.to(device) for input_tensor in val_inputs)
                        val_labels = val_labels.to(device)

                        val_img, val_age_input = val_inputs
                        val_outputs, _ = model(val_img, val_age_input)

                        # Compute loss based on the actual batch size
                        current_batch_size = val_img.size(0)
                        val_loss += criterion(val_outputs, val_labels).item() * current_batch_size
                        total_val_samples += current_batch_size

                val_loss /= total_val_samples
                print(f'\nEpoch: [{epoch}] [{int((batch_counter / total_batches) * 100)}%] | Train Loss: {train_loss / total_train_samples:.3f} | Val Loss: {val_loss:.3f}')
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
                    
                

        model.eval()
        val_loss = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_dataloader:
                val_inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
                val_labels = val_labels.to(device)
                
                val_img, val_age_input = val_inputs
                val_outputs, _ = model(val_img, val_age_input)
                val_loss += criterion(val_outputs, val_labels).item() * val_img.size(0)
                total_val_samples += val_img.size(0)

        val_loss /= total_val_samples
        print(f'\nEpoch: [{epoch}] [{int((batch_counter / total_batches) * 100)}%] | Train Loss: {train_loss / total_train_samples:.3f} | Val Loss: {val_loss:.3f}')
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