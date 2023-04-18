import torch
import os
from torchvision.transforms import ToPILImage
import plotly.express as px
import plotly.offline as pyo
import torch.nn as nn
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm as tqdm_loop
import torch.optim as optim
from sklearn.metrics import classification_report
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import multiprocessing
import umap
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torchinfo import summary

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# Config
image_size = 224
batch_size = 32
num_epochs = 6
train = False
continue_training = False
model_features = True
gen_image = False
save_model = "main8"
load_model = "main8"
train_path = 'D:/DATA/E621/train/'
val_path = 'D:/DATA/E621/val/'


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

        # Convert the image to a tensor and normalize it
        tensor_img = transforms.ToTensor()(resized_img)

        return tensor_img

# Define transforms for data augmentation
transform = transforms.Compose([
    SquareResize(image_size),
    transforms.RandomRotation(degrees=35),
    transforms.RandomResizedCrop(size=image_size, scale=(0.75, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.0001, 0.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=(-0.2, 0.2)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Define transforms for data augmentation
val_transform = transforms.Compose([
    SquareResize(image_size),
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

        return img, label

class MySubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, oversampler=None):
        if oversampler is None:
            super().__init__(dataset, indices)
            self.labels = [dataset.labels[idx] for idx in indices]
        else:
            indices = list(indices)
            self.indices = [oversampler.get_indices()[i] for i in indices]
            self.dataset = dataset
            self.labels = [dataset.labels[idx] for idx in self.indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)



# Add freeze_support() call
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # use either b1 or b0, b0 is not fully tested
    # b1 acc was like %92
    class MyModel(nn.Module):
        def __init__(self, num_classes=2):
            super(MyModel, self).__init__()
            self.efficientnet = models.efficientnet_b0(pretrained=True, dropout_rate=0.5)
            num_ftrs = self.efficientnet.classifier[-1].in_features
            self.efficientnet.classifier[-1] = nn.Linear(num_ftrs, num_classes)

        def forward(self, x):
            x = self.efficientnet(x)
            return x

        def extract_features(self, x):
            # Temporarily remove the classifier
            classifier = self.efficientnet.classifier
            self.efficientnet.classifier = nn.Identity()

            # Extract features
            x = self.efficientnet._forward_impl(x)

            # Restore the classifier
            self.efficientnet.classifier = classifier

            return x

    
    # Define model, loss function, and optimizer
    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # print out the model summary
    summary(model, input_size=(batch_size, 3, image_size, image_size))
    
    
    #Train?
    if train:
        
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

        
        # Continue Training?
        if continue_training:
            model = torch.load(f"{env}/models/{load_model}.pt")
            
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
        
        torch.save(model, f"{env}/models/{save_model}.pt")
    
    

    if model_features:
        # Load the final trained model
        model = torch.load(f"{env}/models/{load_model}.pt")

        # Create a new DataLoader without the WeightedRandomSampler
        feature_dataset = MyDataset(train_path, transform = val_transform)
        feature_extraction_dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=False)

        # Feature extraction
        print("Extracting features...")
        model.eval()
        features = []
        labels = []

        for images, lbls in tqdm_loop(feature_extraction_dataloader):
            images = images.to(device)
            with torch.no_grad():
                feats = model.extract_features(images)
            features.extend(feats.cpu().numpy())
            labels.extend(lbls.numpy())

        features = np.array(features)
        labels = np.array(labels)

        # Dimensionality reduction
        print("Reducing dimensions...")
        reducer = umap.UMAP(n_components=3)
        embedding = reducer.fit_transform(features)

        # Create a pandas DataFrame with the features and labels
        df = pd.DataFrame(embedding, columns=['feature_1', 'feature_2', 'feature_3'])
        df['label'] = labels
        df['image_path'] = [feature_dataset.image_paths[i] for i in range(len(feature_dataset))]

        # Create a 3D scatter plot with Plotly
        fig = px.scatter_3d(df, x='feature_1', y='feature_2', z='feature_3', color='label', opacity=0.8, hover_data=['image_path'])

        # Set the axis labels
        fig.update_layout(scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))
        
        html_filename = f"{env}/models/{load_model}.html"
        pyo.plot(fig, filename=html_filename, auto_open=True)
    
    
    # define a function for activation maximization
    def max_layer(model, layer_index, image_size, num_iterations=2000):
        # initialize the image to a random noise
        image = torch.randn(1, 3, image_size, image_size, device=device, requires_grad=True)

        # define the optimization algorithm
        optimizer = torch.optim.Adam([image], lr=0.1)

        # iterate to maximize the activation of the specified layer
        for i in tqdm_loop(range(num_iterations)):
            optimizer.zero_grad()
            features = model.extract_features(image)
            activation = features[0, :layer_index+1]
            frob_norm = torch.norm(activation.view(-1), p=2)
            frob_norm.backward()
            optimizer.step()

            # clip the image values to [0, 1]
            image.data = torch.clamp(image.data, 0, 1)

        # convert the tensor image to a PIL image
        image = ToPILImage()(image.cpu().detach().squeeze())

        return np.array(image)

    def max_class(model, class_index, image_size, num_iterations=1000):
        # initialize the image to a random noise
        image = torch.randn(1, 3, image_size, image_size, device=device, requires_grad=True)
        
        # define the optimization algorithm
        optimizer = torch.optim.Adam([image], lr=0.1)

        # iterate to maximize the activation of the specified class
        for i in tqdm_loop(range(num_iterations)):
            optimizer.zero_grad()
            outputs = model(image)
            activation = outputs[0, class_index]
            loss = -activation
            loss.backward()
            optimizer.step()

            # clip the image values to [0, 1]
            image.data = torch.clamp(image.data, 0, 1)

        # convert the tensor image to a PIL image
        image = ToPILImage()(image.cpu().detach().squeeze())

        return image
    
    if gen_image:
        
        # Load the final trained model
        model = torch.load(f"{env}/models/{load_model}.pt")
        # generate an image that maximizes the activation of a neuron in a specified layer
        layer_index = 12000
        image = max_layer(model, layer_index, image_size)
        #image = max_class(model, 1, image_size)
        
        
        # display the generated image
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()
    