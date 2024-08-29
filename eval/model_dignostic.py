import torch
import plotly.express as px
import plotly.offline as pyo
import pandas as pd
import shutil
from PIL import Image
import tqdm
import numpy as np
import warnings
from tqdm import tqdm as tqdm_loop
from gradcam import GradCAMpp
from gradcam.utils import visualize_cam
from PIL import ImageFile
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
import umap
from train import *

warnings.filterwarnings("ignore", message="Using a non-full backward hook")
warnings.filterwarnings("ignore", message="nn.functional.upsample is deprecated")


def postprocess_image(image_tensor):
    img = image_tensor.squeeze(0).detach().cpu()
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # Reverse normalization
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)

    img = transforms.functional.to_pil_image(img)
    return img





def create_dataset(input_path, output_path, split):
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    train0 = f"{output_path}/train/0"
    train1 = f"{output_path}/train/1"
    val0 = f"{output_path}/val/0"
    val1 = f"{output_path}/val/1"
    
    os.makedirs(output_path)
    os.makedirs(train0)
    os.makedirs(train1)
    os.makedirs(val0)
    os.makedirs(val1)
    
    # Get a list of all files in input_path/0 and input_path/1
    input0_files = os.listdir(f"{input_path}/0/")
    input1_files = os.listdir(f"{input_path}/1/")

    # Shuffle the files randomly
    random.shuffle(input0_files)
    random.shuffle(input1_files)

    # Calculate the number of files for validation
    num_val_input1 = int(len(input1_files) * split)
    num_val_input0 = num_val_input1 # must be the same as input 1
    
    print(f"Train_0 Size: {len(input0_files) - num_val_input0}")
    print(f"Train_1 Size: {len(input1_files) - num_val_input1}")
    print(f"Val_0 Size: {num_val_input0}")
    print(f"Val_1 Size: {num_val_input1}")

    # Copy files to train and val folders
    for i, (input0_file, input1_file) in enumerate(tqdm_loop(zip(input0_files, input1_files))):
        if i < num_val_input0:
            shutil.copy2(os.path.join(input_path, '0', input0_file), os.path.join(val0, input0_file))
        else:
            shutil.copy2(os.path.join(input_path, '0', input0_file), os.path.join(train0, input0_file))

        if i < num_val_input1:
            shutil.copy2(os.path.join(input_path, '1', input1_file), os.path.join(val1, input1_file))
        else:
            shutil.copy2(os.path.join(input_path, '1', input1_file), os.path.join(train1, input1_file))






            
def process_image(image_path, image):
    try:
        with open(f"{image_path}{image}", 'rb') as f:
            img = Image.open(f)
    except:
        print(f"Removing image: {image}")
        os.remove(f"{image_path}/{image}")

def remove_bad_images(image_path, num_workers=12):
    images = os.listdir(image_path)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, image_path, image) for image in images]



class TestDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform
        self.samples = [os.path.join(input_dir, img) for img in os.listdir(input_dir)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        image_name = os.path.basename(image_path)
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            image = Image.open(image_path).convert("RGB")
        except IOError:
            print(f"Failed to load {image_path}")
            return None, image_name

        if self.transform:
            image = self.transform(image)

        return image, image_name

def image_test(model_name, input_dir, move_images, threshold):
    model = torch.load(f"{env}/models/{model_name}.pt")
    model.to(device)
    model.eval()

    dataset = TestDataset(input_dir, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=12)

    for images, image_names in tqdm_loop(dataloader):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_probs = probabilities[:, 1].cpu().numpy()

        for idx, predicted_prob in enumerate(predicted_probs):
            image_name = image_names[idx]

            if move_images:
                if predicted_prob > threshold:
                    #print(f"{image_name} good")
                    shutil.move(os.path.join(input_dir, image_name), f"D:/DATA/E621/waiting_for_review/good/{image_name}")
                #else:
                    #shutil.move(os.path.join(input_dir, image_name), f"D:/DATA/E621/waiting_for_review/bad/{image_name}")
            else: 
                if predicted_prob > threshold:
                    print(f"{image_name}: {predicted_prob:.3f}")

def deep_dream(model_name, input_image, iterations, lr):
    # Load the final trained model
    model = torch.load(f"{env}/models/{model_name}.pt")
    
    print("Creating Deep Dream")
    model.eval()
    input_image = Image.open(input_image).convert("RGB")
    input_image = val_transform(input_image).unsqueeze(0)
    
    input_image = input_image.to(device).requires_grad_(True)

    for i in range(iterations):
        model.zero_grad()
        out = model.extract_features(input_image)
        loss = out.norm()
        loss.backward()

        input_image.data = input_image.data + lr * input_image.grad.data

    image =  input_image.detach().cpu()
    
    output_image = transforms.ToPILImage()(output_image.squeeze(0))
    output_image.save(f"{env}/outputs/dream1.png")

        
        
def heat_map(model_name, input_image):
    # Load the final trained model
    model = torch.load(f"{env}/models/{model_name}.pt")

    print("Creating heat map")

    # Get the target layer of the EfficientNet model
    target_layer = model.efficientnet._blocks[-1]

    # Instantiate Grad-CAM++ object
    gradcam = GradCAMpp(model, target_layer)

    image = Image.open(input_image).convert("RGB")
    image = val_transform(image)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
    probabilities = torch.softmax(output, dim=1)
    predicted_prob = probabilities[0, 1].item()  
    print(f"Image {input_image}: {predicted_prob*100:.1f}%")

    # Get the Grad-CAM++ mask
    mask, _ = gradcam(image)
    heatmap, result = visualize_cam(mask, image)

    # Save the heatmap and the result
    result_image = postprocess_image(result)
    heatmap_image = postprocess_image(heatmap)
    
    result_image.save(f"{env}/outputs/heat_map.png")
    heatmap_image.save(f"{env}/outputs/heat_map_image.png")
    
    
def model_features(model_name, batch_size):
    # Load the final trained model
    model = torch.load(f"{env}/models/{model_name}.pt")

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
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        features.extend(feats.cpu().numpy())
        labels.extend(lbls.numpy())

    features = np.array(features)
    features = features.reshape(features.shape[0], -1)
    labels = np.array(labels)

    # Dimensionality reduction
    print("Reducing dimensions...")
    reducer = umap.UMAP(n_components=3)
    embedding = reducer.fit_transform(features)

    
    print("Creating DataFrame with the features...")
    # Create a pandas DataFrame with the features and labels
    df = pd.DataFrame(embedding, columns=['feature_1', 'feature_2', 'feature_3'])
    df['label'] = labels
    df['image_path'] = [feature_dataset.image_paths[i] for i in range(len(feature_dataset))]

    print("Creating 3D scatter plot...")
    # Create a 3D scatter plot with Plotly
    fig = px.scatter_3d(df, x='feature_1', y='feature_2', z='feature_3', color='label', opacity=0.8, hover_data=['image_path'])

    # Set the axis labels
    fig.update_layout(scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))
    
    html_filename = f"{env}/models/{model_name}.html"
    pyo.plot(fig, filename=html_filename, auto_open=True)