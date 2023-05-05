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
import torchvision.transforms as transforms
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



def remove_bad_images(image_path):
    for image in tqdm(os.listdir(image_path)):
        try:
            with open(f"{image_path}{image}", 'rb') as f:
                img = Image.open(f)
                #img.convert('RGB')
        except:
            print(f"Removing image: {image}")
            os.remove(f"{image_path}/{image}")

def image_test(model_name, input_dir, move_images, threshold):
    model = torch.load(f"{env}/models/{model_name}.pt")

    for image_name in tqdm_loop(os.listdir(input_dir)):
        try:
            image_path = os.path.join(input_dir, image_name)
            test_image = Image.open(image_path).convert("RGB")
            test_image_tensor = val_transform(test_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(test_image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_prob = probabilities[0, 1].item()  
            
            
            if move_images:
                if predicted_prob > threshold:
                    shutil.move(image_path, f"D:/DATA/E621/waiting_for_review/good/")
                #else:
                    #shutil.move(image_path, f"D:/DATA/E621/waiting_for_review/bad/")
            else: 
                if predicted_prob > threshold:
                    print(f"{image_name}: {predicted_prob:.3f}")
            
        except: 
            pass
            print(f"{image_name} failed")




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