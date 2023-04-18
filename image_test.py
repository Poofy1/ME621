import torch
import os
import shutil
from PIL import Image
from tqdm import tqdm as tqdm_loop
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# Initialize 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = os.path.dirname(os.path.abspath(__file__))

model_name = "main7"
test_images_path = 'D:/DATA/E621/new_good/'
#test_images_path = f'{env}/testing/'
move_images = True
good_images_path = "D:/DATA/E621/GOOD_FINAL_BATCH/"
bad_images_path = "D:/DATA/E621/new_bad/"
image_size = 224
threshold = .925

# Define transform for the test images
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()
        self.efficientnet = models.efficientnet_b1(pretrained=True)
        num_ftrs = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

model = torch.load(f"{env}/models/{model_name}.pt")
model.eval()

# Classify test images
for image_name in tqdm_loop(os.listdir(test_images_path)):
    try:
        image_path = os.path.join(test_images_path, image_name)
        test_image = Image.open(image_path).convert("RGB")
        test_image_tensor = test_transform(test_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(test_image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_prob = probabilities[0, 1].item()  
        
        
        if move_images:
            if predicted_prob > threshold:
                shutil.move(image_path, good_images_path)
                print(f"{image_name}: {predicted_prob:.3f}")
            #else:
                #shutil.move(image_path, bad_images_path)
        else: 
                print(f"{image_name}: {predicted_prob:.3f}")
        
    except: 
        pass
        print(f"{image_name} failed")