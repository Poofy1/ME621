import os
from PIL import Image
from tqdm import tqdm

image_path = "D:/DATA/E621/train/0/"

for image in tqdm(os.listdir(image_path)):
    try:
        with open(f"{image_path}{image}", 'rb') as f:
            img = Image.open(f)
            #img.convert('RGB')
    except:
        print(f"Removing image: {image}")
        os.remove(f"{image_path}/{image}")