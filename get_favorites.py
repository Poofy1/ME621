import os
import base64
import csv
import requests
import json
import io
import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load configuration
def load_config():
    env = os.path.dirname(os.path.abspath(__file__))
    with open(f'{env}/config.json', 'r') as config_file:
        return json.load(config_file)

config = load_config()

# Set up API authentication
key = f"{config['USERNAME']}:{config['API_KEY']}"
key = base64.b64encode(key.encode()).decode()
headers = {
    'User-Agent': "User Annotation (https://github.com/Poofy1/ME621)",
    'Authorization': f"Basic {key}"
}

# Function to fetch user's favorites
def fetch_favorites(page=1):
    url = f'https://e621.net/favorites.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&page={page}'
    response = requests.get(url, headers=headers)
    return response.json()['posts']

# Function to download and process image
def download_and_process_image(post):
    try:
        # Check if 'animated' is in meta tags
        if 'animated' in post['tags']['meta']:
            return None

        response = requests.get(post['sample']['url'], headers=headers)
        img = Image.open(io.BytesIO(response.content))
        img.thumbnail((250, 250))
        
        # Save thumbnail
        images_dir = os.path.join(config['SAVE_DIR'], 'images')
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, f"{post['id']}.png")
        img.save(image_path, format="PNG")
        
        return {
            'id': post['id'],
            'filename': f"{post['id']}.png",
            'label': 1  # Assuming favorites are positive examples
        }
    except Exception as e:
        print(f"Error processing image {post['id']}: {e}")
        return None

# Function to save data to CSV
def save_to_csv(data):
    dataset_path = os.path.join(config['SAVE_DIR'], 'dataset.csv')
    file_exists = os.path.isfile(dataset_path)
    
    # Randomly shuffle the data
    random.shuffle(data)
    
    # Calculate the split index
    split_index = int(len(data) * 0.8)  # 80% for training
    
    with open(dataset_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['image_name', 'label', 'split'])
        
        for i, item in enumerate(data):
            split = 'train' if i < split_index else 'val'
            writer.writerow([item['filename'], item['label'], split])

def main():
    all_favorites = []
    page = 1
    
    while True:
        favorites = fetch_favorites(page)
        if not favorites:
            break
        all_favorites.extend(favorites)
        page += 1
    
    print(f"Found {len(all_favorites)} favorites")
    
    processed_data = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_post = {executor.submit(download_and_process_image, post): post for post in all_favorites}
        for future in as_completed(future_to_post):
            result = future.result()
            if result:
                processed_data.append(result)
    
    print(f"Successfully processed {len(processed_data)} non-animated images")
    
    save_to_csv(processed_data)
    print("Data saved to CSV")

if __name__ == "__main__":
    main()