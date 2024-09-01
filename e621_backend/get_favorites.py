import os
import base64
import csv
import requests
import json
import io
import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import global_config

# Get the directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
SAVE_DIR = os.path.join(parent_dir, 'data')


# Function to fetch user's favorites
def fetch_favorites(page=1):
    url = f'https://e621.net/favorites.json?login={global_config["config"]["USERNAME"]}&api_key={global_config["config"]["E621_API"]}&page={page}'
    response = requests.get(url, headers=global_config['headers'])
    all_posts = response.json()['posts']
    
    # Filter out animated images and posts without a valid sample URL
    valid_posts = [
        post for post in all_posts 
        if 'animated' not in post['tags']['meta'] 
        and post['sample']['url'] is not None 
        and post['sample']['url'].lower() != 'none'
    ]
    
    return valid_posts

def get_existing_favorites():
    dataset_path = os.path.join(SAVE_DIR, 'dataset.csv')
    existing_favorites = set()
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                existing_favorites.add(int(row[0].split('.')[0]))
    return existing_favorites

# Function to download and process image
def download_and_process_image(post):
    try:
        response = requests.get(post['sample']['url'], headers=global_config['headers'])
        img = Image.open(io.BytesIO(response.content))
        img.thumbnail((250, 250))
        
        # Save thumbnail
        images_dir = os.path.join(SAVE_DIR, 'images')
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
    dataset_path = os.path.join(SAVE_DIR, 'dataset.csv')
    file_exists = os.path.isfile(dataset_path)
    
    with open(dataset_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['image_name', 'label'])
        
        for i, item in enumerate(data):
            writer.writerow([item['filename'], item['label']])


def download_favorites():
    existing_favorites = get_existing_favorites()
    all_favorites = []
    page = 1
    
    while True:
        favorites = fetch_favorites(page)
        if not favorites:
            break
        all_favorites.extend(favorites)
        page += 1
    
    new_favorites = [fav for fav in all_favorites if fav['id'] not in existing_favorites]
    
    if not new_favorites:
        print("No new favorites to add.")
        return
    
    processed_data = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_post = {executor.submit(download_and_process_image, post): post for post in new_favorites}
        for future in as_completed(future_to_post):
            result = future.result()
            if result:
                processed_data.append(result)
    
    print(f"Successfully processed {len(processed_data)} images")
    
    save_to_csv(processed_data)
    print("New favorites added to CSV")