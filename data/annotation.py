import os
import random
import csv
import requests
import json
import base64
import tkinter as tk
from PIL import Image, ImageTk
import io
import queue
from threading import Thread
env = os.path.dirname(os.path.abspath(__file__))

# Load configuration
def load_config():
    with open(f'{env}/config.json', 'r') as config_file:
        return json.load(config_file)

config = load_config()

# Set up API authentication
key = f"{config['USERNAME']}:{config['API_KEY']}"
key = base64.b64encode(key.encode()).decode()
headers = {
    'User-Agent': config["HEADER"],
    'Authorization': f"Basic {key}"
}

# Function to get the maximum page_id
def get_max_page_id():
    url = f'https://e621.net/posts.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&page=b999999999&tags=-animated&limit=320'
    response = requests.get(url, headers=headers)
    page = response.json()
    return page['posts'][0]['id']

# Function to fetch images from the API
def fetch_images(page_id):
    url = f'https://e621.net/posts.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&page=a{page_id}&tags=-animated&limit=320'
    response = requests.get(url, headers=headers)
    return response.json()['posts']

# Function to download and process image
def process_image(image_data):
    response = requests.get(image_data['sample']['url'], headers=headers)
    img = Image.open(io.BytesIO(response.content))
    img.thumbnail((512, 512))
    return ImageTk.PhotoImage(img), response.content

# Function to save labeled image
def save_labeled_image(image_data, label):
    # Ensure the images directory exists
    images_dir = os.path.join(config['SAVE_DIR'], 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Download the image
    response = requests.get(image_data['sample']['url'], headers=headers)
    image_id = image_data['id']
    
    # Save the image
    image_path = os.path.join(images_dir, f"{image_id}.png")
    with open(image_path, 'wb') as f:
        f.write(response.content)
    
    # Save the annotation
    dataset_path = os.path.join(config['SAVE_DIR'], 'dataset.csv')
    with open(dataset_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"{image_id}.png", label, "train" if random.random() < 0.8 else "val"])

# Function to undo last annotation
def undo_last_annotation():
    dataset_path = os.path.join(config['SAVE_DIR'], 'dataset.csv')
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:  # Check if there's more than just the header
            last_line = lines[-1].strip().split(',')
            image_name = last_line[0]
            
            # Remove the last line from the CSV
            with open(dataset_path, 'w') as f:
                f.writelines(lines[:-1])
            
            # Remove the image file
            image_path = os.path.join(config['SAVE_DIR'], 'images', image_name)
            if os.path.exists(image_path):
                os.remove(image_path)
            
            print(f"Undone: {image_name}")
            return True
    return False

# Main annotation function
def annotate_images():
    root = tk.Tk()
    root.title("Image Annotation")

    image_label = tk.Label(root)
    image_label.pack()

    instruction_label = tk.Label(root, text="Press 'n' for label 0, 'k' for label 1, 's' to skip, 'u' to undo, 'q' to quit")
    instruction_label.pack()

    image_queue = queue.Queue(maxsize=5)
    processed_images = queue.Queue(maxsize=5)
    current_image = None
    previous_image = None
    max_page_id = get_max_page_id()

    def load_random_page():
        nonlocal max_page_id
        random_page_id = random.randint(1000, max_page_id)
        posts = fetch_images(random_page_id)
        for post in posts:
            if image_queue.qsize() < 5:
                image_queue.put(post)
            else:
                break

    def process_image_thread():
        while True:
            if image_queue.qsize() > 0 and processed_images.qsize() < 5:
                image_data = image_queue.get()
                photo, raw_image = process_image(image_data)
                processed_images.put((image_data, photo, raw_image))
            if image_queue.qsize() < 3:
                load_random_page()

    def load_next_image():
        nonlocal current_image, previous_image
        previous_image = current_image
        if processed_images.empty():
            root.after(100, load_next_image)  # Try again in 100ms
            return
        
        current_image, photo, _ = processed_images.get()
        image_label.config(image=photo)
        image_label.image = photo

    def on_key(event):
        nonlocal current_image, previous_image
        if event.char == 'n':
            label = 0
        elif event.char == 'k':
            label = 1
        elif event.char == 's':
            load_next_image()
            return
        elif event.char == 'u':
            if undo_last_annotation():
                current_image, previous_image = previous_image, None
                if current_image:
                    photo = process_image(current_image['sample']['url'])[0]
                    image_label.config(image=photo)
                    image_label.image = photo
                else:
                    load_next_image()
            return
        elif event.char == 'q':
            root.quit()
            return
        else:
            return
        
        save_labeled_image(current_image, label)
        print(f"Annotated: {current_image['id']}, Label: {label}")
        load_next_image()

    root.bind('<Key>', on_key)

    # Start the image processing thread
    Thread(target=process_image_thread, daemon=True).start()

    # Initial load
    load_random_page()
    load_next_image()

    root.mainloop()

if __name__ == "__main__":
    annotate_images()