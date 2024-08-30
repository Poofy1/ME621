import os
import random
import csv
import requests
import json
import base64
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import webbrowser
from flask import Flask, jsonify, request, render_template, Response
import json
from threading import Timer
from get_favorites import *

app = Flask(__name__)

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
    'User-Agent': "User Annotation (https://github.com/Poofy1/ME621)",
    'Authorization': f"Basic {key}"
}

def load_labeled_images():
    dataset_path = os.path.join(config['SAVE_DIR'], 'dataset.csv')
    labeled_images = set()
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                labeled_images.add(int(row[0].split('.')[0]))  # Add image ID to set
    return labeled_images

labeled_images = load_labeled_images()

# Function to get the maximum page_id
def get_max_page_id():
    url = f'https://e621.net/posts.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&page=b999999999&tags=-animated&limit=200'
    response = requests.get(url, headers=headers)
    page = response.json()
    return page['posts'][0]['id']

# Function to fetch images from the API
def fetch_images(page_id):
    rand_score = random.randint(0, 1000)
    url = f'https://e621.net/posts.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&page=a{page_id}&tags=-animated+score:>={rand_score}&limit=200'
    response = requests.get(url, headers=headers)
    posts = response.json()['posts']
    return [post for post in posts if post['id'] not in labeled_images]

# Function to download image
def download_image(image_data):
    try:
        response = requests.get(image_data['sample']['url'], headers=headers)
        return image_data, response.content
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def get_favorites_count():
    existing_favorites = get_existing_favorites()
    new_favorites_count = 0
    page = 1
    while True:
        favorites = fetch_favorites(page)
        if not favorites:
            break
        for favorite in favorites:
            if favorite['id'] not in existing_favorites:
                new_favorites_count += 1
        page += 1
    return new_favorites_count

# Function to process image
def process_image(image_data, image_content):
    try:
        img = Image.open(io.BytesIO(image_content))
        img.thumbnail((250, 250))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return image_data, img_str
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to save labeled images
def save_labeled_images(images_data):
    images_dir = os.path.join(config['SAVE_DIR'], 'images')
    os.makedirs(images_dir, exist_ok=True)
    dataset_path = os.path.join(config['SAVE_DIR'], 'dataset.csv')

    file_exists = os.path.isfile(dataset_path)

    with open(dataset_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if the file is new
        if not file_exists:
            writer.writerow(['image_name', 'label', 'split'])

        for image_data in images_data:
            image_id = image_data['id']
            image_path = os.path.join(images_dir, f"{image_id}.png")
            
            # Save the image
            image_content = base64.b64decode(image_data['raw_content'])
            with open(image_path, 'wb') as img_file:
                img_file.write(image_content)
            
            # Save the annotation
            writer.writerow([f"{image_id}.png", image_data['label'], "train" if random.random() < 0.8 else "val"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/next-images')
def get_next_images():
    max_page_id = get_max_page_id()
    random_page_id = random.randint(1000, max_page_id)
    posts = fetch_images(random_page_id)
    
    processed_images = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_post = {executor.submit(download_image, post): post for post in posts}
        for future in as_completed(future_to_post):
            result = future.result()
            if result:
                post, raw_content = result
                processed = process_image(post, raw_content)
                if processed:
                    post, img_str = processed
                    processed_images.append({
                        'id': post['id'],
                        'url': f"data:image/png;base64,{img_str}",
                        'raw_content': img_str,
                        'label': 0
                    })
    
    return jsonify({'images': processed_images})

@app.route('/api/images')
def get_images():
    def generate():
        max_page_id = get_max_page_id()
        processed_images = []
        total_fetched = 0
        
        while len(processed_images) < 200 and total_fetched < 1000:  # Set a limit to prevent infinite loop
            random_page_id = random.randint(1000, max_page_id)
            posts = fetch_images(random_page_id)
            total_fetched += len(posts)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_post = {executor.submit(download_image, post): post for post in posts}
                for future in as_completed(future_to_post):
                    result = future.result()
                    if result:
                        post, raw_content = result
                        processed = process_image(post, raw_content)
                        if processed:
                            post, img_str = processed
                            processed_images.append({
                                'id': post['id'],
                                'url': f"data:image/png;base64,{img_str}",
                                'raw_content': img_str,
                                'label': 0
                            })
                    
                    progress = int(min(len(processed_images), 200) / 200 * 100)
                    yield f"data: {json.dumps({'progress': progress})}\n\n"
                    
                    if len(processed_images) >= 200:
                        break
        
        yield f"data: {json.dumps({'images': processed_images[:200]})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/favorites-count')
def favorites_count():
    count = get_favorites_count()
    return jsonify({'count': count})

@app.route('/api/import-favorites', methods=['POST'])
def import_favorites():
    download_favorites()
    return jsonify({'status': 'success'})

@app.route('/api/label-count')
def get_label_count():
    dataset_path = os.path.join(config['SAVE_DIR'], 'dataset.csv')
    count = 0
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            csv_reader = csv.reader(f)
            count = sum(1 for row in csv_reader if row[1] == '1')
    return jsonify({'count': count})

@app.route('/api/save', methods=['POST'])
def save_labels():
    global labeled_images
    labeled_images_data = request.json
    save_labeled_images(labeled_images_data)
    for image_data in labeled_images_data:
        labeled_images.add(image_data['id'])
    return jsonify({'status': 'success'})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)