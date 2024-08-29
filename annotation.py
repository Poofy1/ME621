import os
import random
import csv
from PIL import Image
import tkinter as tk
from PIL import ImageTk
import threading
import queue
import pandas as pd

env = os.path.dirname(os.path.abspath(__file__))

# Directories and files
IMAGE_DIR = "D:/DATA/E621/images"
OUTPUT_CSV = "D:/DATA/E621/dataset.csv"
SOURCE_CSV = "D:/DATA/E621/source_images.csv"

# Preload queue size
QUEUE_SIZE = 5

# Image cache
image_cache = {}

# Load source images data
source_df = pd.read_csv(SOURCE_CSV)

# Set to keep track of labeled images
labeled_images = set()

# Load previously labeled images
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        labeled_images = set(row[0] for row in reader)

# Function to get a random image from the directory with upvote threshold
def get_random_image(upvote_threshold):
    eligible_images = source_df[
        (source_df['Upvotes'] >= upvote_threshold) & 
        (~source_df['ID'].astype(str).add('.png').isin(labeled_images))
    ]
    if eligible_images.empty:
        raise ValueError(f"No unlabeled images found with {upvote_threshold} or more upvotes.")
    
    random_image = eligible_images.sample(1).iloc[0]
    return f"{random_image['ID']}.png"

# Function to write annotation to CSV
def write_annotation(image_name, label, split):
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_name, label, split])
    labeled_images.add(image_name)

# Function to load and process image
def load_image(image_name):
    if image_name in image_cache:
        return image_cache[image_name]
    
    image_path = os.path.join(IMAGE_DIR, image_name)
    img = Image.open(image_path)
    img.thumbnail((512, 512))
    photo = ImageTk.PhotoImage(img)
    image_cache[image_name] = photo
    return photo

# Function to preload images
def preload_images(image_queue, upvote_threshold):
    while True:
        if image_queue.qsize() < QUEUE_SIZE:
            image_name = get_random_image(upvote_threshold)
            image_queue.put(image_name)

# Main annotation function
def annotate_images(upvote_threshold):
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "label", "split"])
    
    root = tk.Tk()
    root.title("Image Annotation")
    
    image_label = tk.Label(root)
    image_label.pack()
    
    instruction_label = tk.Label(root, text="Press 'n' for label 0, 'k' for label 1, 's' to skip, 'u' to undo, 'q' to quit")
    instruction_label.pack()
    
    image_queue = queue.Queue()
    
    # Start preloading thread
    preload_thread = threading.Thread(target=preload_images, args=(image_queue, upvote_threshold), daemon=True)
    preload_thread.start()
    
    current_image_name = None
    previous_image_name = None
    previous_annotation = None
    
    def load_next_image():
        nonlocal current_image_name, previous_image_name
        previous_image_name = current_image_name
        current_image_name = image_queue.get()
        photo = load_image(current_image_name)
        image_label.config(image=photo)
        image_label.image = photo
        
        # Display upvotes
        upvotes = source_df[source_df['ID'] == int(current_image_name[:-4])]['Upvotes'].values[0]
        upvotes_label.config(text=f"Upvotes: {upvotes}")
    
    def undo_last_annotation():
        nonlocal current_image_name, previous_image_name, previous_annotation
        if previous_image_name and previous_annotation:
            current_image_name = previous_image_name
            photo = load_image(current_image_name)
            image_label.config(image=photo)
            image_label.image = photo
            
            # Remove the last line from the CSV file
            with open(OUTPUT_CSV, 'r') as f:
                lines = f.readlines()
            with open(OUTPUT_CSV, 'w') as f:
                f.writelines(lines[:-1])
            
            print(f"Undone: {previous_image_name}")
            
            # Remove from labeled_images set
            labeled_images.remove(previous_image_name)
            
            # Display upvotes
            upvotes = source_df[source_df['ID'] == int(current_image_name[:-4])]['Upvotes'].values[0]
            upvotes_label.config(text=f"Upvotes: {upvotes}")
            
            previous_image_name = None
            previous_annotation = None
    
    def on_key(event):
        nonlocal previous_annotation
        if event.char == 'n':
            label = 0
        elif event.char == 'k':
            label = 1
        elif event.char == 's':
            load_next_image()
            return
        elif event.char == 'u':
            undo_last_annotation()
            return
        elif event.char == 'q':
            root.quit()
            return
        else:
            return
        
        split = "train" if random.random() < 0.8 else "val"
        write_annotation(current_image_name, label, split)
        previous_annotation = (current_image_name, label, split)
        print(f"Annotated: {current_image_name}, Label: {label}, Split: {split}")
        load_next_image()
    
    root.bind('<Key>', on_key)
    
    # Add upvotes label
    upvotes_label = tk.Label(root, text="Upvotes: N/A")
    upvotes_label.pack()
    
    load_next_image()
    root.mainloop()

if __name__ == "__main__":
    upvote_threshold = int(input("Enter the minimum number of upvotes: "))
    annotate_images(upvote_threshold)
    print("Annotation complete. Results saved in", OUTPUT_CSV)