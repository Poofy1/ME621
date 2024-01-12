import os
import time
import requests
import tqdm
import pandas as pd
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(image_info, image_folder, headers, max_retries=5, delay=5):

    image_id = str(image_info['ID'])
    sample_url = image_info['Sample URL']
    file_path = os.path.join(image_folder, image_id + ".png")


    if not os.path.exists(file_path):
        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.get(sample_url, headers=headers)
                if response.status_code == 200:
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                    
                    return 1

            except RequestException as e:
                print(f"Request failed for image: {sample_url}, error: {e}")

            attempt += 1
            time.sleep(delay)

    return 0

def process_images_in_batch(batch, image_folder, headers):
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(download_image, row.to_dict(), image_folder, headers) for _, row in batch.iterrows()]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    
    # Config
    save_dir = 'D:/DATA/E621/'
    headers = {'User-Agent': 'Dataset Creator 2.0 (by Poof75 on e621)'}
    
    # Initialization
    source_dir = f'{save_dir}/source_images.csv'
    image_folder = f'{save_dir}/images/'
    os.makedirs(image_folder, exist_ok=True)

    # Define the columns to load and the chunksize
    columns_to_load = ['ID', 'Sample URL']
    chunksize = 10000 

    # Initialize tqdm progress bar
    print("Loading Source Data and Downloading Images")
    total_chunks = pd.read_csv(source_dir, usecols=columns_to_load).shape[0] // chunksize
    with tqdm.tqdm(total=total_chunks) as pbar:
        for chunk in pd.read_csv(source_dir, usecols=columns_to_load, chunksize=chunksize):
            process_images_in_batch(chunk, image_folder, headers)
            pbar.update(1)