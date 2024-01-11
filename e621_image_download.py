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
                    
                    print(f"Finished download for ID: {image_info['ID']}")
                    return 1

            except RequestException as e:
                print(f"Request failed for image: {sample_url}, error: {e}")

            attempt += 1
            time.sleep(delay)

    return 0


    
if __name__ == "__main__":
    
    # Config
    save_dir = 'D:/DATA/E621/'
    headers = {'User-Agent': 'Dataset Creator 2.0 (by Poof75 on e621)'}
    
    # Initialization
    source_dir = f'{save_dir}/source_images.csv'
    image_folder = f'{save_dir}/images/'
    os.makedirs(image_folder, exist_ok=True)

    # Load CSV data
    print("Loading Source Data")
    image_data = pd.read_csv(source_dir)
    
    print("Downloading Images")
    with tqdm.tqdm(total=len(image_data)) as pbar:
        with ThreadPoolExecutor(max_workers=9) as executor:
            # Ensure each row is passed as a dictionary
            futures = [executor.submit(download_image, row.to_dict(), image_folder, headers) for _, row in image_data.iterrows()]

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

                pbar.update(1)