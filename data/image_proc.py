import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool


class SquareResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Calculate the dimensions of the padded image
        max_dim = max(img.size)
        h_pad = max_dim - img.size[1]
        w_pad = max_dim - img.size[0]

        # Pad the image with zeros
        padded_img = transforms.functional.pad(img, (w_pad//2, h_pad//2, w_pad - w_pad//2, h_pad - h_pad//2), 0)

        # Resize the image to the desired size
        resized_img = transforms.Resize((self.size, self.size))(padded_img)

        return resized_img
    
    
    

def process_single_image(img_path, output_dir, resize_and_pad):
    try:
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)
        
        if not os.path.exists(output_path):
            image = Image.open(img_path).convert('RGB')
            image = resize_and_pad(image)
            image.save(output_path)
        
        try:
            with Image.open(output_path) as img:
                img.load()
            return output_path, img_path
        except: 
            print("Removing: ", img_name)
            os.remove(output_path)
            os.remove(img_path)
            return 0
    except:
        return 0
    

def file_exists(row, image_dir):
    img_path = os.path.join(image_dir, f"{row}.png")
    return img_path


def preprocess_and_save_images(df, image_dir, output_dir, image_size):
    # Calculate the aspect ratio
    df['aspect_ratio'] = df['Source Width'] / df['Source Height']
    df = df[(df['aspect_ratio'] >= 0.5) & (df['aspect_ratio'] <= 2)]
    df = df.drop(columns=['Created Date', 'Saved Date', 'Source Width', 'Source Height', 'aspect_ratio'])

    # Prepare ID list for multiprocessing
    ids = df['ID'].tolist()

    # Use multiprocessing Pool to check file existence in parallel
    with Pool() as pool:
        results = list(pool.starmap(file_exists, [(id, image_dir) for id in ids]))

    # Assign results to DataFrame
    df['path'] = results
    
    
    csv_file_path = os.path.join(output_dir, f'images_{image_size}.csv')
    folder_output = f'{output_dir}/images_{image_size}'
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    # Load processed images and create a mapping from ID to processed path
    processed_images = {}
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as file:
            for line in file:
                new_path, original_path = line.strip().split(',')
                processed_images[original_path] = new_path

    # Split the dataframe
    processed_df = df[df['path'].isin(processed_images)]
    unprocessed_df = df[~df['path'].isin(processed_images)]
    print(f'Preprocessed: {len(processed_df)}')
    print(f'Processing: {len(unprocessed_df)}')

    # Map original paths to new paths using a dictionary
    processed_images_map = df['path'].map(processed_images)
    df.loc[df['path'].isin(processed_images), 'path'] = processed_images_map
    
    resize_and_pad = SquareResize(image_size)
    failed_images = []
    batch_size = 10000  # Define your batch size

    unprocessed_df.reset_index(inplace=True)
    unprocessed_df = unprocessed_df.to_dict('records')

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        total_batches = len(unprocessed_df) // batch_size + (len(unprocessed_df) % batch_size > 0)
        with tqdm(total=total_batches, desc="Resizing Images") as pbar:
            for i in range(0, len(unprocessed_df), batch_size):
                batch = unprocessed_df[i:i+batch_size]
                futures = {executor.submit(process_single_image, row['path'], folder_output, resize_and_pad): row for row in batch}

                with open(csv_file_path, 'a' if os.path.exists(csv_file_path) else 'w') as csv_file:
                    for future in as_completed(futures):
                        row = futures[future]
                        result = future.result()
                        if result == 0: 
                            failed_images.append(row['index'])
                        else:
                            new_path, original_path = result
                            csv_file.write(f"{new_path},{original_path}\n")
                            df.at[row['index'], 'path'] = new_path

                pbar.update()

    print("Failed Images: ", len(failed_images))
    df = df.drop(failed_images)
    df = df.dropna(subset=['path'])
    
    return df