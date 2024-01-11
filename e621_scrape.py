import urllib.request
from urllib.request import FancyURLopener
import json, os, warnings, urllib, csv
import concurrent.futures
import tqdm
import pickle
import datetime

# Init connection
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
class MyOpener(FancyURLopener):
    version = 'Dataset Creator 2.0 (by Poof75 on e621)'
myopener = MyOpener()





def download_image(image_info):
    image_id, image_url = image_info
    image_path1 = f"D:/DATA/E621/imageClassifier/images/uncategorised/{image_id}.jpg"
    image_path2 = f"D:/DATA/E621/imageClassifier/images/good/{image_id}.jpg"
    image_path3 = f"D:/DATA/E621/imageClassifier/images/bad/{image_id}.jpg"
    if not (os.path.isfile(image_path1) or os.path.isfile(image_path2) or os.path.isfile(image_path3)):
        try:
            urllib.request.urlretrieve(image_url, image_path1)
        except:
            return f"{image_url} ERROR"


    
    
    
if __name__ == "__main__":
    
    # Config
    source_images = True
    download_images = False
    save_dir = 'D:/DATA/E621/'
    
    # Initialization
    source_dir = f'{save_dir}/source_images.csv'
    image_dir = f'{save_dir}/images/'
    pickle_file = f'{save_dir}/source_pageID.pkl'
    os.makedirs(image_dir, exist_ok=True)
    
    
    
    
    

    if source_images:
    
        # Get pageID ending point (starting_point)
        page = json.loads(myopener.open('https://e621.net/posts.json?page=b999999999&tags=-animated&limit=320').read().decode("utf-8"))
        starting_point = page['posts'][0]['id']

        # Load the last pageID if it exists
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as pf:
                pickle_data = pickle.load(pf)
            pageID = pickle_data['current_pageID']
            last_stored_starting_point = pickle_data['starting_point']
            forgotten_starting_point = pickle_data['last_stored_starting_point']
        else:
            pageID = 0
            last_stored_starting_point = 0
            forgotten_starting_point = 0
        
        # Finding restore point
        if (pageID <= forgotten_starting_point):
            pageID = starting_point
        else:
            starting_point = last_stored_starting_point
            last_stored_starting_point = forgotten_starting_point
            
            
        # Check if CSV file exists and write headers if it doesn't
        csv_headers = ['ID', 'Created Date', 'Saved Timestamp', 'Source URL', 'Sample URL', 'General Tags', 'Artist Tags', 'Copyright Tags', 'Character Tags', 'Species Tags', 'Meta Tags', 'Rating', 'Upvotes', 'Downvotes', 'Score', 'Favorites', ]
        if not os.path.exists(source_dir):
            with open(source_dir, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)
        
        # Getting data
        while pageID > last_stored_starting_point:
            image_data = []

            print(f"Sourcing page b{pageID}")
            page = json.loads(myopener.open(f'https://e621.net/posts.json?page=b{pageID}&tags=-animated&limit=320').read().decode("utf-8"))
            posts = page['posts']
            
            # Current timestamp
            current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
            for post in posts:
                tags = post['tags']
                score = post['score']
                
                # Columns
                id = post['id']
                created_date = post['created_at']
                source_url = post['file']['url']
                sample_url = post['sample']['url']
                general_tags = tags['general']
                artist_tags = tags['artist']
                copyright_tags = tags['copyright']
                character_tags = tags['character']
                species_tags = tags['species']
                meta_tags = tags['meta']
                rating = post['rating']
                upvotes = score['up']
                downvotes = score['down']
                score = score['total']
                favorites = post['fav_count']
                
                if (id > last_stored_starting_point and source_url is not None):
                    image_data.append([id,
                                    created_date, 
                                    current_timestamp,
                                    source_url,
                                    sample_url,
                                    general_tags,
                                    artist_tags,
                                    copyright_tags,
                                    character_tags,
                                    species_tags,
                                    meta_tags,
                                    rating,
                                    upvotes,
                                    downvotes,
                                    score,
                                    favorites])

            pageID = id
            
            # Update the pickle file with the current pageID and starting point
            pickle_data = {
                'current_pageID': pageID,
                'starting_point': starting_point,
                'last_stored_starting_point': last_stored_starting_point
            }
            with open(pickle_file, 'wb') as pf:
                pickle.dump(pickle_data, pf)
                
            
            # Update the CSV file
            with open(source_dir, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(image_data)

    
    
    if download_images:
        print(f"Downloading {len(image_data)} Images")

        # Initialize a progress bar with the total number of files
        pbar = tqdm.tqdm(total=len(image_data))

        # Use a ThreadPoolExecutor to run 8 download_image functions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
            # Schedule each download_image function with its respective image_info
            future_to_image_info = {executor.submit(download_image, image_info): image_info for image_info in image_data}

            # Iterate over the completed futures to update the progress bar and print results
            for future in concurrent.futures.as_completed(future_to_image_info):
                # Update the progress bar after processing each file
                pbar.update(1)

        # Close the progress bar
        pbar.close()