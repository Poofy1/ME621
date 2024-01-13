import os, csv, pickle, datetime, requests, time
from requests.exceptions import RequestException
import pandas as pd
import numpy as np


def fetch_with_retries(url, headers, max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: Status code {response.status_code}, attempt {attempt + 1}")
        except RequestException as e:
            print(f"Request failed: {e}, attempt {attempt + 1}")

        time.sleep(delay)  # Wait before retrying

    print("Max retries reached. Exiting.")
    return None
    
    
def add_valid_column(csv_file, split=0.2):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize all rows in 'Valid' column as 0
    df['Valid'] = 0

    # Determine the number of rows to mark as valid
    num_valid = int(len(df) * split)

    # Randomly select rows and set them to 1
    valid_indices = np.random.choice(df.index, num_valid, replace=False)
    df.loc[valid_indices, 'Valid'] = 1

    # Save the updated dataframe
    df.to_csv(csv_file, index=False)

    
if __name__ == "__main__":
    
    # Config
    save_dir = 'D:/DATA/E621/'
    headers = {'User-Agent': 'Dataset Creator 2.0 (by Poof75 on e621)'}
    
    # Initialization
    source_dir = f'{save_dir}/source_images.csv'
    pickle_file = f'{save_dir}/source_pageID.pkl'
    os.makedirs(save_dir, exist_ok=True)


    # Get pageID ending point (starting_point)
    url = f'https://e621.net/posts.json?page=b999999999&tags=-animated&limit=320'
    response = requests.get(url, headers=headers)
    page = response.json()
    starting_point = page['posts'][0]['id']
    
    
    starting_point = page['posts'][0]['id']

    # Load the last pageID if it exists
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as pf:
            pickle_data = pickle.load(pf)
        pageID = pickle_data['current_pageID']
        last_stored_starting_point = pickle_data['starting_point']
        forgotten_starting_point = pickle_data['last_stored_starting_point']
        
        
    else:
        url = f'https://e621.net/posts.json?page=b320&tags=-animated&limit=320'
        response = requests.get(url, headers=headers)
        page = response.json()
        last_stored_starting_point = page['posts'][0]['id']
    
        pageID = 0
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
        url = f'https://e621.net/posts.json?page=b{pageID}&tags=-animated&limit=320'
        page = fetch_with_retries(url, headers)
        if page is None:
            break 
        
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

            
            
            
    # Apply validation split
    add_valid_column(source_dir, split = .05)
    
    print("Finished Operations")