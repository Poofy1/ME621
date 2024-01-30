import os, csv, pickle, datetime, requests, time, json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException
import pandas as pd
import numpy as np
env = os.path.dirname(os.path.abspath(__file__))

def load_config():
    with open(f'{env}/config.json', 'r') as config_file:
        config = json.load(config_file)
        return config
    
    
def add_valid_column(csv_file, num_valid_rows):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize all rows in 'Valid' column as 0
    df['Valid'] = 0

    # Ensure num_valid_rows does not exceed the number of rows in the dataframe
    num_valid_rows = min(num_valid_rows, len(df))

    # Randomly select rows and set them to 1
    valid_indices = np.random.choice(df.index, num_valid_rows, replace=False)
    df.loc[valid_indices, 'Valid'] = 1

    # Save the updated dataframe
    df.to_csv(csv_file, index=False)

    
    
    
def fetch_with_retries(url, headers, max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                return -1
            else:
                print(f"Error: Status code {response.status_code}, attempt {attempt + 1}")
        except RequestException as e:
            print(f"Request failed: {e}, attempt {attempt + 1}")
            

        time.sleep(delay)  # Wait before retrying

    print("Max retries reached. Exiting.")
    return None
    
    
def get_user_data(post, headers):

    # Columns
    id = post['id']
    created_at = post['created_at']
    name = post['name']
    level = post['level']
    base_upload_limit = post['base_upload_limit']
    post_upload_count = post['post_upload_count']
    post_update_count = post['post_update_count']
    is_banned = post['is_banned']
    can_approve_posts = post['can_approve_posts']
    can_upload_free = post['can_upload_free']
    level_string = post['level_string']
    avatar_id = post['avatar_id']
    
    user_url = f'https://e621.net/favorites.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&user_id={id}'
    user_page = fetch_with_retries(user_url, headers)
    favorites = []
    is_private = False
    
    if user_page == -1:
        is_private = True
    else:
        user_page = user_page['posts']
        
        for user_fav in user_page:
            favorites.append(user_fav['id'])

    output = [id,
            created_at,
            name,
            level,
            base_upload_limit,
            post_upload_count,
            post_update_count,
            is_banned,
            can_approve_posts,
            can_upload_free,
            level_string,
            avatar_id,
            is_private,
            favorites]
    
    
    return output


    
if __name__ == "__main__":
    config = load_config()
    
    key = f"{config['USERNAME']}:{config['API_KEY']}"
    key = base64.b64encode(key.encode()).decode()
    
    headers = {'User-Agent': config["HEADER"],
               'Authorization': f"Basic {key}"}
    
    # Initialization
    source_dir = f'{config["SAVE_DIR"]}/source_users.csv'
    pickle_file = f'{config["SAVE_DIR"]}/source_users_pageID.pkl'
    os.makedirs(config["SAVE_DIR"], exist_ok=True)

    # Get pageID ending point (starting_point)
    url = f'https://e621.net/users.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&page=b999999999'
    response = requests.get(url, headers=headers)
    page = response.json()
    ending_point = page[0]['id']
    

    # Load the last pageID if it exists
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as pf:
            pickle_data = pickle.load(pf)
        pageID = pickle_data['current_pageID']
    else:
        pageID = 0
    
    print(f'Current pageID: {pageID}')
    print(f'Newest post: {ending_point}')


    # Check if CSV file exists and write headers if it doesn't
    csv_headers = ['ID', 'created_at', 'name', 'level',  'base_upload_limit',  'post_upload_count',  'post_update_count',  'is_banned',  'can_approve_posts', 'can_upload_free', 'level_string', 'avatar_id', 'is_private', 'favorites', ]
    if not os.path.exists(source_dir):
        with open(source_dir, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    
    # Getting data
    with ThreadPoolExecutor(max_workers=8) as executor:
        while pageID < ending_point:
            user_data = []

            print(f"Sourcing page a{pageID}")
            url = f'https://e621.net/users.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&page=a{pageID}&limit=320'
            page = fetch_with_retries(url, headers)
            if page is None:
                break 
            

            pageID = page[0]['id']
            
            
            
            futures = [executor.submit(get_user_data, post, headers) for post in page]
            for future in as_completed(futures):
                try:
                    user_data.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")
        


            

            # Update the pickle file with the current pageID and starting point
            pickle_data = {
                'current_pageID': pageID,
            }
            with open(pickle_file, 'wb') as pf:
                pickle.dump(pickle_data, pf)
                
            
            # Update the CSV file
            with open(source_dir, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(user_data)

            
            
    # Apply validation split
    add_valid_column(source_dir, num_valid_rows = 50)
    
    print("Finished Operations")