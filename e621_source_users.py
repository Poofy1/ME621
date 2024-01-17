import os, csv, pickle, datetime, requests, time, json
import base64
from requests.exceptions import RequestException
import pandas as pd
import numpy as np
env = os.path.dirname(os.path.abspath(__file__))

def load_config():
    with open(f'{env}/config.json', 'r') as config_file:
        config = json.load(config_file)
        return config
    

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
    csv_headers = ['ID', 'Favorites', ]
    if not os.path.exists(source_dir):
        with open(source_dir, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    
    # Getting data
    while pageID < ending_point:
        image_data = []

        print(f"Sourcing page a{pageID}")
        url = f'https://e621.net/users.json?login={config["USERNAME"]}&api_key={config["API_KEY"]}&page=a{pageID}&limit=320'
        page = fetch_with_retries(url, headers)
        if page is None:
            break 
        
        posts = page['posts']
        pageID = posts[0]['id']
        

        for post in posts:

            
            # Columns
            id = post['id']

        
            image_data.append([id,])

        # Update the pickle file with the current pageID and starting point
        pickle_data = {
            'current_pageID': pageID,
        }
        with open(pickle_file, 'wb') as pf:
            pickle.dump(pickle_data, pf)
            
        
        # Update the CSV file
        with open(source_dir, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(image_data)
            
            
            
    # Apply validation split
    add_valid_column(source_dir, split = .025)
    
    print("Finished Operations")