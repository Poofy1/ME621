import urllib.request
from urllib.request import FancyURLopener
import json, os, warnings, urllib, csv
import concurrent.futures
import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
class MyOpener(FancyURLopener):
    version = 'Dataset Creator 1.0 (by Poof75 on e621)'
myopener = MyOpener()


image_data = []
page = json.loads(myopener.open('https://e621.net/posts.json?page=b0&limit=1&tags=-animated').read().decode("utf-8"))
pageID = page['posts'][0]['id']

limit = 999999999999
source = False
download = True

if source:
    while len(page['posts']) != 0 and limit > 0:
        try:
            print(f"Downloading page b{pageID}")
            page = json.loads(myopener.open(f'https://e621.net/posts.json?page=b{pageID}&tags=rating%3Aexplict+-animated&limit=320').read().decode("utf-8"))
            for i in range(len(page['posts'])):
                if limit > 0:
                    page2 = page['posts'][i]
                    url = page2['sample']['url'] #'file' for better quality 
                    num = page2['id']
                    if url != None:
                        image_data.append([num, url])
                        limit -= 1
                pageID = num
        except:
            print(f"Page b{pageID} Failed")

    with open('D:/DATA/E621/links.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(image_data)


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
    
    with open('D:/DATA/E621/links.csv', 'r') as f:
        reader = csv.reader(f)
        image_data = list(reader)

    if download:
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