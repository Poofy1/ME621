import os, sys
import telegram
import aiohttp
import asyncio
import requests
import torch
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
SAVE_DIR = os.path.join(parent_dir, 'data')

sys.path.append(parent_dir)
from model.trainer import load_model, FurryClassifier
from config import global_config, load_config
from e621_backend.annotation import get_max_page_id

# Initialize 
warnings.filterwarnings("ignore", category=DeprecationWarning)
bot : telegram.Bot


def load_last_image_id():
    if os.path.exists(f"{current_dir}/last_image_id.txt"):
        with open(f"{current_dir}/last_image_id.txt", "r") as file:
            return int(file.read().strip())
        
    return get_max_page_id() - 100



async def scrape_new_images():
    print("Finding new images")
    
    last_image_id = load_last_image_id()
    pageID = last_image_id
    
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    image_size = 256
    model = FurryClassifier(model, image_size)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/models/ME621.pth"))
    model.to(device)
    model.eval()
    
    while True:
        try:
            response = requests.get(f'https://e621.net/posts.json?login={global_config["config"]["USERNAME"]}&api_key={global_config["config"]["E621_API"]}&page=a{pageID}&tags=-animated&limit=100', headers=global_config['headers'])
            page = response.json()
            if not page['posts']:
                break
            
            image_data = []
            for i in range(len(page['posts'])):
                post = page['posts'][i]
                url = post['sample']['url']
                num = post['id']
                if url is not None and num > last_image_id:
                    image_data.append([num, url])
                pageID = num
            
            print(f"Testing {len(image_data)} images")
            async with aiohttp.ClientSession() as session:
                for image_info in image_data:
                    try:
                        image_id, image_url = image_info
                        response = requests.get(image_url, headers=global_config['headers'])
                        downloaded_image_data = response.content
                        image = Image.open(BytesIO(downloaded_image_data)).convert("RGB")
                        await Evaluate(model, image, image_id, image_size)
                    except Exception as e:
                        print(f"ERROR: {e}")
            
            # Save the newest_image_id as the last_image_id for future runs
            with open(f"{current_dir}/last_image_id.txt", "w") as file:
                file.write(str(pageID))

    
        except Exception as e:
            print(f"Page a{pageID} Failed: {e}")
            import traceback
            print(traceback.format_exc())
            break
    
    

    
    print(f"All Images Tested\n")

    
    
    

async def Evaluate(model, image, image_id, image_size, me621_threshold=0.8):
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Classify test images
    device = next(model.parameters()).device
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(image_tensor)
            
    predicted = torch.sigmoid(output).item()  # Convert to scalar

    if predicted >= me621_threshold:
        print(f"Image ID: {image_id} | {predicted * 100:.1f}% Good (PASSED)")

        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        await bot.send_photo(chat_id=global_config['config']['TELEGRAM_GROUP_ID'], caption=f"Image ID: {image_id}\n{predicted * 100:.1f}% Confidence", photo=image_bytes)
    else:
        print(f"Image ID: {image_id} | {predicted * 100:.1f}% Good")

    
async def botmain():
    global bot
    bot = telegram.Bot(global_config['config']['TELEGRAM_API'])


async def main():
    await botmain()
    await scrape_new_images()

def launch_bot(print=print):
    load_config()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())



if __name__ == "__main__":
    launch_bot()