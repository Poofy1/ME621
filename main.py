import os
import json
import telegram
import aiohttp
import asyncio
from urllib.request import FancyURLopener
import torch
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import warnings

# Initialize 
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = os.path.dirname(os.path.abspath(__file__))
bot : telegram.Bot

class MyOpener(FancyURLopener):
    version = 'ME621 1.0 (by Poof75 on e621)'
myopener = MyOpener()


model_name = "main8"
image_size = 224
threshold = 0.75


def load_last_image_id():
    if os.path.exists(f"{env}/last_image_id.txt"):
        with open(f"{env}/last_image_id.txt", "r") as file:
            return int(file.read().strip())
    return None

# Define transform for the test images
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

async def scrape_new_images():
    print("Finding new images")
    
    page = json.loads(myopener.open('https://e621.net/posts.json?page=b0&limit=1&tags=-animated').read().decode("utf-8"))
    
    newest_image_id = page['posts'][0]['id']
    last_image_id = load_last_image_id()

    if last_image_id is None:
        last_image_id = newest_image_id - 1
    elif newest_image_id <= last_image_id:
        return

    image_data = []
    pageID = newest_image_id
    
    if pageID > last_image_id:
        try:
            page = json.loads(myopener.open(f'https://e621.net/posts.json?page=b0&tags=rating%3Aexplicit+-animated&limit=320').read().decode("utf-8"))
            for i in range(len(page['posts'])):
                post = page['posts'][i]
                url = post['sample']['url']
                num = post['id']
                if url is not None and num > last_image_id:
                    image_data.append([num, url])
                pageID = num
        except Exception as e:
            print(f"Page b{pageID} Failed: {e}")
    
    # Save the newest_image_id as the last_image_id for future runs
    with open(f"{env}/last_image_id.txt", "w") as file:
        file.write(str(newest_image_id))

    print(f"Testing {len(image_data)} images")
    await download_images_sequentially(image_data)
    print(f"Batch finished\n")



async def download_images_sequentially(image_data):
    # Load Model
    model = torch.load(f"{env}/models/{model_name}.pt").to(device)
    model.eval()

    async with aiohttp.ClientSession() as session:
        for image_info in image_data:
            try:
                image_id, image_url = image_info
                async with session.get(image_url) as response:
                    downloaded_image_data = await response.read()
                image = Image.open(BytesIO(downloaded_image_data))
                await Evaluate(model, image, image_id)
            except Exception as e:
                print(f"ERROR: {e}")
    
    


async def Evaluate(model, image, image_id):
    # Classify test images
    try:
        image_input = image.convert("RGB")
        test_image_tensor = test_transform(image_input).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(test_image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_prob = probabilities[0, 1].item()
    except:
        print(f"Failed test")
        return

    if predicted_prob >= threshold:
        print(f"Image {image_id} Passed: {predicted_prob:.3f}")

        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        await bot.send_photo(chat_id=-1001930008144, caption=f"Image ID: {image_id}\n{(predicted_prob * 100):.1f}% Confidence", photo=image_bytes)
        
     
     
async def botmain():
    global bot
    bot = telegram.Bot("6168112426:AAFuH0S3gJyuLT5crDZHGoVHN1g1RMBGEqs")


async def main():
    await botmain()
    await scrape_new_images()

if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
