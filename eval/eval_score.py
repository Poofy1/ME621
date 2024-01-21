import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from train_score import *
from fastai.vision.all import *



def load_image(img_path, image_size):
    resize_and_pad = SquareResize(image_size)
    
    image = Image.open(img_path).convert('RGB')
    image = resize_and_pad(image)

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    return image


def process_folder(folder_path, image_size, model, device):
    # Generate a list of ages with exponential distribution up to a max of 200
    test_ages = np.logspace(start=0, stop=np.log10(200), num=15, dtype=int)

    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img_name = os.path.basename(img_path)

        # Prepare lists to store results
        upvotes, downvotes, favorites, total_scores = [], [], [], []

        # Process and load image
        image_tensor = load_image(img_path, image_size).to(device)

        for age in test_ages:
            # Create age tensor
            age_tensor = torch.tensor([[age]], dtype=torch.float32).to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(image_tensor, age_tensor).squeeze()
                outputs = outputs.round().int()
                upvotes.append(outputs[0].item())
                downvotes.append(outputs[1].item())
                favorites.append(outputs[2].item())
                total_scores.append(outputs[0].item() + outputs[1].item())  # Sum of upvotes and downvotes

        # Plotting the results for each age
        plt.figure()
        plt.plot(test_ages, upvotes, label='Upvotes', marker='o', color='green')
        plt.plot(test_ages, downvotes, label='Downvotes', marker='o', color='red')
        plt.plot(test_ages, favorites, label='Favorites', marker='o', color='blue')
        plt.plot(test_ages, total_scores, label='Total Score', marker='o', color='black')

        # Adding text to the points
        for i in range(len(test_ages)):
            plt.text(test_ages[i], upvotes[i], str(upvotes[i]), fontsize=10)
            plt.text(test_ages[i], downvotes[i], str(downvotes[i]), fontsize=10)
            plt.text(test_ages[i], favorites[i], str(favorites[i]), fontsize=10)
            plt.text(test_ages[i], total_scores[i], str(total_scores[i]), fontsize=10)

        plt.xlabel('Age')
        plt.ylabel('Scores')
        plt.title(f'Results for Image: {img_name}')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5) # Adding grid lines
        plt.show()
        
        
        
if __name__ == "__main__":
    # Config
    name = 'ME621_Score_2'
    image_size = 300
    dropout_rate = 0.0
    image_folder = f'{current_dir}/test_images/'
    model_path = f"{parent_dir}/models/{name}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = ME621_Model(num_classes=3, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path)['model_state'])
    model.eval()

    # Process all images in the folder
    process_folder(image_folder, image_size, model, device)