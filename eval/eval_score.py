import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import cv2
from train_score import *
from fastai.vision.all import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(img_path, image_size):
    resize_and_pad = SquareResize(image_size)
    
    image = Image.open(img_path).convert('RGB')
    image = resize_and_pad(image)

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    return image

def visualize_heatmap(model, image_folder, results_dir, image_size):
    os.makedirs(results_dir, exist_ok=True)

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img_name = os.path.basename(img_path)

        # Process and load image
        image_tensor = load_image(img_path, image_size).to(device)
        
        age = 24  # Set the age, modify as needed

        # Forward pass to get feature maps and outputs
        with torch.no_grad():
            outputs, features = model(image_tensor, torch.tensor([[age]], dtype=torch.float32).to(device))
            features = features.squeeze(0)  # Remove batch dimension
            outputs = outputs.squeeze(0).round().int()

        # Get weights from the last layer of the classifier
        params = list(model.classifier.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        # Initialize CAMs for each class
        cams = []
        for idx in range(weight_softmax.shape[0]):  # Iterate over number of classes
            cam = np.zeros(features.shape[1:], dtype=np.float32)  # Shape 11x11
            for i, w in enumerate(weight_softmax[idx]):  # Iterate over 1280 features
                cam += w * features[i, :, :].cpu().numpy()

            cam = np.maximum(cam, 0)  # Apply ReLU
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cam = np.uint8(255 * cam)
            cams.append(cv2.resize(cam, (image_size, image_size)))

        # Also load and process the original image for display
        resize_and_pad = SquareResize(image_size)
        original_image = Image.open(img_path).convert('RGB')
        original_image = resize_and_pad(original_image)
        original_image = np.array(original_image)

        # Plotting
        fig, axs = plt.subplots(1, len(cams) + 2, figsize=(20, 5), gridspec_kw={'wspace':0, 'hspace':0})

        axs[0].imshow(original_image)
        axs[0].set_title('Original Image')

        class_names = ['Upvote Map', 'Downvote Map', 'Favorite Map']
        for i, cam in enumerate(cams):
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_HOT)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            result = heatmap * 0.3 + original_image * 0.5
            axs[i+1].imshow(result.astype('uint8'))
            axs[i+1].set_title(class_names[i])

        for ax in axs:
            ax.axis('off')

        # The last subplot for displaying text
        fav_ratio = outputs[2] / outputs[0] if outputs[0] != 0 else 0
        axs[4].axis('off')  
        text_str = f"Age: {age}\n" 
        text_str += f"Upvotes: {outputs[0]}\n"  
        text_str += f"Downvotes: {outputs[1]}\n" 
        text_str += f"Favorites: {outputs[2]}\n" 
        text_str += f"Fav Ratio: {fav_ratio:.3f}\n"

        axs[-1].text(0.5, 0.5, text_str, ha='center', va='center', fontsize=12)

        # Adjust the layout as before
        plt.subplots_adjust(left=0, right=1, top=.95, bottom=0, wspace=0, hspace=0)
        
        plt.savefig(f'{results_dir}/{img_name}_map.png', bbox_inches='tight')
        plt.close()

        
def process_folder(model, image_folder, results_dir, image_size):
    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Prepare a dictionary to store results for each image
    results = {}

    # Define ages for evaluation
    ages = [1, 6, 12, 24, 36, 48, 72]

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img_name = os.path.basename(img_path)

        # Initialize a list to store the total scores (upvotes + downvotes) for each age
        total_scores_per_age = []

        for age in ages:
            # Process and load image
            image_tensor = load_image(img_path, image_size).to(device)
            
            # Create age tensor
            age_tensor = torch.tensor([[age]], dtype=torch.float32).to(device)

            # Perform inference
            with torch.no_grad():
                outputs, features = model(image_tensor, age_tensor)
                outputs = outputs.squeeze(0).round().int()
                features = features.squeeze(0)
                # Calculate the total score (upvotes + downvotes) and append to the list
                total_scores_per_age.append(outputs[0].item() + outputs[1].item())

        # Store the results in the dictionary
        results[img_name] = total_scores_per_age

    # Plotting the results
    plt.figure()
    for img_name, scores in results.items():
        plt.plot(ages, scores, label=f'{img_name}', marker='o')  # Plot each image's scores
        # Annotate each point with its score
        for age, score in zip(ages, scores):
            plt.text(age, score, str(score), fontsize=8, ha='right', va='bottom')

    # Set x-axis ticks to ages
    plt.xticks(ages)

    plt.xlabel('Age')
    plt.ylabel('Total Score (Upvotes + Downvotes)')
    plt.title('Total Scores by Age for Each Image')
    plt.legend()

    # Enable grid lines
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', alpha=0.5)

    plt.savefig(f'{results_dir}/result_popularity.png', bbox_inches='tight')
    plt.close()
        
        


if __name__ == "__main__":
    # Config
    name = 'ME621_Score_2'
    image_size = 350
    dropout_rate = 0.0
    image_folder = f'{current_dir}/test_images/'
    results_dir = f'{current_dir}/results/'
    model_path = f"{parent_dir}/models/{name}.pt"

    # Load Model
    model = ME621_Model(num_classes=3, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path)['model_state'])
    model.eval()


    print("Finding Heat Maps")
    visualize_heatmap(model, image_folder, results_dir, image_size)
    
    # Process all images in the folder
    print("Finding Popularity")
    process_folder(model, image_folder, results_dir, image_size)
    
    
    