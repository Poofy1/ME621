from train import *
from model_dignostic import *


model_name = "main10"

#Training
train = False
continue_training = False
batch_size = 32
num_epochs = 3

#Debug
dataset_validation = False

#Feature Analysis
mass_image_test = True
feature_analysis = False
heat_map_analysis = False
dream = False





if __name__ == '__main__':
    
    if train:
        train_model(model_name, continue_training, batch_size, num_epochs)
    
    

    if dataset_validation:
        remove_bad_images("D:/DATA/E621/sorted/0/")
        remove_bad_images("D:/DATA/E621/sorted/1/")

    if mass_image_test:
        image_test(model_name, "D:/DATA/E621/sorted/0/", True, 0.9)
        
    if feature_analysis:
        model_features(model_name, batch_size)
    
    test_image = "D:/DATA/E621/sorted/1/0b2d690cbab0ed10ce213e833b10b5c1.jpg"
    
    if heat_map_analysis:
        heat_map(model_name, test_image)
        
    if dream:
        deep_dream(model_name, test_image, 50, .0001)