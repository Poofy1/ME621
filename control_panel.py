from train import *
from model_dignostic import *


model_name = "main10"

#Training
train = False
continue_training = False
batch_size = 32
num_epochs = 5

#Dataset
dataset_update = True

#Debug
dataset_validation = False

#Feature Analysis
mass_image_test = False
feature_analysis = False
heat_map_analysis = False
dream = False





if __name__ == '__main__':
    
    if train:
        train_model(model_name, continue_training, batch_size, num_epochs)
    
    if dataset_update:
        create_dataset("D:/DATA/E621/sorted/", "F:/CODE/ME621/Dataset/", 0.1)

    if dataset_validation:
        #remove_bad_images("D:/DATA/E621/sorted/0/")
        #remove_bad_images("D:/DATA/E621/sorted/1/")
        remove_bad_images("D:/DATA/E621/untested/")

    if mass_image_test:
        image_test(model_name, "D:/DATA/E621/untested/", True, 0.95)
        
    if feature_analysis:
        model_features(model_name, batch_size)
    
    test_image = "D:/DATA/E621/sorted/1/0b2d690cbab0ed10ce213e833b10b5c1.jpg"
    
    if heat_map_analysis:
        heat_map(model_name, test_image)
        
    if dream:
        deep_dream(model_name, test_image, 50, .0001)