import torch
import torchvision.models as models
import torch.nn as nn




class ME621_Model(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, embedding_dim=512, interpreter_fc_size=256):
        super(ME621_Model, self).__init__()

        # CNN feature extractor
        self.efficientnet = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        
        # Norm Layers
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.features_norm = nn.BatchNorm1d(embedding_dim)
        
        # New layer to match feature size to embedding size
        self.feature_transform = nn.Linear(num_ftrs, embedding_dim)

        # Preference Embedding
        self.preference_embedding = nn.Parameter(torch.zeros(embedding_dim))

        # Intermediate FC layer
        self.intermediate_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, interpreter_fc_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        

        # Prediction Layer
        self.prediction_layer = nn.Linear(interpreter_fc_size, num_classes)
            
    def reset_preference_embedding(self):
        #Reset the preference embedding to its initial state (e.g., all zeros).
        self.preference_embedding.data.fill_(0.0)

    def forward(self, image, mode='prediction', train = False):
        # Extract features from the image using the CNN
        image_features = self.efficientnet.features(image)
        x = self.efficientnet.avgpool(image_features)
        x = torch.flatten(x, 1)

        # Transform the image features to match embedding size
        x_transformed = self.feature_transform(x)    

        if mode == 'siamese':
            
            # normalize features
            x_transformed = self.features_norm(x_transformed)
            
            # Assuming images come in pairs: [neg, pos, neg, pos, ...]
            neg_features = x_transformed[0::2]  # Negative features
            pos_features = x_transformed[1::2]  # Positive features

            # Update the preference embedding
            if not train:
                self.update_preference_embedding(neg_features, pos_features)
            
            return neg_features, pos_features
        
        elif mode == 'prediction':
            # Combine transformed image features with user's preference embedding
            preference_expanded = self.preference_embedding.unsqueeze(0).expand(x_transformed.size(0), -1)
            combined = torch.cat((preference_expanded, x_transformed), dim=1) 

            # Pass the combined features through the intermediate FC layer
            intermediate_output = self.intermediate_fc(combined)

            # Predict and return the likelihood of the user liking the image
            prediction = torch.sigmoid(self.prediction_layer(intermediate_output)).squeeze(-1)

            return prediction
    
    

    def update_preference_embedding(self, neg_image_features, pos_image_features, alpha=0.01):
        """
        Update the preference embedding based on a pair of images.
        - pos_image_features: extracted features of the liked (positive) images
        - neg_image_features: extracted features of the not liked (negative) images
        - alpha: learning rate for updating the embedding
        """

        # Update direction is +1 for positive (liked) and -1 for negative (not liked)
        pos_update_direction = 1
        neg_update_direction = -1

        # Calculate updates for both positive and negative images
        pos_update = alpha * (pos_update_direction * (pos_image_features - self.preference_embedding))
        neg_update = alpha * (neg_update_direction * (neg_image_features - self.preference_embedding))

        # Combine updates and apply to the preference embedding
        self.preference_embedding.data += (pos_update + neg_update).sum(dim=0)
        
        # Normalize the preference embedding after the update
        self.preference_embedding.data = self.embedding_norm(self.preference_embedding.unsqueeze(0)).squeeze(0)




def contrastive_loss(neg_features, pos_features, margin=1.0):
    """
    Calculates a more complete contrastive loss between negative and positive features.

    Args:
    - neg_features (Tensor): Features from the negative (dissimilar) images.
    - pos_features (Tensor): Features from the positive (similar) images.
    - margin (float): The margin for dissimilar images.

    Returns:
    - Tensor: The contrastive loss.
    """
    # Calculate Euclidean distances
    positive_loss = torch.norm(pos_features - neg_features, p=2, dim=1)
    negative_distance = torch.norm(pos_features + neg_features, p=2, dim=1)

    # Loss for negative pairs (dissimilar) - should be large
    negative_loss = torch.relu(margin - negative_distance)

    # Combine losses
    combined_loss = positive_loss + negative_loss

    # Average loss over the batch
    return combined_loss.mean()