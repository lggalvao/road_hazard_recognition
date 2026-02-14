import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the CNN-LSTM model with ResNet backbone
class Embedding_CNN_LSTM(nn.Module):
    def __init__(self, cfg):
        super(Embedding_CNN_LSTM, self).__init__()

        self.multi_GPU = cfg.system.multi_gpu
        
        ##General Parameters
        self.device = cfg.system.device
        self.batch_size = cfg.training.batch_size
        
        #Embedding parameters
        self.num_dynamic_features = cfg.data.num_dynamic_features
        self.output_embedding_size = cfg.model.output_embedding_size
        self.dropout_embedding_feature = cfg.model.dropout_embedding_feature
        self.dropout_embedding = nn.Dropout(self.dropout_embedding_feature)


        # Load pre-trained ResNet-18 model
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the fully connected layers at the end
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1,1))

        # 1D CNN layer for dynamic features
        self.cnn_dynamic = nn.Conv1d(in_channels=self.num_dynamic_features, out_channels=self.output_embedding_size, kernel_size=3, padding=1)

        # LSTM layers
        self.enc_hidden_size = cfg.model.enc_hidden_size
        self.enc_layers_num = cfg.model.enc_layers_num
        self.lstm = nn.LSTM(input_size=resnet.fc.in_features + self.output_embedding_size, hidden_size=self.enc_hidden_size, num_layers=self.enc_layers_num, batch_first=True)

        # Dropout layer
        dropout_rate = cfg.model.dropout1
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.num_classes = cfg.model.num_classes
        self.fc = nn.Linear(self.enc_hidden_size, self.num_classes)

    def forward(self, *args):
        
        dynamic_features = args[0]
        time_series_images = args[1]
        
        # ResNet forward pass for time series images
        batch_size, seq_length, c, h, w = time_series_images.size()
        time_series_images = time_series_images.view(batch_size * seq_length, c, h, w)
        resnet_out = self.resnet(time_series_images)

        # Global average pooling to reduce spatial dimensions
        resnet_out = self.AdaptiveAvgPool2d(resnet_out)
        resnet_out = resnet_out.view(batch_size, seq_length, -1)

        # Transpose to match the expected input of the 1D CNN for dynamic features
        #print('dynamic_features:', dynamic_features.shape)
        dynamic_features = dynamic_features.permute(0, 2, 1)
        #print('dynamic_features2:', dynamic_features.shape)

        # 1D CNN forward pass for dynamic features
        cnn_dynamic_out = self.cnn_dynamic(dynamic_features)
        #print('cnn_dynamic_out:', cnn_dynamic_out.shape)
        cnn_dynamic_out = cnn_dynamic_out.permute(0, 2, 1)
        #print('cnn_dynamic_out2:', cnn_dynamic_out.shape)

        # Concatenate ResNet features and dynamic features
        #print('resnet_out:', resnet_out.shape)
        combined_features = torch.cat((resnet_out, cnn_dynamic_out), dim=2)
        #print('combined_features:', combined_features.shape)

        # LSTM forward pass
        if self.multi_GPU == True:
            self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(combined_features)
        #print('lstm_out:', lstm_out.shape)

        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        #print('lstm_out2:', lstm_out.shape)
        lstm_out = self.dropout(lstm_out)

        # Fully connected layer
        output = self.fc(lstm_out)

        return output







# Instantiate the model
if __name__ == "__main__":

    # Generate synthetic time series data
    np.random.seed(42)
    seq_length = 10
    num_samples = 32
    num_dynamic_features = 5  # Number of dynamic features
    num_channels = 3  # Assuming 3 channels in the time series images
    
    # Creating synthetic time series data
    dynamic_features = np.random.randn(num_samples, seq_length, num_dynamic_features).astype(np.float32)
    time_series_images_images = np.random.randn(num_samples, seq_length, 3, 224, 224).astype(np.float32)
    #time_series_images = np.random.randn(num_samples, seq_length, num_channels).astype(np.float32)
    labels = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)
    
    # Convert NumPy arrays to PyTorch tensors
    dynamic_features_tensor = torch.from_numpy(dynamic_features)
    time_series_images_tensor = torch.from_numpy(time_series_images_images)
    labels_tensor = torch.from_numpy(labels)
    
    # Define a custom dataset
    dataset = TensorDataset(dynamic_features_tensor, time_series_images_tensor, labels_tensor)
    
    # DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dynamic_feature_size = num_dynamic_features
    hidden_size = 256
    num_classes = 1  # Binary classification in this example
    num_layers = 1
    
    cfg = Config()
    
    cfg.training.batch_size = batch_size
    cfg.system.device = torch.device('cuda:0')
    cfg.data.num_dynamic_features = num_dynamic_features
    cfg.model.output_embedding_size = 64
    cfg.model.dropout_embedding_feature = 0.1
    cfg.model.enc_hidden_size = hidden_size
    cfg.model.enc_layers_num = num_layers
    cfg.model.num_classes = num_classes

    model = Embedding_CNN_LSTM(cfg).to(cfg.system.device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for dynamic_features, time_series_images, labels in dataloader:
            # Forward pass
            outputs = model(dynamic_features.to(cfg.system.device), time_series_images.to(cfg.system.device))
            loss = criterion(outputs.squeeze(), labels.to(cfg.system.device))
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Make predictions on new data (you would typically use a validation set)
    new_dynamic_features = torch.from_numpy(np.random.randn(1, seq_length, num_dynamic_features).astype(np.float32))
    new_time_series_images = torch.from_numpy(np.random.randn(1, seq_length, 3, 224, 224).astype(np.float32))

    predictions = model(new_dynamic_features.to(cfg.system.device), new_time_series_images.to(cfg.system.device))
    print(f'Predictions: {predictions.item()}')
