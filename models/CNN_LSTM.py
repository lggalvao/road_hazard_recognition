import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


class CNN_LSTM(nn.Module):
    def __init__(self, cfg):
        super(CNN_LSTM, self).__init__()

        ##General Parameters
        self.device = cfg.system.device
        self.batch_size = cfg.training.batch_size
        
        # Load pre-trained ResNet-18 model
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the fully connected layers at the end
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        self.avgpool  = nn.AdaptiveAvgPool2d((1,1))
    
        # LSTM layers
        self.enc_hidden_size = cfg.model.enc_hidden_size
        self.enc_layers_num = cfg.model.enc_layers_num
        cnn_feature_dim = resnet.fc.in_features
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=self.enc_hidden_size,
            num_layers=self.enc_layers_num,
            batch_first=True
        )
        
        # Dropout layer
        dropout_rate = cfg.model.dropout1
        self.dropout = nn.Dropout(dropout_rate)
    
        # Fully connected layer
        self.num_classes = cfg.model.num_classes
        self.fc = nn.Linear(self.enc_hidden_size, self.num_classes)

    def forward(self, inputs):
        """
        inputs:
            x: Tensor (B, T, C, H, W)
            missing_object_mask: Tensor (B, T) with {0,1}
        """
    
        images = inputs["images"]
        missing_object_mask = inputs.get("missing_object_mask", None)
        
        #x, missing_object_mask = inputs
        B, T, C, H, W = images.size()
    
        # ---------------------------
        # CNN feature extraction
        # ---------------------------
        images = images.view(B * T, C, H, W)
        resnet_out = self.resnet(images)
        resnet_out = self.avgpool(resnet_out)
        resnet_out = resnet_out.view(B, T, -1)  # (B, T, F)
    
        # ---------------------------
        # Apply temporal mask (robust)
        # ---------------------------
        mask = missing_object_mask.float()  # (B, T)
        resnet_out = resnet_out * mask.unsqueeze(-1)
    
        # ---------------------------
        # LSTM
        # ---------------------------
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(resnet_out)  # (B, T, H)
    
        # ---------------------------
        # Find LAST valid timestep (non-contiguous safe)
        # ---------------------------
        # mask example: [1, 1, 1, 0, 1]
        # reversed_mask example: [1, 0, 1, 1, 1]
        #reversed mask is used because torch.argmax finds the first occurrence of the maximum value
        reversed_mask = torch.flip(mask, dims=[1])
        last_valid_from_end = reversed_mask.argmax(dim=1)
        #T -1 becaseu array start from 0
        last_valid_idx = (T - 1) - last_valid_from_end
    
        # Safety: handle fully-missing sequences
        has_valid = mask.sum(dim=1) > 0
        last_valid_idx = torch.where(
            has_valid,
            last_valid_idx,
            torch.zeros_like(last_valid_idx)
        )
        batch_idx = torch.arange(B, device=images.device)
        lstm_out = lstm_out[batch_idx, last_valid_idx, :]  # (B, H)
    
        # ---------------------------
        # Classification head
        # ---------------------------
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
    
        return output


if __name__ == "__main__":

    import os
    import sys
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, PROJECT_ROOT)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from utils.config import Config

    torch.manual_seed(42)
    np.random.seed(42)

    # ---------------------------
    # Synthetic data parameters
    # ---------------------------
    seq_length = 10
    num_samples = 16
    batch_size = 8
    num_classes = 1

    # ---------------------------
    # Synthetic image data
    # ---------------------------
    images = np.random.randn(
        num_samples, seq_length, 3, 224, 224
    ).astype(np.float32)

    # ---------------------------
    # Synthetic NON-CONTIGUOUS mask
    # Example: [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    # ---------------------------
    masks = np.random.randint(
        0, 2, size=(num_samples, seq_length)
    ).astype(np.float32)

    # Ensure at least one valid frame per sequence
    masks[:, 0] = 1

    # Labels
    labels = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)

    # ---------------------------
    # Custom Dataset
    # ---------------------------
    class DebugDataset(Dataset):
        def __init__(self, images, masks, labels):
            self.images = torch.from_numpy(images)
            self.masks = torch.from_numpy(masks)
            self.labels = torch.from_numpy(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return (
                self.images[idx],      # (T, C, H, W)
                self.masks[idx]        # (T,)
            ), self.labels[idx]

    dataset = DebugDataset(images, masks, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # ---------------------------
    # Model configuration
    # ---------------------------
    cfg = Config()
    cfg.training.batch_size = batch_size
    cfg.system.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.model.enc_hidden_size = 256
    cfg.model.enc_layers_num = 1
    cfg.model.num_classes = num_classes
    cfg.model.dropout1 = 0.3

    model = CNN_LSTM(cfg).to(cfg.system.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------------
    # Debug training loop
    # ---------------------------
    model.train()
    for epoch in range(2):
        for (images_batch, mask_batch), labels_batch in dataloader:

            images_batch = images_batch.to(cfg.system.device)
            mask_batch = mask_batch.to(cfg.system.device)
            labels_batch = labels_batch.to(cfg.system.device)

            # Forward pass (DEBUG TARGET)
            outputs = model((images_batch, mask_batch))

            loss = criterion(outputs.squeeze(), labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/2] | Loss: {loss.item():.4f}")

    # ---------------------------
    # Single-sample debug inference
    # ---------------------------
    model.eval()
    with torch.no_grad():
        test_images = torch.randn(1, seq_length, 3, 224, 224).to(cfg.system.device)

        # Example pathological mask: [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        test_mask = torch.tensor(
            [[1, 1, 0, 1, 0, 0, 1, 0, 1, 0]],
            dtype=torch.float32
        ).to(cfg.system.device)

        pred = model((test_images, test_mask))
        print("Test prediction:", torch.sigmoid(pred).item())

