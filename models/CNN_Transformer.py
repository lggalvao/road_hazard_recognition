import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


class CNN_Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = cfg.system.device

        # --------------------------
        # CNN backbone
        # --------------------------
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        cnn_feature_dim = resnet.fc.in_features  # 512

        # --------------------------
        # Projection to hidden size
        # --------------------------
        self.hidden_size = cfg.model.enc_hidden_size

        self.feature_proj = nn.Linear(
            cnn_feature_dim,
            self.hidden_size
        )

        # --------------------------
        # Positional Encoding
        # --------------------------
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 16, self.hidden_size)
        )

        # --------------------------
        # Transformer Encoder
        # --------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.dropout = nn.Dropout(cfg.model.dropout1)

        # --------------------------
        # Classifier
        # --------------------------
        self.fc = nn.Linear(
            self.hidden_size,
            cfg.model.num_classes
        )

    def forward(self, inputs):

        images = inputs["images"]
        mask = inputs.get("missing_object_mask", None)

        B, T, C, H, W = images.size()

        # --------------------------
        # CNN feature extraction
        # --------------------------
        images = images.view(B * T, C, H, W)
        feats = self.resnet(images)
        feats = self.avgpool(feats)
        feats = feats.view(B, T, -1)  # (B,T,512)

        # --------------------------
        # Project to hidden size
        # --------------------------
        x = self.feature_proj(feats)  # (B,T,H)

        # --------------------------
        # Add positional encoding
        # --------------------------
        x = x + self.pos_embedding[:, :T, :]

        # --------------------------
        # Proper transformer mask
        # --------------------------
        if mask is not None:
            # Transformer expects True for padding
            key_padding_mask = mask == 0
        else:
            key_padding_mask = None

        x = self.temporal_transformer(
            x,
            src_key_padding_mask=key_padding_mask
        )

        # --------------------------
        # Masked mean pooling (better than last index)
        # --------------------------
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # --------------------------
        # Classification
        # --------------------------
        x = self.dropout(x)
        logits = self.fc(x)

        return logits



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

