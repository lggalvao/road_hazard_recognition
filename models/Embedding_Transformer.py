import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for numeric sequences.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, E)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return x

class Embedding_Transformer(nn.Module):
    """
    Transformer for numeric/categorical time series without CNN.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_features = cfg.data.num_dynamic_features
        self.d_model = cfg.model.output_embedding_size
        self.num_classes = cfg.model.num_classes
        self.num_heads = cfg.model.num_heads
        self.num_layers = cfg.model.enc_layers_num
        self.dropout = cfg.model.dropout1

        # Linear embedding for each feature
        self.embedding = nn.Linear(self.num_features, self.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output classifier
        self.fc = nn.Linear(self.d_model, self.num_classes)

    def forward(self, inputs):
        """
        Args:
            inputs: tuple(dynamic_features, mask) or just dynamic_features
                dynamic_features: (B, T, F)
                mask: (B, T) optional, 1=valid, 0=missing
        """
        if isinstance(inputs, tuple):
            x, mask = inputs
        else:
            x = inputs
            mask = None

        # Linear embedding
        x = self.embedding(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Prepare attention mask for missing timesteps
        if mask is not None:
            src_key_padding_mask = mask == 0  # (B, T)
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # Pooling: mean over valid timesteps
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).to(x.dtype)
            x = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)
        else:
            x = x.mean(dim=1)

        # Classification
        logits = self.fc(x)
        return logits