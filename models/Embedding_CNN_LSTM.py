import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models.Embedding_Temporal_LSTM import Embedding_Temporal_LSTM
from models.CNN_LSTM import CNN_LSTM

class Embedding_CNN_LSTM(nn.Module):
    def __init__(self, cfg, embedding_model, cnn_model):
        super().__init__()

        self.cnn_model = cnn_model
        self.embedding_model = embedding_model

        # Remove internal classifiers
        self.cnn_model.fc = nn.Identity()
        self.embedding_model.fc = nn.Identity()

        H = cfg.model.enc_hidden_size

        # ---- Projection ----
        self.proj_cnn = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.GELU()
        )

        self.proj_embed = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.GELU()
        )

        # ---- Cross Attention ----
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=H,
            num_heads=4,
            batch_first=True
        )

        # ---- Gating ----
        self.gate = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.Sigmoid()
        )

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout1),
            nn.Linear(H // 2, cfg.model.num_classes)
        )

    def forward(self, inputs):

        cnn_feat = self.cnn_model(inputs)         # (B,H)
        embed_feat = self.embedding_model(inputs) # (B,H)

        # Project
        cnn_feat = self.proj_cnn(cnn_feat)
        embed_feat = self.proj_embed(embed_feat)

        # Add sequence dimension for attention
        cnn_feat_seq = cnn_feat.unsqueeze(1)      # (B,1,H)
        embed_feat_seq = embed_feat.unsqueeze(1)  # (B,1,H)

        # Cross attention
        attn_output, _ = self.cross_attn(
            query=cnn_feat_seq,
            key=embed_feat_seq,
            value=embed_feat_seq
        )

        attn_output = attn_output.squeeze(1)  # (B,H)

        # Gated fusion
        fusion_input = torch.cat([cnn_feat, attn_output], dim=1)
        gate = self.gate(fusion_input)

        fused = gate * cnn_feat + (1 - gate) * attn_output

        logits = self.classifier(fused)

        return logits



#class Embedding_CNN_LSTM(nn.Module):
#    def __init__(self, cfg, embedding_model, cnn_model):
#        super().__init__()
#
#        self.cnn_model = cnn_model
#        self.embedding_model = embedding_model
#
#        # Remove branch classifiers
#        self.cnn_model.fc = nn.Identity()
#        self.embedding_model.fc = nn.Identity()
#
#        fusion_dim = cfg.model.enc_hidden_size * 2
#
#        self.classifier = nn.Sequential(
#            nn.Linear(fusion_dim, fusion_dim // 2),
#            nn.ReLU(),
#            nn.Dropout(cfg.model.dropout1),
#            nn.Linear(fusion_dim // 2, cfg.model.num_classes)
#        )
#
#    def forward(self, inputs):
#
#        cnn_feat = self.cnn_model(inputs)         # (B,H)
#        embed_feat = self.embedding_model(inputs) # (B,H)
#
#        fused = torch.cat([cnn_feat, embed_feat], dim=1)
#
#        logits = self.classifier(fused)
#
#        return logits

