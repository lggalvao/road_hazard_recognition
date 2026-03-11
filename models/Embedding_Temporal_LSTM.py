import torch
import torch.nn as nn
import math
from typing import Optional


class TemporalAttention(nn.Module):
    """
    LSTM-based temporal attention module.

    This module encodes a temporal sequence using an LSTM and applies
    learned attention weights over time to produce a fixed-length
    representation. Optional masking allows the model to ignore
    missing or padded timesteps safely.
    """

    def __init__(self, input_size: int, hidden_size: int, enc_layers_num: int, dropout: float):
        super().__init__()

        # Temporal encoder
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=enc_layers_num,
            batch_first=True,
            dropout=dropout
        )

        # Attention projection (scalar score per timestep)
        self.fc = nn.Linear(hidden_size, 1, bias=False)

        # ---- Best-practice initialization ----
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x:    Tensor of shape (B, T, F)
            mask: Optional tensor of shape (B, T)
                  1 = valid timestep, 0 = missing/padded

        Returns:
            attended_features: (B, H)
            attn_weights:      (B, T)
        """

        # Ensure efficient LSTM execution on GPU
        self.rnn.flatten_parameters()

        # Temporal encoding
        output, _ = self.rnn(x)  # (B, T, H)

        B, T, H = output.shape

        # Compute unnormalized attention scores
        scores = self.fc(output)                  # (B, T, 1)
        scores = scores / math.sqrt(H)            # scale for stability

        # Apply temporal mask if provided
        if mask is not None:
            mask = (mask > 0).unsqueeze(-1)       # (B, T, 1), boolean
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, neg_inf)

        # Normalize over time
        attn_weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted temporal aggregation
        attended_features = (attn_weights * output).sum(dim=1)  # (B, H)

        return attended_features, attn_weights.squeeze(-1)


class Embedding_Temporal_LSTM(nn.Module):
    """
    End-to-end temporal model for numeric/categorical time-series data.

    Architecture:
        1) Temporal embedding via 1D CNN
        2) LSTM-based temporal attention
        3) Fully connected classifier

    Supports missing timesteps via explicit masking.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = cfg.system.device
        self.batch_size = cfg.training.batch_size
        
        self.use_object_visible_side = cfg.model.use_object_visible_side
        self.use_rear_light_status = cfg.model.use_rear_light_status
        self.embedding_size = cfg.model.output_embedding_size

        self.object_type_emb = nn.Embedding(
            cfg.model.num_object_types,
            cfg.model.emb_dim_object_type,
            padding_idx=0
        )
        
        cat_dim = 0
        
        if self.use_object_visible_side:
            self.visible_side_emb = nn.Embedding(
                cfg.model.num_visible_sides,
                cfg.model.emb_dim_visible_side,
                padding_idx=0
            )
            
            cat_dim += cfg.model.emb_dim_visible_side
        
        if self.use_rear_light_status:
            self.rear_light_emb = nn.Embedding(
                cfg.model.num_rear_light_statuses,
                cfg.model.emb_dim_rear_light_status,
                padding_idx=0
            )
            
            cat_dim += cfg.model.emb_dim_rear_light_status
        
        
        # Feature dimensions
        self.num_dynamic_features = (
            cfg.model.num_kinematic_features +
            cfg.model.num_bbox_features +
            cfg.model.emb_dim_object_type +
            cat_dim
        )
        
        # ---- Temporal feature embedding ----
        # CNN operates over time; channels correspond to feature dimensions
        self.cnn_dynamic = nn.Conv1d(
            in_channels=self.num_dynamic_features,
            out_channels=self.embedding_size,
            kernel_size=3,
            padding=1
        )

        # ---- Temporal attention ----
        self.hidden_size = cfg.model.enc_hidden_size
        self.enc_layers_num = cfg.model.enc_layers_num
        self.lstm_dropout = cfg.model.lstm_dropout
        self.temporal_attention = TemporalAttention(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            enc_layers_num=self.enc_layers_num,
            dropout=self.lstm_dropout
        )

        # Regularization
        self.dropout_cnn_dynamic = nn.Dropout(cfg.model.dropout_cnn_dynamic)
        self.dropout_pre_attention = nn.Dropout(cfg.model.dropout_pre_attention)
        self.dropout_fc = nn.Dropout(cfg.model.dropout_fc)
        
        

        # Classification head
        self.num_classes = cfg.model.num_classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # Optional: stabilize CNN output (uncomment if features are noisy)
        self.norm = nn.LayerNorm(self.embedding_size)

    def forward(self, inputs):
        """
        Args:
            inputs: dict with keys
                kinematic: (B, T, K)
                bbox: (B, T, B)
                object_type: (B, T)
                use_object_visible_side: (B, T)
                use_rear_light_status: (B, T)
                missing_object_mask: (B, T)
    
        Returns:
            logits: (B, num_classes)
        """
        kinematic = inputs["kinematic"]          # (B, T, K)
        bbox = inputs["bbox"]                    # (B, T, B)
        object_type = inputs["object_type"]      # (B, T)
        
        mask = inputs.get("missing_object_mask", None)
        
        assert object_type.min().item() >= 0
        assert object_type.dtype == torch.long
        assert object_type.max().item() < self.object_type_emb.num_embeddings
        
        # --------------------------------------------------
        # 1. Embed categorical features
        # --------------------------------------------------
        obj_emb = self.object_type_emb(object_type)          # (B, T, E1)
        
        cat_features = [obj_emb]
        
        if self.use_object_visible_side:
            visible_side = inputs["object_visible_side"]
            assert visible_side.dtype == torch.long
            assert visible_side.min().item() >= 0
            assert visible_side.max().item() < self.visible_side_emb.num_embeddings
            
            side_emb = self.visible_side_emb(visible_side)
            cat_features.append(side_emb)
    
        if self.use_rear_light_status:
            use_rear_light_status = inputs["rear_light_status"]
            assert use_rear_light_status.dtype == torch.long
            assert use_rear_light_status.min().item() >= 0
            assert use_rear_light_status.max().item() < self.rear_light_emb.num_embeddings
        
            light_emb = self.rear_light_emb(use_rear_light_status)
            cat_features.append(light_emb)
        
        cat_emb = torch.cat(cat_features, dim=-1)

        # --------------------------------------------------
        # 2. Concatenate numeric + categorical
        # --------------------------------------------------
        assert isinstance(kinematic, torch.Tensor)
        assert isinstance(bbox, torch.Tensor)
        numeric = torch.cat([kinematic, bbox], dim=-1)       # (B, T, Kn)
        dynamic_features = torch.cat(
            [numeric, cat_emb],
            dim=-1
        )                                                     # (B, T, F)

        assert dynamic_features.shape[2] == self.cnn_dynamic.in_channels, \
            f"Feature mismatch: {dynamic_features.shape[2]} vs {self.cnn_dynamic.in_channels}"
    
        # --------------------------------------------------
        # 3. Temporal CNN embedding
        # --------------------------------------------------
        x = dynamic_features.permute(0, 2, 1)                # (B, F, T)
        x = self.cnn_dynamic(x)
        x = self.dropout_cnn_dynamic(x)
        x = x.permute(0, 2, 1)                                # (B, T, E)
    
        x = self.norm(x)
        x = self.dropout_pre_attention(x)
    
        # --------------------------------------------------
        # 4. Temporal attention
        # --------------------------------------------------
        attended_features, attn_weights = self.temporal_attention(x, mask)
        attended_features = self.dropout_fc(attended_features)
    
        # --------------------------------------------------
        # 5. Classification
        # --------------------------------------------------
        logits = self.classifier(attended_features)
    
        return logits
