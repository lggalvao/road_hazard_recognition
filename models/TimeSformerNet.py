import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification, TimesformerConfig

class TimeSformerNet(nn.Module):
    def __init__(self, cfg):
        """
        Args (dict):
            num_classes (int): Number of behavior classes for classification
            enc_input_seq_length (int): Number of frames (T)
            freeze_strategy (str): "head", "partial", or "full"
        """
        super(TimeSformerNet, self).__init__()
        
        # Load pretrained config and adjust
        config = TimesformerConfig.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=cfg.model.num_classes
        )
        config.num_frames = cfg.model.enc_input_seq_length  # <-- set here
        config.image_size = 224                          # must match your frame size
        config.num_channels = 3                          # RGB
        
        # Load pretrained model with adjusted config
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            config=config,
            ignore_mismatched_sizes=True
        )

        # Apply freezing strategy
        self._apply_freezing(cfg.model.freeze_strategy) 

    def _apply_freezing(self, strategy):
        """Freezing strategies: head / partial / full"""
        if strategy == "head":
            # Freeze backbone, train classifier only
            for param in self.model.timesformer.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        elif strategy == "partial":
            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            # Unfreeze last 2 transformer encoder blocks
            for name, param in self.model.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name:
                    param.requires_grad = True

        elif strategy == "full":
            # Train entire model
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown freezing strategy: {strategy}")

    def forward(self, inputs, labels=None):
        """
        Args:
            x: Video tensor [B, T, C, H, W] (float, normalized pixel values)
            labels: Optional class labels [B] for computing loss
        Returns:
            ModelOutput (with logits and optionally loss)
        """
        images = inputs["images"]
        
        outputs = self.model(pixel_values=images, labels=labels)
        
        return outputs.logits
