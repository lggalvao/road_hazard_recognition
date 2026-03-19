import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

class VideoMAENet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_frames = cfg.model.enc_input_seq_length

        config = VideoMAEConfig.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=cfg.model.num_classes
        )

        config.num_frames = self.num_frames

        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base",
            config=config,
            ignore_mismatched_sizes=True
        )

        self._apply_freezing(cfg.model.freeze_strategy)

    def _apply_freezing(self, strategy):

        if strategy == "head":
            for param in self.model.videomae.parameters():
                param.requires_grad = False

        elif strategy == "partial":
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze last 2 blocks
            for name, param in self.model.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name:
                    param.requires_grad = True

            for param in self.model.classifier.parameters():
                param.requires_grad = True

        elif strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True

        else:
            raise ValueError("Unknown freezing strategy")

    def forward(self, inputs, labels=None):

        images = inputs["images"]   # (B, T, C, H, W)

        outputs = self.model(
            pixel_values=images,
            labels=labels
        )

        return outputs.logits