import segmentation_models_pytorch as smp
import torch.nn as nn
from typing import Optional, Union, List

class UNetWithDropout(smp.Unet):
    def __init__(self, 
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        # whether to add a reconstruction head
        add_reconstruction_head: bool = False,      #Added by me
        dropout_prob: float = 0.5,                  #Added by me
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            aux_params=aux_params,
    
        )
        self.dropout_prob = dropout_prob

        # Add dropout to each decoder block
        for decoder_block in self.decoder.blocks:
            decoder_block.add_module("dropout", nn.Dropout2d(p=self.dropout_prob))

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        
        # Apply dropout to each block's output
        for block in self.decoder.blocks:
            block = nn.Sequential(
                block,
                nn.Dropout2d(p=self.dropout_prob)
            )

        masks = self.segmentation_head(decoder_output)

        if self.reconstruction_head is not None:
            reconstructions = self.reconstruction_head(decoder_output)
            return masks, reconstructions

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

