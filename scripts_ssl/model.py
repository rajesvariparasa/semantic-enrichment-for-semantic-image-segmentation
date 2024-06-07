import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from typing import Optional, Union, List, OrderedDict

class UNetWithDropout(smp.Unet):
    def __init__(self, 
        encoder_name: str = "resnet50",
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



class UNetWithDropout_AuxiTask_v0(smp.Unet):
    def __init__(self, 
        encoder_name: str = "resnet50",
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
    
        num_heads= 2 # one for segmentation, one for reconstruction
        self.log_vars = nn.Parameter(torch.zeros(num_heads))

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
            return masks, reconstructions, self.log_vars

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels, self.log_vars

        return masks, self.log_vars



class UNetWithDropout_AuxiTask_v1(smp.base.SegmentationModel):  # Extend from SegmentationModel
    def __init__(self, 
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        add_reconstruction_head: bool = False,
        dropout_prob: float = 0.5,
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = smp.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if add_reconstruction_head:
            self.reconstruction_head = smp.base.ReconstructionHead(
                in_channels=decoder_channels[-1],
                out_channels=in_channels,
                kernel_size=1,
            )
        else:
            self.reconstruction_head = None

        if aux_params is not None:
            self.classification_head = smp.base.ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = f"u-{encoder_name}"
        self.initialize()

        self.log_vars = nn.Parameter(torch.zeros(2))
        self.dropout_prob = dropout_prob

        for decoder_block in self.decoder.blocks:
            decoder_block.add_module("dropout", nn.Dropout2d(p=self.dropout_prob))

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debug statement
        self.check_input_shape(x)

        features = self.encoder(x)
        print(f"Features shape: {[f.shape for f in features]}")  # Debug statement
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        reconstructions = None
        if self.reconstruction_head is not None:
            reconstructions = self.reconstruction_head(decoder_output)
            return masks, reconstructions, self.log_vars

        labels = None
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels, self.log_vars

        return masks




# modified from implementation by Felix Krober
class SMP_Unet_Multitask_v0(nn.Module):
    """
    inspired by...
    Felix Krober's implementation of a multi-task learning model,
    Audebert 2019: Distance transform regression for spatially-aware deep semantic segmentation
    """
    def __init__(self, in_channels, classes, encoder_name,encoder_weights ):
        super(SMP_Unet_Multitask_v0, self).__init__()
        # create uncertainty params for weighting of tasks
        n_tasks = 2
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        # segmentation model (core)
        self.seg_model = smp.create_model(
            arch="unet",
            encoder_name=encoder_name,
            classes=classes,
            in_channels=in_channels,
            encoder_weights=encoder_weights,
        )

        self.encoder = self.seg_model.encoder
        self.decoder = self.seg_model.decoder

        # segmentation head
        self.seg_head = nn.Sequential(
            OrderedDict(
                [
                    # in_channels=16 - is the number of output channels of the last decoder block
                    ("conv2d", nn.Conv2d(in_channels=16, out_channels=classes,
                                          kernel_size=(3, 3), padding=1)),
                ]
            )
        )
        # regression head
        self.reg_head = nn.Sequential(
            OrderedDict(
                [
                    ("conv2d", nn.Conv2d(in_channels=16, out_channels=in_channels, 
                                         kernel_size=(3, 3), padding=1)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        # core model to produce feature maps
        feats = self.seg_model.encoder(x)

        feats = self.seg_model.decoder(*feats)
        # segmentation output
        seg_out = self.seg_head(feats)
        # regression output
        reg_out = self.reg_head(feats)
        # return predictions from both tasks & uncertainties
        return seg_out, reg_out, self.log_vars