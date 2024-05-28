import torch
import torchgeometry as tgm
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchmetrics import R2Score
import kornia

# reconstruction loss
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target):
        return nn.L1Loss()(input, target)
    

# segmentation losses

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        return smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)(input, target)
    
class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, input, target):
        return nn.CrossEntropyLoss()(input, target)    

def load_loss(loss_name):

    if loss_name == 'L1Loss':
        return L1Loss()
    elif loss_name == 'DiceLoss':
        return DiceLoss()
    elif loss_name == 'CELoss':
        return CELoss()
    else:
        raise ValueError(f"Loss {loss_name} not implemented")
    
# ------------------------------- metrics --------------------------------

# reconstruction metric
class R2Metric(nn.Module):
    def __init__(self):
        super(R2Metric, self).__init__()
        self.r2 = R2Score()

    def forward(self, input, target):
        return self.r2(input, target)

class IoUScore(nn.Module):
    def __init__(self):
        super(IoUScore, self).__init__()

    def forward(self, tp, fp, fn, tn):
        return smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

class F1Score(nn.Module):
    def __init__(self):
        super(F1Score, self).__init__()

    def forward(self, tp, fp, fn, tn):
        return smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, tp, fp, fn, tn):
        return smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

# ssim from kornia
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, input, target):
        return kornia.metrics.SSIM(5)(input, target)

def load_metric(metric_name):
    if metric_name == 'IoUScore':
        return IoUScore()
    elif metric_name == 'F1Score':
        return F1Score()
    elif metric_name == 'Accuracy':
        return Accuracy()
    elif metric_name == 'R2Metric':
        return R2Metric()
    elif metric_name == 'SSIM':
        return SSIM()
    
    else:
        raise ValueError(f"Metric {metric_name} not implemented")
