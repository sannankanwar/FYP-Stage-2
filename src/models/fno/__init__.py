"""FNO Models Package"""
from src.models.fno.fno_resnet18 import FNOResNet18
from src.models.fno.fno_resnet50 import FNOResNet50
from src.models.fno.fno_vgg19 import FNOVGG19
from src.models.fno.fno_unet import FNOUNet

__all__ = ['FNOResNet18', 'FNOResNet50', 'FNOVGG19', 'FNOUNet']
