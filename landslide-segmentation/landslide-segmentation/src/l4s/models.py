"""Model factories (torch / smp imported lazily)."""


def build_segmentation_model(arch, in_channels=14, encoder=None, encoder_weights=None, classes=1):
    import segmentation_models_pytorch as smp
    if arch == "unet":
        # plain U-Net-like: smp Unet with a light encoder trained from scratch
        return smp.Unet(encoder_name=encoder or "resnet18", encoder_weights=None,
                        in_channels=in_channels, classes=classes)
    if arch == "resunet":
        return smp.Unet(encoder_name=encoder or "resnet34", encoder_weights=encoder_weights,
                        in_channels=in_channels, classes=classes)
    raise ValueError(arch)


def dice_bce_loss():
    import torch
    import torch.nn.functional as F

    def loss(logits, target, eps=1.0):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = 1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
        return bce + dice.mean()
    return loss


def build_baseline(name, params):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    raise ValueError(name)
