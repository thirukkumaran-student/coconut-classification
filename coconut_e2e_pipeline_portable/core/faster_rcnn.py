import torch

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)


def load_fasterrcnn(path, device):

    device_obj = torch.device(device)

    # Create Faster R-CNN architecture
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    # Number of classes:
    # 0 = background
    # 1 = coconut
    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            2
        )
    )

    # Load checkpoint
    checkpoint = torch.load(
        path,
        map_location=device_obj
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        checkpoint = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(
        checkpoint,
        strict=False
    )

    model.to(device_obj)

    model.eval()

    return model