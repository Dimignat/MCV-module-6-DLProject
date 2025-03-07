from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

from .utils import LABEL_MAP


def get_fasterrcnn(pretrained: bool = True):
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, trainable_backbone_layers=0)
    num_classes = len(LABEL_MAP)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_yolo():
    return YOLO("yolo11m.pt")
