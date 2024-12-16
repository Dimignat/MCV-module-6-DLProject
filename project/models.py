from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

LABEL_MAP = {
    "aegypti": 0,
    "albopictus": 1,
    "anopheles": 2,
    "culex": 3,
    "culiseta": 4,
    "japonicus/koreicus": 5,
    "None": 6,
}


def get_fasterrcnn():
    model = fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=0)
    num_classes = len(LABEL_MAP)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_yolo():
    return YOLO("yolo11m.pt")
