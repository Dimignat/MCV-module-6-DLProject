from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from .utils import parse_label


class FasterRCNNDataset(Dataset):
    def __init__(self, images_paths: list[Path], labels_paths: list[Path] | None):
        self.images_paths = images_paths
        self.labels_paths = labels_paths
        self.train = bool(labels_paths)
        self.size = 224
        self.train_transforms = v2.Compose(
            [
                v2.ConvertBoundingBoxFormat("XYXY"),
                v2.PILToTensor(),
                v2.Resize(size=(self.size, self.size), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test_transforms = v2.Compose(
            [
                v2.ConvertBoundingBoxFormat("XYXY"),
                v2.PILToTensor(),
                v2.Resize(size=(self.size, self.size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx) -> dict[str, torch.Tensor | None]:
        img = Image.open(self.images_paths[idx])
        if self.train:
            klass, *bbox = parse_label(self.labels_paths[idx])
            img_w, img_h = img.size
            x, y = bbox[0] * img_w, bbox[1] * img_h
            w, h = bbox[2] * img_w, bbox[3] * img_h

            bbox = tv_tensors.BoundingBoxes(
                [x, y, w, h],
                format="CXCYWH",
                dtype=torch.float32,
                canvas_size=(img_h, img_w),
            )
            img, bbox = self.train_transforms(img, bbox)
            klass = torch.tensor([klass])
        else:
            klass, bbox = None, None
            img = self.test_transforms(img)

        return {"klass": klass, "bbox": bbox, "img": img}
