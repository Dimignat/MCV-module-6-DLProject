import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from ultralytics import YOLO

from .datasets import FasterRCNNDataset
from .models import get_fasterrcnn, get_yolo
from .utils import (INV_LABEL_MAP, TEST_IMAGES_PATHS, TRAIN_EPOCHS,
                    TRAIN_IMAGES_PATHS, TRAIN_LABELS_PATHS, VAL_IMAGES_PATHS,
                    VAL_LABELS_PATHS)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["train", "test"])
parser.add_argument("--model", type=str, choices=["fasterrcnn", "yolo"])
parser.add_argument("--checkp-path", type=Path)
args = parser.parse_args()


class MosquitoDetector(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        imgs = []
        targets = []
        for d in batch:
            imgs.append(d["img"].to(self.device))
            target = {}
            target["boxes"] = d["bbox"].to(self.device)
            target["labels"] = d["klass"].to(self.device)
            targets.append(target)
        loss_dict = self.model(imgs, targets)
        loss = sum(v for v in loss_dict.values())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = []
        targets = []
        for d in batch:
            imgs.append(d["img"].to(self.device))
            target = {}
            target["boxes"] = d["bbox"].to(self.device)
            target["labels"] = d["klass"].to(self.device)
            targets.append(target)
        self.model.train()
        loss_dict = self.model(imgs, targets)
        self.model.eval()
        loss = sum(v for v in loss_dict.values())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer


def _train_fastercnn():
    model = get_fasterrcnn()

    train_ds = FasterRCNNDataset(TRAIN_IMAGES_PATHS, TRAIN_LABELS_PATHS)
    val_ds = FasterRCNNDataset(VAL_IMAGES_PATHS, VAL_LABELS_PATHS)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda x: x)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=True, collate_fn=lambda x: x)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    mosquito_detector = MosquitoDetector(model, optimizer)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        callbacks=[early_stopping], max_epochs=TRAIN_EPOCHS, accelerator=accelerator
    )
    trainer.fit(mosquito_detector, train_dl, val_dl)


def _test_fastercnn(checkp_path: Path):
    test_ds = FasterRCNNDataset(TEST_IMAGES_PATHS)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=True, collate_fn=lambda x: x)
    model = MosquitoDetector.load_from_checkpoint(checkp_path)
    predictions = model.eval(test_dl)

    data = []
    for pred in predictions:
        x1, y1, x2, y2 = pred["boxes"][0].cpu().numpy()
        res = {
            "LabelName": INV_LABEL_MAP[int(pred["labels"][0].cpu().numpy())],
            "Conf": float(pred["scores"][0].cpu().numpy()),
            "xcenter": float((x1 + x2) / 2) / 224,
            "ycenter": float((y1 + y2) / 2) / 224,
            "bbx_width": float(x2 - x1) / 224,
            "bbx_height": float(y2 - y1) / 224,
        }
        data.append(res)
    df = pd.DataFrame(data)
    sample = pd.read_csv("dlp-object-detection-224/sample_submission.csv")
    sample[["id", "ImageID"]].merge(df, on="ImageID").to_csv(
        "fasterrcnn-res.csv", index=False
    )


def _train_yolo():
    model = get_yolo()
    model.train(
        data="dlp-object-detection-640/yolo-ds.yaml", epochs=TRAIN_EPOCHS, imgsz=640
    )


def _test_yolo(checkp_path: Path):
    model = YOLO(checkp_path)
    results = model.predict(
        "dlp-object-detection-640/final_dlp_data/final_dlp_data/test/images",
        max_det=1,
        conf=0,
    )

    data = []
    for res in results:
        x, y, w, h = tuple(res.boxes.xywhn.cpu().numpy().tolist()[0])
        data.append(
            {
                "ImageID": Path(res.path).name,
                "LabelName": res.names[int(res.boxes.cls.cpu().numpy()[0])],
                "Conf": res.boxes.conf.cpu().numpy()[0],
                "xcenter": x,
                "ycenter": y,
                "bbx_width": w,
                "bbx_height": h,
            }
        )
    df = pd.DataFrame(data)
    sample = pd.read_csv("dlp-object-detection-640/sample_submission.csv")
    sample[["id", "ImageID"]].merge(df, on="ImageID").to_csv(
        "yolo-res.csv", index=False
    )


def train(model_type: str):
    if model_type == "fasterrcnn":
        _train_fastercnn()
    elif model_type == "yolo":
        _train_yolo()
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


def test(model_type: str, checkp_path: Path):
    if model_type == "fasterrcnn":
        _test_fastercnn(checkp_path)
    elif model_type == "yolo":
        _test_yolo(checkp_path)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


if __name__ == "__main__":
    if args.mode == "train":
        train(args.model)
    else:
        test(args.model, args.checkp_path)
