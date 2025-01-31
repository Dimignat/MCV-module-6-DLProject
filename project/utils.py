import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split


def load_config(path: Path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


config = load_config(Path("project/config.yaml"))


ROOT_DS_DIR = Path(config["ROOT_DS_DIR"])
TRAIN_IMAGES_DIR = Path(config["TRAIN_IMAGES_DIR"])
TRAIN_LABELS_DIR = Path(config["TRAIN_LABELS_DIR"])
VAL_IMAGES_DIR = Path(config["VAL_IMAGES_DIR"])
VAL_LABELS_DIR = Path(config["VAL_LABELS_DIR"])
TEST_IMAGES_DIR = Path(config["TEST_IMAGES_DIR"])
TRAIN_IMAGES_PATHS = sorted(TRAIN_IMAGES_DIR.glob("*"))
TRAIN_LABELS_PATHS = [TRAIN_LABELS_DIR / (p.stem + ".txt") for p in TRAIN_IMAGES_PATHS]
VAL_IMAGES_PATHS = sorted(VAL_IMAGES_DIR.glob("*"))
VAL_LABELS_PATHS = [VAL_LABELS_DIR / (p.stem + ".txt") for p in VAL_IMAGES_PATHS]
TEST_IMAGES_PATHS = sorted(TEST_IMAGES_DIR.glob("*"))

SEED = config["SEED"]
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

LABEL_MAP = config["LABEL_MAP"]
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def parse_label(path: Path) -> tuple[str, float, float, float, float]:
    with open(path) as f:
        line = f.readline()
    splt = line.split()
    label, xcenter, ycenter, w, h = tuple(map(float, splt))
    return int(label), xcenter, ycenter, w, h


def load_data() -> tuple[
    list[Image.Image],
    list[tuple[str, float, float, float, float]],
    list[Image.Image],
]:
    train_images = [Image.open(img_path) for img_path in TRAIN_IMAGES_PATHS]
    train_labels = [parse_label(label_path) for label_path in TRAIN_LABELS_PATHS]
    test_images = [Image.open(img_path) for img_path in TEST_IMAGES_PATHS]
    return train_images, train_labels, test_images


def resize(img: Image.Image, size: int) -> Image.Image:
    img_new = img.resize((size, size), Image.LANCZOS)
    return img_new


def copy_and_resize_ds(path: Path, size: int):
    new_root = list(path.parts)[0] + "-" + str(size)
    for i, path in enumerate(ROOT_DS_DIR.rglob("*")):
        if path.is_dir():
            continue

        parts = list(path.parts)
        parts[0] = new_root
        new_path = Path(*parts)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".jpeg":
            img = Image.open(path)
            img = resize(img, size)
            img.save(new_path)
        else:
            shutil.copy(path, new_path)


def split_ds(path: Path):
    train_labels = [parse_label(p)[0] for p in TRAIN_LABELS_PATHS]
    train_paths, val_paths, train_txt, val_txt = train_test_split(
        TRAIN_IMAGES_PATHS,
        TRAIN_LABELS_PATHS,
        stratify=train_labels,
        random_state=SEED,
        shuffle=True,
        test_size=0.2,
    )

    for img, txt in zip(val_paths, val_txt):
        parts = list(img.parts)
        parts = parts[:3] + ["val"] + parts[4:]
        new_img = Path(*parts)
        parts = list(txt.parts)
        parts = parts[:3] + ["val"] + parts[4:]
        new_txt = Path(*parts)

        new_img.parent.mkdir(parents=True, exist_ok=True)
        new_txt.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(img, new_img)
        shutil.move(txt, new_txt)
