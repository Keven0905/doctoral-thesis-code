import csv
from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .common import fingerprint, read_json, write_json


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def group_mapping(root, group_csv):
    if not group_csv:
        return {}
    mapping = {}
    with Path(group_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "group"}.issubset(reader.fieldnames or []):
            raise ValueError("groups.csv requires path,group columns.")
        for row in reader:
            path = (root / row["path"]).resolve()
            if not row["group"].strip() or path in mapping:
                raise ValueError("Empty group or duplicate group CSV path.")
            mapping[path] = row["group"].strip()
    return mapping


def scan_dataset(root, group_csv=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    present = [(root / name).is_dir() for name in SPLITS]
    if any(present) and not all(present):
        raise ValueError("Pre-split data requires train, val and test directories together.")
    explicit = all(present)
    parents = [root / name for name in SPLITS] if explicit else [root]
    classes = sorted({p.name for parent in parents for p in parent.iterdir() if p.is_dir()})
    if len(classes) < 2:
        raise ValueError("At least two class directories are required.")
    groups = group_mapping(root, group_csv)
    records = []
    seen = set()
    for parent in parents:
        for label, name in enumerate(classes):
            class_dir = parent / name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    raise ValueError(f"Duplicate image path: {path}")
                seen.add(resolved)
                if groups and resolved not in groups:
                    raise ValueError(f"Image absent from groups.csv: {path}")
                stat = path.stat()
                records.append({"path": path.relative_to(root).as_posix(), "label": label,
                                "group": groups.get(resolved, path.relative_to(root).as_posix()),
                                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                "split": parent.name if explicit else None})
    if not records:
        raise ValueError("No supported image files found.")
    if groups and seen != set(groups):
        raise ValueError("groups.csv contains paths outside the scanned image collection.")
    return root, classes, records, explicit, bool(groups)


def assign_splits(records, explicit, grouped, seed):
    if not explicit:
        units = defaultdict(lambda: defaultdict(list))
        for record in records:
            units[record["label"]][record["group"]].append(record)
        generator = np.random.default_rng(seed)
        for label, group_records in sorted(units.items()):
            keys = sorted(group_records)
            generator.shuffle(keys)
            if len(keys) < 3:
                raise ValueError(f"Class {label} requires at least 3 independent split units.")
            test_count = max(1, round(len(keys) * 0.20))
            val_count = max(1, round(len(keys) * 0.08))
            train_count = len(keys) - test_count - val_count
            if train_count < 1:
                raise ValueError(f"Class {label} cannot form three nonempty splits.")
            for position, key in enumerate(keys):
                split = "train" if position < train_count else "val"
                if position >= train_count + val_count:
                    split = "test"
                for record in group_records[key]:
                    record["split"] = split
    if grouped:
        membership = defaultdict(set)
        for record in records:
            membership[record["group"]].add(record["split"])
        if any(len(splits) > 1 for splits in membership.values()):
            raise ValueError("Original-image group crosses train/val/test boundaries.")
    for split in SPLITS:
        if not any(r["split"] == split for r in records):
            raise ValueError(f"Empty split: {split}")


def prepare_manifest(specification, output, seed):
    root, classes, records, explicit, grouped = scan_dataset(
        specification["root"], specification.get("groups_csv"))
    source_signature = fingerprint({"root": str(root), "classes": classes,
                                    "records": records, "seed": seed})
    path = Path(output) / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["source_signature"] != source_signature:
            raise ValueError("Dataset or split settings changed. Use a new output directory.")
        return manifest
    assign_splits(records, explicit, grouped, seed)
    counts = {split: sum(r["split"] == split for r in records) for split in SPLITS}
    if not grouped:
        warnings.warn("No original-image grouping supplied; using image-level splits.")
    manifest = {"root": str(root), "classes": classes,
                "class_to_idx": {name: i for i, name in enumerate(classes)},
                "records": records, "source_signature": source_signature,
                "split_policy": "existing" if explicit else "class_stratified_72_8_20",
                "original_image_grouping": grouped, "counts": counts, "seed": seed}
    manifest["signature"] = fingerprint(manifest)
    write_json(path, manifest)
    return manifest


class IndexedImages(Dataset):
    def __init__(self, manifest, split):
        self.root = Path(manifest["root"])
        self.records = [r for r in manifest["records"] if r["split"] == split]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record["label"], index


def loader(dataset, config, shuffle=False, epoch_seed=None):
    generator = torch.Generator().manual_seed(epoch_seed or config["seed"])
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=config["num_workers"], pin_memory=config["device"] == "cuda",
                      generator=generator)
