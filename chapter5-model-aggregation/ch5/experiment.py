from pathlib import Path
import platform
import sys

import numpy as np
import torch
import torchvision
import sklearn
import matplotlib
import PIL

from .aggregation import aggregate, fuzzy_weights, update_difficulty
from .common import fingerprint, read_json, run_lock, write_csv, write_json
from .data import IndexedImages, prepare_manifest, SPLITS
from .engine import predict_checkpoint, train_model
from .metrics import classification_metrics, export_evaluation, plot_history
from .models import CHAIN, MODEL_NAMES, WEIGHTS


DEFAULT_TRAINING = {"learning_rate": 1e-4, "batch_size": 32, "epochs": 50,
                    "patience": 10, "seed": 42, "num_workers": 0,
                    "device": "auto", "pretrained": True, "difficulty_coefficient": 1.0,
                    "epsilon": 1e-12}


def resolved_config(configuration, classes):
    training = {**DEFAULT_TRAINING, **configuration.get("training", {})}
    unknown = set(training) - set(DEFAULT_TRAINING)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    if training["device"] == "auto":
        training["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if training["device"] not in ("cuda", "cpu"):
        raise ValueError("device must be auto, cuda or cpu.")
    if training["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    for key in ("batch_size", "epochs", "patience"):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if training["learning_rate"] <= 0 or training["difficulty_coefficient"] < 0:
        raise ValueError("Invalid learning rate or difficulty coefficient.")
    if not 0 < training["epsilon"] < 0.5 or training["num_workers"] < 0:
        raise ValueError("Invalid epsilon or worker count.")
    training["num_classes"] = len(classes)
    return training


class Experiment:
    def __init__(self, configuration, dataset_name, resume=False, evaluation=False):
        self.configuration = configuration
        self.name = dataset_name
        self.output = Path(configuration["output_root"]).resolve() / dataset_name
        self.resume = resume
        self.evaluation = evaluation

    def setup(self):
        self.manifest = prepare_manifest(self.configuration["datasets"][self.name], self.output,
                                        self.configuration.get("training", {}).get("seed", 42))
        self.classes = self.manifest["classes"]
        self.config = resolved_config(self.configuration, self.classes)
        self.datasets = {split: IndexedImages(self.manifest, split) for split in SPLITS}
        signature = fingerprint({"training": self.config, "manifest": self.manifest["signature"]})
        identity_path = self.output / "run.json"
        if identity_path.exists():
            if read_json(identity_path)["signature"] != signature:
                raise ValueError("Run settings changed. Select a new output_root.")
            if not self.resume and not self.evaluation:
                raise FileExistsError("Output already initialized. Pass --resume or use a new output_root.")
        else:
            if self.evaluation:
                raise FileNotFoundError("No initialized run exists for independent evaluation.")
            write_json(identity_path, {"signature": signature, "dataset": self.name,
                "training": self.config, "manifest_signature": self.manifest["signature"],
                "determinism": "fixed_seeds_cudnn_deterministic_torch_warn_only",
                "models": MODEL_NAMES, "pretrained_weights": WEIGHTS,
                "environment": {"python": sys.version, "platform": platform.platform(),
                    "torch": torch.__version__, "torchvision": torchvision.__version__,
                    "sklearn": sklearn.__version__, "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__, "pillow": PIL.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}})
        print(f"Dataset {self.name}: classes={len(self.classes)}, counts={self.manifest['counts']}, "
              f"device={self.config['device']}", flush=True)

    def context(self, **extra):
        return {"manifest": self.manifest["signature"], **extra}

    def record(self, model_id, probabilities, metadata):
        directory = self.output / "results" / model_id
        metrics = {split: export_evaluation(directory, split, self.datasets[split],
                                           probabilities[split], self.classes) for split in SPLITS}
        write_json(directory / "model.json", metadata)
        row = {"model": model_id, "train_accuracy": metrics["train"]["accuracy"],
               "train_loss": metrics["train"]["loss"], "test_accuracy": metrics["test"]["accuracy"],
               "test_loss": metrics["test"]["loss"], "precision": metrics["test"]["precision"],
               "recall": metrics["test"]["recall"], "f1": metrics["test"]["f1"],
               "auc": metrics["test"]["auc"]}
        write_json(directory / "summary.json", row)
        self.refresh_summary()

    def refresh_summary(self):
        rows = [read_json(p) for p in sorted((self.output / "results").glob("*/summary.json"))]
        write_csv(self.output / "summary.csv", rows)
        write_json(self.output / "summary.json", rows)

    def single(self, model_id):
        difficulty = np.full(len(self.datasets["train"]), 1.0 / len(self.datasets["train"]))
        probabilities, history = train_model(model_id, self.datasets, self.classes, self.config,
            self.output / "single" / model_id, difficulty, self.context(kind="single"), self.resume)
        self.record(model_id, probabilities, {"kind": "single", "model_id": model_id,
            "checkpoint": f"single/{model_id}/best.pt"})
        return probabilities, history

    def chain(self, ablation=False):
        chain_dir = self.output / "chain"
        chain_dir.mkdir(parents=True, exist_ok=True)
        difficulty = np.full(len(self.datasets["train"]), 1.0 / len(self.datasets["train"]))
        prior = {split: [] for split in SPLITS}
        statistics, stages, histories = [], [], []
        resnet, history = self.single("M2")
        for stage_index, model_id in enumerate(CHAIN):
            stage_name = f"{stage_index + 1}_{model_id}"
            checkpoint_path = f"chain/{stage_name}/best.pt"
            callback = self.ensemble_callback(prior, statistics, difficulty)
            if stage_index == 0:
                probabilities = resnet
                checkpoint_path = "single/M2/best.pt"
                history = [{**row, **{f"{kind}_{split}_{key}": row[f"{split}_{key}"]
                    for kind in ("M7", "M8") for split in ("train", "val")
                    for key in ("accuracy", "loss")}} for row in history]
            else:
                probabilities, history = train_model(model_id, self.datasets, self.classes, self.config,
                    chain_dir / stage_name, difficulty,
                    self.context(kind="chain", stage=stage_index + 1, prior=statistics),
                    self.resume, callback)
            labels = np.array([r["label"] for r in self.datasets["train"].records])
            next_difficulty, stats, hard = update_difficulty(probabilities["train"], labels, difficulty,
                self.config["difficulty_coefficient"], self.config["epsilon"])
            write_csv(chain_dir / f"{stage_name}_sample_weights.csv", [
                {"path": record["path"], "sample_index": i, "difficulty_before": float(difficulty[i]),
                 "difficulty_after": float(next_difficulty[i]), "hard": bool(hard[i]),
                 "error": float(1 - probabilities["train"][i, record["label"]])}
                for i, record in enumerate(self.datasets["train"].records)])
            statistics.append(stats)
            stages.append({"model_id": model_id, "checkpoint": checkpoint_path,
                           "statistics": stats, "best_epoch_metadata": str(Path(checkpoint_path).parent / "training.json")})
            write_json(chain_dir / "stages.json", stages)
            for split in SPLITS:
                prior[split].append(probabilities[split])
            difficulty = next_difficulty
            histories.append(history)
            if ablation:
                self.record_ensemble(f"A{stage_index + 1}", prior, stages, fuzzy=False)
            print(f"Stage {stage_index + 1} complete: hard={stats['hard_count']} rho={stats['rho']:.6f}", flush=True)
        self.record_ensemble("M7", prior, stages, fuzzy=False)
        self.record_ensemble("M8", prior, stages, fuzzy=True)
        if ablation:
            self.record_ensemble("A7", prior, stages, fuzzy=True)
        self.export_chain_curves(histories)

    def ensemble_callback(self, prior, statistics, difficulty):
        def callback(train_probabilities, val_probabilities):
            labels = np.array([r["label"] for r in self.datasets["train"].records])
            _, current, _ = update_difficulty(train_probabilities, labels, difficulty,
                self.config["difficulty_coefficient"], self.config["epsilon"])
            stats = statistics + [current]
            deterministic = [s["rho"] for s in stats]
            fuzzy, _ = fuzzy_weights([s["mean_error"] for s in stats], deterministic, self.config["epsilon"])
            row = {}
            for split, probability in (("train", train_probabilities), ("val", val_probabilities)):
                for kind, weights in (("M7", deterministic), ("M8", fuzzy)):
                    combined = aggregate(prior[split] + [probability], weights)
                    metrics = classification_metrics([r["label"] for r in self.datasets[split].records],
                                                     combined, self.classes)
                    row.update({f"{kind}_{split}_{key}": metrics[key] for key in ("accuracy", "loss")})
            return row
        return callback

    def record_ensemble(self, identifier, predictions, stages, fuzzy):
        deterministic = [s["statistics"]["rho"] for s in stages]
        weights = deterministic
        fuzzy_metadata = None
        if fuzzy:
            weights, fuzzy_metadata = fuzzy_weights(
                [s["statistics"]["mean_error"] for s in stages], deterministic, self.config["epsilon"])
        if len(stages) == 1:
            combined = {split: predictions[split][0] for split in SPLITS}
        else:
            combined = {split: aggregate(predictions[split], weights) for split in SPLITS}
        self.record(identifier, combined, {"kind": "fuzzy" if fuzzy else "heterogeneous",
            "stages": stages, "aggregation_weights": np.asarray(weights).tolist(),
            "fuzzy": fuzzy_metadata})

    def export_chain_curves(self, histories):
        boundaries, total = [], 0
        for history in histories[:-1]:
            total += len(history)
            boundaries.append(total + 0.5)
        for kind in ("M7", "M8"):
            trace = []
            for stage, history in enumerate(histories, start=1):
                for row in history:
                    trace.append({"step": len(trace) + 1, "stage": stage, "stage_epoch": row["step"],
                                  **{f"{split}_{key}": row[f"{kind}_{split}_{key}"]
                                     for split in ("train", "val") for key in ("accuracy", "loss")}})
            directory = self.output / "results" / kind
            write_csv(directory / "history.csv", trace)
            plot_history(trace, directory / "curves.png", boundaries)
            if kind == "M7":
                for index in range(1, 7):
                    ablation_dir = self.output / "results" / f"A{index}"
                    if ablation_dir.exists():
                        prefix = [row for row in trace if row["stage"] <= index]
                        write_csv(ablation_dir / "history.csv", prefix)
                        plot_history(prefix, ablation_dir / "curves.png", boundaries[:index - 1])
            elif (self.output / "results" / "A7").exists():
                ablation_dir = self.output / "results" / "A7"
                write_csv(ablation_dir / "history.csv", trace)
                plot_history(trace, ablation_dir / "curves.png", boundaries)

    def evaluate(self, identifier):
        metadata = read_json(self.output / "results" / identifier / "model.json")
        if metadata["kind"] == "single":
            predictions = predict_checkpoint(self.output / metadata["checkpoint"], self.datasets, self.config)
        else:
            weights = np.asarray(metadata["aggregation_weights"], dtype=np.float64)
            weights /= weights.sum()
            predictions = {split: np.zeros((len(data), len(self.classes)), dtype=np.float64)
                           for split, data in self.datasets.items()}
            for weight, stage in zip(weights, metadata["stages"]):
                current = predict_checkpoint(self.output / stage["checkpoint"], self.datasets, self.config)
                for split in SPLITS:
                    predictions[split] += weight * current[split]
            if len(metadata["stages"]) > 1:
                for split in SPLITS:
                    predictions[split] /= predictions[split].sum(axis=1, keepdims=True)
        self.record(identifier, predictions, metadata)


def execute(configuration, dataset_name, command, model=None, resume=False):
    experiment = Experiment(configuration, dataset_name, resume, evaluation=command == "evaluate")
    with run_lock(experiment.output):
        experiment.setup()
        if command == "single":
            experiment.single(model)
        elif command in ("ensemble", "ablation"):
            experiment.chain(ablation=command == "ablation")
        elif command == "all":
            for model_id in MODEL_NAMES:
                experiment.single(model_id)
            experiment.resume = True
            experiment.chain(ablation=dataset_name.upper() == "CMI5")
        elif command == "evaluate":
            experiment.evaluate(model)
        else:
            raise ValueError(f"Unsupported command {command}")
