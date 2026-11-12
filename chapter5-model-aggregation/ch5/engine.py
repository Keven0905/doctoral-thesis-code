import gc
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as functional

from .common import (atomic_write, fingerprint, random_state, restore_random_state,
                     seed_everything, write_csv, write_json)
from .data import loader
from .metrics import classification_metrics, plot_history
from .models import build_model, WEIGHTS


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(path, value):
    atomic_write(path, lambda temporary: torch.save(value, temporary))


def cpu_model_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@torch.no_grad()
def predict(model, dataset, config):
    model.eval()
    probabilities = np.empty((len(dataset), config["num_classes"]), dtype=np.float64)
    for images, labels, indices in loader(dataset, config):
        logits = model(images.to(config["device"], non_blocking=True))
        probability = logits.double().softmax(dim=1).cpu().numpy()
        probabilities[indices.numpy()] = probability
    return probabilities


def train_epoch(model, optimizer, dataset, config, difficulty, epoch_seed):
    model.train()
    weight_tensor = torch.as_tensor(difficulty * len(dataset), dtype=torch.float32,
                                    device=config["device"])
    total = 0.0
    for images, labels, indices in loader(dataset, config, shuffle=True, epoch_seed=epoch_seed):
        images = images.to(config["device"], non_blocking=True)
        labels = labels.to(config["device"], non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss_per_sample = functional.cross_entropy(model(images), labels, reduction="none")
        loss = (loss_per_sample * weight_tensor[indices.to(config["device"])]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(labels)
    return total / len(dataset)


def train_model(model_id, datasets, classes, config, directory, difficulty,
                context, resume=False, ensemble_callback=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"model": model_id, "classes": classes, "config": config,
                             "difficulty": difficulty.tolist(), "context": context})
    last_path, best_path = directory / "last.pt", directory / "best.pt"
    checkpoint = None
    if last_path.exists():
        if not resume:
            raise FileExistsError(f"Existing training state at {directory}. Use --resume or a new output.")
        checkpoint = load_checkpoint(last_path)
        if checkpoint["signature"] != signature:
            raise ValueError(f"Training configuration changed at {directory}.")
    seed = config["seed"] + int(model_id[1:])
    seed_everything(seed)
    model = build_model(model_id, len(classes), pretrained=config["pretrained"] and checkpoint is None)
    model.to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, best_loss, best_epoch, bad_epochs, start_epoch = [], float("inf"), 0, 0, 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(config["device"])
        history = checkpoint["history"]
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
        bad_epochs = checkpoint["bad_epochs"]
        start_epoch = checkpoint["epoch"] + 1
        restore_random_state(checkpoint["random_state"])
        best = load_checkpoint(best_path) if best_path.exists() else None
        if best is None or best["epoch"] != best_epoch:
            save_checkpoint(best_path, {"model": checkpoint["best_model"], "epoch": best_epoch,
                                       "signature": signature, "model_id": model_id,
                                       "classes": classes, "config": config})
        del best
    best_model = checkpoint["best_model"] if checkpoint is not None else None
    already_finished = checkpoint is not None and checkpoint["finished"]
    del checkpoint
    for epoch in range(start_epoch, config["epochs"] + 1):
        if already_finished:
            break
        started = time.monotonic()
        objective = train_epoch(model, optimizer, datasets["train"], config, difficulty, seed + epoch)
        train_probabilities = predict(model, datasets["train"], config)
        val_probabilities = predict(model, datasets["val"], config)
        train_metrics = classification_metrics(
            [r["label"] for r in datasets["train"].records], train_probabilities, classes)
        val_metrics = classification_metrics(
            [r["label"] for r in datasets["val"].records], val_probabilities, classes)
        row = {"step": epoch, "train_objective": objective,
               "train_accuracy": train_metrics["accuracy"], "train_loss": train_metrics["loss"],
               "val_accuracy": val_metrics["accuracy"], "val_loss": val_metrics["loss"]}
        if ensemble_callback:
            row.update(ensemble_callback(train_probabilities, val_probabilities))
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch, bad_epochs = val_metrics["loss"], epoch, 0
            best_model = cpu_model_state(model)
            save_checkpoint(best_path, {"model": best_model, "epoch": epoch,
                                       "signature": signature, "model_id": model_id,
                                       "classes": classes, "config": config})
        else:
            bad_epochs += 1
        finished = bad_epochs >= config["patience"] or epoch == config["epochs"]
        save_checkpoint(last_path, {"signature": signature, "model_id": model_id,
            "classes": classes, "config": config, "context": context,
            "weights_source": WEIGHTS[model_id] if config["pretrained"] else "random",
            "epoch": epoch, "model": cpu_model_state(model), "best_model": best_model,
            "optimizer": optimizer.state_dict(), "random_state": random_state(),
            "best_loss": best_loss, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "history": history, "finished": finished})
        print(f"{directory.name} epoch {epoch}/{config['epochs']} "
              f"train_acc={train_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.6f} "
              f"seconds={time.monotonic() - started:.1f}", flush=True)
        if finished:
            break
    write_csv(directory / "history.csv", history)
    plot_history(history, directory / "curves.png")
    write_json(directory / "training.json", {"best_epoch": best_epoch, "best_val_loss": best_loss,
               "epochs_completed": len(history), "signature": signature, "model_id": model_id,
               "weights_source": WEIGHTS[model_id] if config["pretrained"] else "random"})
    model.load_state_dict(load_checkpoint(best_path)["model"])
    del optimizer, best_model
    predictions = {split: predict(model, data, config) for split, data in datasets.items()}
    del model
    gc.collect()
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    return predictions, history


def predict_checkpoint(path, datasets, config):
    checkpoint = load_checkpoint(path)
    model = build_model(checkpoint["model_id"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(config["device"])
    predictions = {split: predict(model, data, config) for split, data in datasets.items()}
    del model, checkpoint
    gc.collect()
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    return predictions
