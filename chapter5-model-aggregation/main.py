import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path

from ch5.common import read_json
from ch5.experiment import execute


def main():
    parser = argparse.ArgumentParser(description="Thesis Chapter 5 training and evaluation")
    parser.add_argument("command", choices=["single", "ensemble", "ablation", "all", "evaluate", "batch"])
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "experiment.json")
    parser.add_argument("--dataset", default="CMI5")
    parser.add_argument("--model", choices=[f"M{i}" for i in range(1, 9)] + [f"A{i}" for i in range(1, 8)])
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    configuration = read_json(arguments.config)
    base = arguments.config.resolve().parent
    configuration["output_root"] = str((base / configuration["output_root"]).resolve())
    for spec in configuration["datasets"].values():
        spec["root"] = str((base / spec["root"]).resolve())
        if spec.get("groups_csv"):
            spec["groups_csv"] = str((base / spec["groups_csv"]).resolve())
    if arguments.command == "single" and arguments.model not in [f"M{i}" for i in range(1, 7)]:
        parser.error("single requires --model M1 through M6")
    if arguments.command == "evaluate" and arguments.model is None:
        parser.error("evaluate requires --model")
    if arguments.command == "ablation" and arguments.dataset.upper() != "CMI5":
        parser.error("The approved ablation dataset is CMI5.")
    names = list(configuration["datasets"]) if arguments.command == "batch" else [arguments.dataset]
    for name in names:
        if name not in configuration["datasets"]:
            parser.error(f"Unknown dataset {name}")
        execute(configuration, name, "all" if arguments.command == "batch" else arguments.command,
                arguments.model, arguments.resume)


if __name__ == "__main__":
    main()
