
import argparse
import json
from pathlib import Path
import re
from types import SimpleNamespace

import torch

from model import MLPParamNet
from trainer import Trainer


DEFAULT_RUN_CONFIG = {
    "percentage": 0.01,
    "epochs": 20,
    "lr": 1e-3,
    "scorer": "lhuman",
    "device": "auto",
    "seed": 42,
    "log_every": 10,
    "z_dim": 64,
    "hidden": 512,
    "log_dir": "logs",
    "mobius_layers": 1,
    "exp_name": "runs",
    "vlm_model": "qwen",
    "img_size": 300,
    "param_limits": (0.2, 0.2, 0.4, 0.5),
}


def image_lookup_key(name):
    stem = Path(name).stem if name else ""
    digits = re.findall(r"\d+", stem.lower())
    if digits:
        return f"n:{int(''.join(digits))}"

    compact = re.sub(r"[^a-z0-9]+", "", stem.lower())
    return f"s:{compact}"


def get_questions_answers(image_name, json_path=None):
    if not json_path:
        return [], []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_key = image_lookup_key(image_name)
    questions = []
    answers = []
    for item in data:
        if image_lookup_key(item["image_name"]) == target_key:
            questions.append(item["question"])
            answers.append(item["answers"])
    return questions, answers


def build_args(
    image_path,
    *,
    questions=None,
    answers=None,
    json_path=None,
    log_dir=None,
    exp_name=None,
    overrides=None,
):
    config = dict(DEFAULT_RUN_CONFIG)
    if overrides:
        config.update(overrides)

    image_name = Path(image_path).name
    if questions is None or answers is None:
        auto_questions, auto_answers = get_questions_answers(image_name, json_path=json_path)
        if questions is None:
            questions = auto_questions
        if answers is None:
            answers = auto_answers

    return SimpleNamespace(
        image_path=str(image_path),
        percentage=config["percentage"],
        epochs=config["epochs"],
        lr=config["lr"],
        scorer=config["scorer"],
        device=config["device"],
        seed=config["seed"],
        log_every=config["log_every"],
        z_dim=config["z_dim"],
        hidden=config["hidden"],
        log_dir=log_dir or config["log_dir"],
        mobius_layers=config["mobius_layers"],
        exp_name=exp_name or config["exp_name"],
        json_path=json_path,
        vlm_model=config["vlm_model"],
        img_size=config["img_size"],
        param_limits=config["param_limits"],
        questions=questions,
        answers=answers,
    )


def run_llmind_inference(
    image_path,
    *,
    questions=None,
    answers=None,
    json_path=None,
    log_dir=None,
    exp_name=None,
    overrides=None,
    progress_callback=None,
):
    args = build_args(
        image_path=image_path,
        questions=questions,
        answers=answers,
        json_path=json_path,
        log_dir=log_dir,
        exp_name=exp_name,
        overrides=overrides,
    )

    torch.manual_seed(args.seed)
    model = MLPParamNet(
        z_dim=args.z_dim,
        hidden=args.hidden,
        mobius_layers=args.mobius_layers,
        param_limits=args.param_limits,
    )
    trainer = Trainer(args, model, progress_callback=progress_callback)
    run_outputs = trainer.run()

    with open(run_outputs["results_path"], "r", encoding="utf-8") as f:
        results = json.load(f)

    run_outputs["results"] = results
    return run_outputs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--percentage", type=float, default=DEFAULT_RUN_CONFIG["percentage"])
    p.add_argument("--epochs", type=int, default=DEFAULT_RUN_CONFIG["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULT_RUN_CONFIG["lr"])
    p.add_argument("--scorer", type=str, default=DEFAULT_RUN_CONFIG["scorer"])
    p.add_argument("--device", type=str, default=DEFAULT_RUN_CONFIG["device"])
    p.add_argument("--seed", type=int, default=DEFAULT_RUN_CONFIG["seed"])
    p.add_argument("--log_every", type=int, default=DEFAULT_RUN_CONFIG["log_every"])
    p.add_argument("--z_dim", type=int, default=DEFAULT_RUN_CONFIG["z_dim"])
    p.add_argument("--hidden", type=int, default=DEFAULT_RUN_CONFIG["hidden"])
    p.add_argument("--log_dir", type=str, default=DEFAULT_RUN_CONFIG["log_dir"])
    p.add_argument("--mobius_layers", type=int, default=DEFAULT_RUN_CONFIG["mobius_layers"])
    p.add_argument("--exp_name", type=str, default=DEFAULT_RUN_CONFIG["exp_name"])
    p.add_argument("--json_path", type=str, default=None)
    p.add_argument(
        "--vlm_model",
        type=str,
        default=DEFAULT_RUN_CONFIG["vlm_model"],
        help="VLM model for question answering. Supported values currently map to Qwen2.5-VL.",
    )
    p.add_argument("--img_size", type=int, default=DEFAULT_RUN_CONFIG["img_size"], help="image size for the VLM model")
    p.add_argument(
        "--param_limits",
        type=float,
        nargs=4,
        default=DEFAULT_RUN_CONFIG["param_limits"],
        metavar=("A", "B", "C", "D"),
    )

    args = p.parse_args()
    questions, answers = get_questions_answers(Path(args.image_path).name, json_path=args.json_path)
    args.questions = questions
    args.answers = answers
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model = MLPParamNet(
        z_dim=args.z_dim,
        hidden=args.hidden,
        mobius_layers=args.mobius_layers,
        param_limits=args.param_limits,
    )
    Trainer(args, model).run()


if __name__ == "__main__":
    main()
