from __future__ import annotations
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import load_config


def merge(config_path: str = "configs/train_config.yaml") -> None:
    cfg = load_config(config_path)
    dtype = torch.float16 if cfg["training"].get("fp16", True) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, cfg["output_dir"])
    merged = model.merge_and_unload()
    merged.save_pretrained(cfg["merged_output_dir"], safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.save_pretrained(cfg["merged_output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    merge(args.config)


if __name__ == "__main__":
    main()
