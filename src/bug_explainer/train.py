from __future__ import annotations
import argparse
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from .config import load_config
from .data import build_sft_dataset


def train(config_path: str = "configs/train_config.yaml") -> None:
    cfg = load_config(config_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if cfg["training"].get("fp16", True) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=dtype,
        device_map="auto",
    )

    lora_cfg = cfg["lora"]
    lora = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )

    dataset = build_sft_dataset(cfg["data_path"])
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(output_dir=cfg["output_dir"], **cfg["training"]),
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
