from __future__ import annotations
import argparse
import torch
from transformers import pipeline, AutoTokenizer
from .config import load_config
from .data import INFER_TEMPLATE


def load_pipeline(model_path: str, base_model: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path if base_model is None else base_model)
    tokenizer.pad_token = tokenizer.eos_token
    return pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )


def explain_error(error_message: str, model_path: str = "outputs/bug_explainer_final", config_path: str = "configs/train_config.yaml") -> str:
    cfg = load_config(config_path)
    pipe = load_pipeline(model_path)
    prompt = INFER_TEMPLATE.format(error_message=error_message)
    result = pipe(
        prompt,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        do_sample=True,
        temperature=cfg["generation"]["temperature"],
        repetition_penalty=cfg["generation"]["repetition_penalty"],
    )
    return result[0]["generated_text"].split("[/INST]")[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("error")
    parser.add_argument("--model-path", default="outputs/bug_explainer_final")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    print(explain_error(args.error, args.model_path, args.config))


if __name__ == "__main__":
    main()
