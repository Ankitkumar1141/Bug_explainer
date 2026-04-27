import json
from pathlib import Path

from datasets import Dataset


PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}
"""

INFER_TEMPLATE = """### Instruction:
{instruction}

### Response:
"""


def load_error_rows(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("errors.json must contain a list of objects")

    for i, row in enumerate(rows):
        if "instruction" not in row or "output" not in row:
            raise ValueError(
                f"Row {i} must contain 'instruction' and 'output' keys"
            )

    return rows


def build_sft_dataset(rows: list[dict]) -> Dataset:
    texts = []

    for row in rows:
        text = PROMPT_TEMPLATE.format(
            instruction=row["instruction"],
            output=row["output"],
        )
        texts.append({"text": text})

    return Dataset.from_list(texts)


if __name__ == "__main__":
    rows = load_error_rows("data/errors.json")
    dataset = build_sft_dataset(rows)

    print(f"Rows loaded: {len(rows)}")
    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])