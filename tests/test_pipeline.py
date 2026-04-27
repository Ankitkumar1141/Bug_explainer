from pathlib import Path
from bug_explainer import config, data


def test_data_pipeline():
    root = Path(__file__).resolve().parents[1]  # ✅ ADD THIS LINE

    cfg = config.load_config(root / "configs/train_config.yaml")

    rows = data.load_error_rows(root / cfg["data"]["path"])
    dataset = data.build_sft_dataset(rows)

    assert len(rows) > 0
    assert len(dataset) > 0