from bug_explainer.data import load_error_rows, build_sft_dataset


def test_load_error_rows():
    rows = load_error_rows("data/bug_errors.jsonl")
    assert len(rows) > 0
    assert {"instruction", "output"}.issubset(rows[0])


def test_build_sft_dataset():
    ds = build_sft_dataset("data/bug_errors.jsonl")
    assert "[INST]" in ds[0]["text"]
