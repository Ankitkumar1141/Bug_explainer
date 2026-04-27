from bug_explainer import config, data


def test_data_pipeline():
    cfg = config.load_config("configs/train_config.yaml")

    rows = data.load_error_rows(cfg["data_path"])
    dataset = data.build_sft_dataset(rows)

    assert rows is not None
    assert dataset is not None
    assert len(rows) > 0
    assert len(dataset) > 0
    assert "text" in dataset[0]