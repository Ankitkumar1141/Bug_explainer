# Bug Explainer — End-to-End Project Template

This project turns your notebook into a maintainable ML application. It fine-tunes `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with LoRA so it can explain programming errors in beginner-friendly language.

## Project structure

```text
bug_explainer_project/
├── app/
│   ├── api.py                # FastAPI inference service
│   └── streamlit_app.py      # Simple UI
├── configs/
│   └── train_config.yaml     # Model, LoRA, training, generation settings
├── data/
│   └── bug_errors.jsonl      # 73 training examples from your notebook
├── scripts/
│   ├── train.sh
│   ├── merge.sh
│   └── run_api.sh
├── src/bug_explainer/
│   ├── config.py
│   ├── data.py
│   ├── train.py
│   ├── merge.py
│   └── infer.py
├── tests/
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Train the LoRA adapter

```bash
bash scripts/train.sh
```

The adapter is saved to `outputs/bug_explainer_adapter`.

## Merge adapter into a final model

```bash
bash scripts/merge.sh
```

The merged model is saved to `outputs/bug_explainer_final`.

## Run inference from CLI

```bash
python -m bug_explainer.infer "NameError: name 'x' is not defined" --model-path outputs/bug_explainer_final
```

## Run the API

```bash
bash scripts/run_api.sh
```

Then call:

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"error":"IndexError: list index out of range"}'
```

## Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Docker

```bash
docker build -t bug-explainer .
docker run -p 8000:8000 bug-explainer
```

## Notes

- Training a 1.1B model is much easier with a CUDA GPU.
- Keep adding more real error messages to `data/bug_errors.jsonl` for better generalization.
- The config file is the main place to adjust model, LoRA, batch size, epochs, and generation settings.
