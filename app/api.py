from __future__ import annotations
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
from bug_explainer.infer import load_pipeline
from bug_explainer.data import INFER_TEMPLATE

MODEL_PATH = "outputs/bug_explainer_final"

app = FastAPI(title="Bug Explainer API", version="0.1.0")

class ExplainRequest(BaseModel):
    error: str

@lru_cache(maxsize=1)
def get_pipe():
    return load_pipeline(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/explain")
def explain(req: ExplainRequest):
    pipe = get_pipe()
    prompt = INFER_TEMPLATE.format(error_message=req.error)
    result = pipe(prompt, max_new_tokens=220, do_sample=True, temperature=0.3, repetition_penalty=1.4)
    answer = result[0]["generated_text"].split("[/INST]")[-1].strip()
    return {"error": req.error, "explanation": answer}
