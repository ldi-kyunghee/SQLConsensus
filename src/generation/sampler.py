"""Generation step: greedy prediction, sampled queries and Verify for every question of a record file."""
from __future__ import annotations

from pathlib import Path

from src.generation.extraction import extract_final_sql, extract_reasoning_sql
from src.generation.openai_engine import OpenAIEngine
from src.generation.prompts import clause_check_messages, generation_messages
from src.generation.vllm_engine import VLLMEngine
from src.utils.io import jsonl_line, load_json, read_jsonl

DEFAULTS = dict(num_samples=5, temperature=1.0, top_p=0.95, max_tokens=4096, max_model_len=16384,
                tensor_parallel_size=1, seed=0, batch_size=64)


def _engine(backend: str, model: str, *, max_model_len: int, tensor_parallel_size: int):
    """Generation engine, SQL extractor and prompt style of a backend."""
    if backend == "vllm":
        return VLLMEngine(model, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size), \
            extract_reasoning_sql, False
    if backend == "openai":
        return OpenAIEngine(model), extract_final_sql, True
    raise ValueError(f"Unknown backend: {backend}")


def generate(data_path: Path, backend: str, model: str, out: Path, *, num_samples: int = 5,
             temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 4096, max_model_len: int = 16384,
             tensor_parallel_size: int = 1, seed: int = 0, batch_size: int = 64) -> None:
    """Write one line per question with the greedy prediction, the sampled queries, and Verify; resumable."""
    records = load_json(data_path)
    engine, extract, api = _engine(backend, model, max_model_len=max_model_len,
                                   tensor_parallel_size=tensor_parallel_size)
    done = {row["id"] for row in read_jsonl(out)} if Path(out).exists() else set()
    todo = [(i, r) for i, r in enumerate(records) if r["id"] not in done]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("a", encoding="utf-8") as stream:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            generated = engine.generate([generation_messages(r, api=api) for _, r in batch], num_samples=num_samples,
                                        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                        seeds=[seed + i for i, _ in batch])
            predictions = [extract(main) for main, _ in generated]
            checked = [(k, p) for k, p in enumerate(predictions) if p]
            verify = dict(zip([k for k, _ in checked],
                              engine.p_yes([clause_check_messages(batch[k][1], p) for k, p in checked])))
            for k, ((_, record), (main, samples)) in enumerate(zip(batch, generated)):
                stream.write(jsonl_line({"id": record["id"], "pred_sql": predictions[k],
                                         "samples": [extract(s) for s in samples], "verify": verify.get(k),
                                         "response": main, "sample_responses": samples}))
            stream.flush()
            print(f"{min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
